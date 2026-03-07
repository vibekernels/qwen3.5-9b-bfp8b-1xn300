// SPDX-License-Identifier: Apache-2.0
// Qwen3.5-9B inference engine for Tenstorrent N300 via tt-metal.
//
// Matrix multiplications run on-device via ttnn::matmul on Tensix cores.
// Small element-wise ops (RoPE, gating, SSM recurrence) remain on host CPU.

// ttnn headers FIRST (they pull in tt-metalium internally with correct include order)
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/types.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/data_movement/slice/slice.hpp>
#include <ttnn/operations/normalization/rmsnorm/rmsnorm.hpp>
#include <ttnn/operations/trace.hpp>
#include <ttnn/types.hpp>

#include "engine.h"
#include "model_config.h"
#include "gguf_loader.h"
#include "tokenizer.h"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/device.hpp>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <chrono>
#include <span>
#include <map>
#include <tuple>

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using MC = ModelConfig;

// ============================================================================
// Global state
// ============================================================================
static std::shared_ptr<MeshDevice> g_mesh;   // chip 0: layer weights
static std::shared_ptr<MeshDevice> g_mesh2;  // chip 1: wo, ssm_out, LM head
static ModelBuffers g_model;
static Tokenizer g_tokenizer;
static bool g_loaded = false;
static int g_max_ctx = 0;
static int g_pos = 0;

// Host-side f32 hidden state
static std::vector<float> g_hidden_f32(MC::n_embd);

// KV cache for attention layers [8 layers][max_ctx * kv_dim]
static constexpr int kv_dim = MC::n_head_kv * MC::head_dim;  // 1024
static std::vector<float> g_k_cache[8];
static std::vector<float> g_v_cache[8];

// SSM recurrent state [24 layers]
static constexpr int ssm_n_v_heads = MC::ssm_dt_rank;       // 32
static constexpr int ssm_head_k_dim = MC::ssm_d_state;      // 128
static constexpr int ssm_head_v_dim_c = MC::ssm_head_v_dim; // 128
static std::vector<float> g_ssm_state[24];

// Conv1d state [24 layers]
static constexpr int conv_state_len = MC::ssm_conv_kernel - 1;  // 3
static std::vector<float> g_conv_state[24];

// ============================================================================
// Cached weight Tensors (wrapping on-device MeshBuffers for ttnn::matmul)
// ============================================================================
struct WeightTensors {
    // Chip 0: Attention layers (8)
    Tensor attn_wqkv[8];
    Tensor attn_ffn_gate_up[8];
    Tensor attn_ffn_down[8];

    // Chip 0: SSM layers (24)
    Tensor ssm_w_combined[24];
    Tensor ssm_ffn_gate_up[24];
    Tensor ssm_ffn_down[24];

    // Chip 1: Output projections + LM head
    Tensor attn_wo[8];           // [n_embd, n_head*head_dim] on chip 1
    Tensor ssm_out[24];          // [n_embd, ssm_d_inner] on chip 1
    Tensor lm_head;              // [n_vocab, n_embd] on chip 1

    // Norm weights as device tensors (for on-device RMSNorm)
    Tensor attn_norm[32];        // pre-attention norm for each layer
    Tensor post_norm[32];        // post-attention norm for each layer
    Tensor output_norm_tensor;   // final output norm
};
static WeightTensors g_wt;

// Persistent hidden state on device (avoids PCIe round-trips between layers)
static std::shared_ptr<MeshBuffer> g_hidden_dev_buf;
static Tensor g_hidden_dev;
static std::shared_ptr<MeshBuffer> g_residual_dev_buf;
static Tensor g_residual_dev;
// Temp buffer for matmul input after rms_norm (on device)
static std::shared_ptr<MeshBuffer> g_norm_dev_buf;
static Tensor g_norm_dev;

// Wrap an existing on-device MeshBuffer as a ttnn Tensor for use with ttnn::matmul
static Tensor wrap_weight(std::shared_ptr<MeshBuffer> buf, uint32_t M, uint32_t K) {
    DeviceStorage storage(buf, {MeshCoordinate(0, 0)});
    TensorSpec spec(
        Shape({1, 1, M, K}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));
    TensorTopology topology;
    return Tensor(std::move(storage), spec, topology);
}

// ============================================================================
// Pre-allocated device buffers for fast GEMV (avoids per-call alloc/dealloc)
// ============================================================================
struct GemvBuf {
    std::shared_ptr<MeshBuffer> act_buf;    // device-side activation buffer
    Tensor act_tensor;                       // wraps act_buf
    std::vector<bfloat16> act_host_tiled;   // pre-allocated host tilized buffer
    uint32_t K_padded;

    std::shared_ptr<MeshBuffer> out_buf;    // device-side output buffer
    Tensor out_tensor;                       // wraps out_buf
    std::vector<bfloat16> out_host_tiled;   // pre-allocated host read buffer
    uint32_t M_padded;
};

// Map from (device_ptr, K, M) -> pre-allocated buffers
static std::map<std::tuple<MeshDevice*, uint32_t, uint32_t>, GemvBuf> g_gemv_bufs;

static GemvBuf& get_gemv_buf(MeshDevice* device, uint32_t M, uint32_t K) {
    auto key = std::make_tuple(device, K, M);
    auto it = g_gemv_bufs.find(key);
    if (it != g_gemv_bufs.end()) return it->second;

    // Create new pre-allocated buffers
    GemvBuf buf;
    uint32_t TW = TILE_WIDTH, TH = TILE_HEIGHT;
    buf.K_padded = ((K + TW - 1) / TW) * TW;
    buf.M_padded = ((M + TW - 1) / TW) * TW;

    uint32_t act_tiles = buf.K_padded / TW;
    uint32_t out_tiles = buf.M_padded / TW;
    uint32_t tile_bytes = TH * TW * sizeof(bfloat16);

    DeviceLocalBufferConfig dram_cfg{.page_size = tile_bytes, .buffer_type = BufferType::DRAM};

    buf.act_buf = MeshBuffer::create(ReplicatedBufferConfig{.size = act_tiles * tile_bytes},
                                      dram_cfg, device);
    buf.act_host_tiled.resize(act_tiles * TH * TW, bfloat16(0.0f));

    // Wrap as Tensor
    TensorSpec act_spec(
        Shape({1, 1, 1, buf.K_padded}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));
    buf.act_tensor = Tensor(DeviceStorage(buf.act_buf, {MeshCoordinate(0, 0)}),
                            act_spec, TensorTopology{});

    // Output buffer
    buf.out_buf = MeshBuffer::create(ReplicatedBufferConfig{.size = out_tiles * tile_bytes},
                                      dram_cfg, device);
    buf.out_host_tiled.resize(out_tiles * TH * TW, bfloat16(0.0f));

    TensorSpec out_spec(
        Shape({1, 1, 1, buf.M_padded}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));
    buf.out_tensor = Tensor(DeviceStorage(buf.out_buf, {MeshCoordinate(0, 0)}),
                            out_spec, TensorTopology{});

    auto [ins, _] = g_gemv_bufs.emplace(key, std::move(buf));
    printf("  [gemv_buf] allocated K=%u M=%u on device\n", K, M);
    return ins->second;
}

// Fast bf16↔f32 conversion using raw bit operations (avoids bfloat16 class overhead)
static inline uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return static_cast<uint16_t>(bits >> 16);
}
static inline float bf16_to_f32(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// ============================================================================
// Device-side GEMV: y[M] = W[M,K] @ x[K]  (W on device, x/y on host)
// Pre-allocated buffers eliminate per-call alloc/dealloc overhead.
// ============================================================================
// Pre-allocated bf16 scratch for bulk conversion (max possible K or M)
static std::vector<uint16_t> g_bf16_scratch(MC::n_vocab_padded);

// Per-matmul trace storage: indexed by a sequential ID per unique weight
struct GemvTrace {
    ttnn::MeshTraceId trace_id;
    bool valid = false;
};
static GemvTrace g_gemv_traces[128];  // max 65 unique matmuls + headroom

// Detailed timing counters for device_gemv sub-operations
static double g_t_tilize = 0, g_t_enqueue = 0, g_t_dispatch = 0, g_t_read = 0, g_t_host = 0;
static int g_gemv_count = 0;

static void device_gemv(MeshDevice* device, const Tensor& weight, uint32_t M, uint32_t K,
                        const float* x, float* y, int trace_idx = -1) {
    using Clock = std::chrono::high_resolution_clock;
    auto& gb = get_gemv_buf(device, M, K);
    auto& cq = device->mesh_command_queue();

    // 1. Host-side tilize + PCIe write
    auto t0 = Clock::now();
    uint16_t* scratch = g_bf16_scratch.data();
    const uint32_t* xbits = reinterpret_cast<const uint32_t*>(x);
    for (uint32_t i = 0; i < K; i++)
        scratch[i] = static_cast<uint16_t>(xbits[i] >> 16);

    uint16_t* ht = reinterpret_cast<uint16_t*>(gb.act_host_tiled.data());
    uint32_t num_tile_cols = gb.K_padded / TILE_WIDTH;
    for (uint32_t tc = 0; tc < num_tile_cols; tc++) {
        uint32_t tile_off = tc * TILE_HEIGHT * TILE_WIDTH;
        uint32_t base = tc * TILE_WIDTH;
        memcpy(ht + tile_off, scratch + base, 16 * sizeof(uint16_t));
        memcpy(ht + tile_off + 256, scratch + base + 16, 16 * sizeof(uint16_t));
    }
    auto t0b = Clock::now();
    EnqueueWriteMeshBuffer(cq, gb.act_buf, gb.act_host_tiled, false);
    auto t1 = Clock::now();

    // 2. matmul dispatch (with optional trace)
    auto do_matmul = [&]() {
        ttnn::matmul(gb.act_tensor, weight,
                     false, true, ttnn::DRAM_MEMORY_CONFIG,
                     std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                     std::nullopt, std::nullopt, gb.out_tensor);
    };

    if (trace_idx >= 0 && trace_idx < 128) {
        auto& tr = g_gemv_traces[trace_idx];
        if (!tr.valid) {
            do_matmul();
            EnqueueReadMeshBuffer(cq, gb.out_host_tiled, gb.out_buf, true);
            auto tid = ttnn::operations::trace::begin_trace_capture(device, std::nullopt);
            do_matmul();
            ttnn::operations::trace::end_trace_capture(device, tid, std::nullopt);
            tr.trace_id = tid;
            tr.valid = true;
        } else {
            ttnn::operations::trace::execute_trace(device, tr.trace_id, std::nullopt, false);
        }
    } else {
        do_matmul();
    }
    auto t2 = Clock::now();

    // 3. Blocking read (waits for device compute + PCIe DMA)
    EnqueueReadMeshBuffer(cq, gb.out_host_tiled, gb.out_buf, true);
    auto t3 = Clock::now();

    // 4. Host-side untilize + convert
    const uint16_t* oht = reinterpret_cast<const uint16_t*>(gb.out_host_tiled.data());
    uint32_t out_tile_cols = gb.M_padded / TILE_WIDTH;
    for (uint32_t tc = 0; tc < out_tile_cols; tc++) {
        uint32_t tile_off = tc * TILE_HEIGHT * TILE_WIDTH;
        uint32_t base = tc * TILE_WIDTH;
        memcpy(scratch + base, oht + tile_off, 16 * sizeof(uint16_t));
        memcpy(scratch + base + 16, oht + tile_off + 256, 16 * sizeof(uint16_t));
    }
    uint32_t* ybits = reinterpret_cast<uint32_t*>(y);
    for (uint32_t i = 0; i < M; i++)
        ybits[i] = static_cast<uint32_t>(scratch[i]) << 16;
    auto t4 = Clock::now();

    g_t_tilize += std::chrono::duration<double, std::milli>(t0b - t0).count();
    g_t_enqueue += std::chrono::duration<double, std::milli>(t1 - t0b).count();
    g_t_dispatch += std::chrono::duration<double, std::milli>(t2 - t1).count();
    g_t_read += std::chrono::duration<double, std::milli>(t3 - t2).count();
    g_t_host += std::chrono::duration<double, std::milli>(t4 - t3).count();
    g_gemv_count++;
}

// ============================================================================
// Pre-allocated intermediate buffers for on-device FFN
// ============================================================================
struct FfnBuf {
    // Intermediates for gate/up split + SiLU + multiply
    std::shared_ptr<MeshBuffer> gate_buf;   // [1, n_ff] tiled
    Tensor gate_tensor;
    std::shared_ptr<MeshBuffer> up_buf;     // [1, n_ff] tiled
    Tensor up_tensor;
    std::shared_ptr<MeshBuffer> act_buf;    // [1, n_ff] tiled (silu result / multiply result)
    Tensor act_tensor;
    bool initialized = false;
};
static std::map<MeshDevice*, FfnBuf> g_ffn_bufs;

static FfnBuf& get_ffn_buf(MeshDevice* device) {
    auto it = g_ffn_bufs.find(device);
    if (it != g_ffn_bufs.end()) return it->second;

    FfnBuf buf;
    uint32_t n_ff_padded = ((MC::n_ff + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;
    uint32_t n_ff_tiles = n_ff_padded / TILE_WIDTH;
    uint32_t tile_bytes = TILE_HEIGHT * TILE_WIDTH * sizeof(bfloat16);

    DeviceLocalBufferConfig dram_cfg{.page_size = tile_bytes, .buffer_type = BufferType::DRAM};

    buf.gate_buf = MeshBuffer::create(ReplicatedBufferConfig{.size = n_ff_tiles * tile_bytes},
                                       dram_cfg, device);
    buf.up_buf = MeshBuffer::create(ReplicatedBufferConfig{.size = n_ff_tiles * tile_bytes},
                                     dram_cfg, device);
    buf.act_buf = MeshBuffer::create(ReplicatedBufferConfig{.size = n_ff_tiles * tile_bytes},
                                      dram_cfg, device);

    TensorSpec half_spec(
        Shape({1, 1, 1, n_ff_padded}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));

    buf.gate_tensor = Tensor(DeviceStorage(buf.gate_buf, {MeshCoordinate(0, 0)}),
                              half_spec, TensorTopology{});
    buf.up_tensor = Tensor(DeviceStorage(buf.up_buf, {MeshCoordinate(0, 0)}),
                            half_spec, TensorTopology{});
    buf.act_tensor = Tensor(DeviceStorage(buf.act_buf, {MeshCoordinate(0, 0)}),
                             half_spec, TensorTopology{});
    buf.initialized = true;

    auto [ins, _] = g_ffn_bufs.emplace(device, std::move(buf));
    printf("  [ffn_buf] allocated n_ff=%u intermediates on device\n", MC::n_ff);
    return ins->second;
}

// ============================================================================
// Per-layer FFN trace captures (avoids dispatch overhead on replay)
// ============================================================================
static ttnn::MeshTraceId g_ffn_traces[32];
static bool g_ffn_traces_valid[32] = {};

// Run FFN device operations (for capture or direct execution)
static void ffn_device_ops(MeshDevice* device, const Tensor& gate_up_weight,
                           const Tensor& down_weight) {
    auto& gb_gu = get_gemv_buf(device, 2 * MC::n_ff, MC::n_embd);
    auto& gb_dn = get_gemv_buf(device, MC::n_embd, MC::n_ff);
    auto& fb = get_ffn_buf(device);

    ttnn::matmul(gb_gu.act_tensor, gate_up_weight,
                 false, true, ttnn::DRAM_MEMORY_CONFIG,
                 std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                 std::nullopt, std::nullopt, gb_gu.out_tensor);

    std::array<uint32_t, 4> gate_begin = {0, 0, 0, 0};
    std::array<uint32_t, 4> gate_end = {1, 1, 1, (uint32_t)MC::n_ff};
    std::array<uint32_t, 4> step = {1, 1, 1, 1};
    std::array<uint32_t, 4> up_begin = {0, 0, 0, (uint32_t)MC::n_ff};
    std::array<uint32_t, 4> up_end = {1, 1, 1, (uint32_t)(2 * MC::n_ff)};

    ttnn::slice(gb_gu.out_tensor, gate_begin, gate_end, step, ttnn::DRAM_MEMORY_CONFIG,
                fb.gate_tensor);
    ttnn::slice(gb_gu.out_tensor, up_begin, up_end, step, ttnn::DRAM_MEMORY_CONFIG,
                fb.up_tensor);

    ttnn::silu(fb.gate_tensor, ttnn::DRAM_MEMORY_CONFIG, fb.act_tensor);
    ttnn::multiply(fb.act_tensor, fb.up_tensor, std::nullopt, ttnn::DRAM_MEMORY_CONFIG,
                   fb.act_tensor);

    ttnn::matmul(fb.act_tensor, down_weight,
                 false, true, ttnn::DRAM_MEMORY_CONFIG,
                 std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                 std::nullopt, std::nullopt, gb_dn.out_tensor);
}

// Device-side FFN with trace capture/replay
static void device_ffn(MeshDevice* device, int layer_idx,
                       const Tensor& gate_up_weight,
                       const Tensor& down_weight,
                       const float* x_in, float* y_out) {
    auto& gb_gu = get_gemv_buf(device, 2 * MC::n_ff, MC::n_embd);
    auto& gb_dn = get_gemv_buf(device, MC::n_embd, MC::n_ff);
    auto& cq = device->mesh_command_queue();

    // 1. Write x_in to device
    uint16_t* scratch = g_bf16_scratch.data();
    const uint32_t* xbits = reinterpret_cast<const uint32_t*>(x_in);
    for (uint32_t i = 0; i < (uint32_t)MC::n_embd; i++)
        scratch[i] = static_cast<uint16_t>(xbits[i] >> 16);

    uint16_t* ht = reinterpret_cast<uint16_t*>(gb_gu.act_host_tiled.data());
    uint32_t num_tile_cols = gb_gu.K_padded / TILE_WIDTH;
    for (uint32_t tc = 0; tc < num_tile_cols; tc++) {
        uint32_t tile_off = tc * TILE_HEIGHT * TILE_WIDTH;
        uint32_t base = tc * TILE_WIDTH;
        memcpy(ht + tile_off, scratch + base, 16 * sizeof(uint16_t));
        memcpy(ht + tile_off + 256, scratch + base + 16, 16 * sizeof(uint16_t));
    }
    EnqueueWriteMeshBuffer(cq, gb_gu.act_buf, gb_gu.act_host_tiled, false);

    // 2. Execute FFN ops (capture trace on first call, replay on subsequent)
    if (!g_ffn_traces_valid[layer_idx]) {
        // First call: warm up program cache
        ffn_device_ops(device, gate_up_weight, down_weight);
        EnqueueReadMeshBuffer(cq, gb_dn.out_host_tiled, gb_dn.out_buf, true);

        // Capture trace for subsequent calls
        auto tid = ttnn::operations::trace::begin_trace_capture(device, std::nullopt);
        ffn_device_ops(device, gate_up_weight, down_weight);
        ttnn::operations::trace::end_trace_capture(device, tid, std::nullopt);
        g_ffn_traces[layer_idx] = tid;
        g_ffn_traces_valid[layer_idx] = true;
    } else {
        // Replay captured trace (single dispatch for all FFN ops)
        ttnn::operations::trace::execute_trace(device, g_ffn_traces[layer_idx],
                                                std::nullopt, false);
        EnqueueReadMeshBuffer(cq, gb_dn.out_host_tiled, gb_dn.out_buf, true);
    }

    // 3. Untilize + convert result
    const uint16_t* oht = reinterpret_cast<const uint16_t*>(gb_dn.out_host_tiled.data());
    uint32_t out_tile_cols = gb_dn.M_padded / TILE_WIDTH;
    for (uint32_t tc = 0; tc < out_tile_cols; tc++) {
        uint32_t tile_off = tc * TILE_HEIGHT * TILE_WIDTH;
        uint32_t base = tc * TILE_WIDTH;
        memcpy(scratch + base, oht + tile_off, 16 * sizeof(uint16_t));
        memcpy(scratch + base + 16, oht + tile_off + 256, 16 * sizeof(uint16_t));
    }
    uint32_t* ybits = reinterpret_cast<uint32_t*>(y_out);
    for (uint32_t i = 0; i < (uint32_t)MC::n_embd; i++)
        ybits[i] = static_cast<uint32_t>(scratch[i]) << 16;
}

// ============================================================================
// RMSNorm: out[i] = x[i] / sqrt(mean(x^2) + eps) * w[i]
// ============================================================================
static void rmsnorm(const float* x, const float* w, float* out, int dim) {
    float sum_sq = 0;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float rms = 1.0f / sqrtf(sum_sq / dim + MC::rms_norm_eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * rms * w[i];
}


// ============================================================================
// Cached small weights (norms, SSM params — tiny, keep permanently on host)
// ============================================================================
struct SmallAttnWeights {
    std::vector<float> q_norm;
    std::vector<float> k_norm;
};
static SmallAttnWeights g_attn_small[8];

struct LayerNorms {
    std::vector<float> attn_norm;
    std::vector<float> post_norm;
};
static LayerNorms g_layer_norms[32];
static std::vector<float> g_output_norm;

// Helper: read bf16 weights from device MeshBuffer → host bf16 vector (flat)
static std::vector<bfloat16> read_tiled_bf16(MeshCommandQueue& cq,
    std::shared_ptr<MeshBuffer> buf) {
    std::vector<bfloat16> tiled;
    EnqueueReadMeshBuffer(cq, tiled, buf, true);
    return tiled;
}

// Untilize 1D weight stored as [1, len] tiled → f32 vector
static std::vector<float> read_1d_f32(MeshCommandQueue& cq,
    std::shared_ptr<MeshBuffer> buf, uint32_t len) {
    auto tiled = read_tiled_bf16(cq, buf);
    uint32_t cp = ((len + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;
    auto flat = untilize_nfaces(tiled, TILE_HEIGHT, cp);
    std::vector<float> result(len);
    for (uint32_t i = 0; i < len; i++)
        result[i] = static_cast<float>(flat[i]);
    return result;
}

static void cache_small_weights(MeshCommandQueue& cq) {
    g_output_norm = read_1d_f32(cq, g_model.output_norm, MC::n_embd);

    int attn_idx = 0, ssm_idx = 0;
    for (int layer = 0; layer < MC::n_layers; layer++) {
        auto& ln = g_layer_norms[layer];

        if (MC::is_recurrent(layer)) {
            auto& lw = g_model.ssm_layers[ssm_idx];
            ln.attn_norm = read_1d_f32(cq, lw.attn_norm, MC::n_embd);
            ln.post_norm = read_1d_f32(cq, lw.post_attn_norm, MC::n_embd);
            ssm_idx++;
        } else {
            auto& lw = g_model.attn_layers[attn_idx];
            ln.attn_norm = read_1d_f32(cq, lw.attn_norm, MC::n_embd);
            ln.post_norm = read_1d_f32(cq, lw.post_attn_norm, MC::n_embd);

            auto& aw = g_attn_small[attn_idx];
            aw.q_norm = read_1d_f32(cq, lw.attn_q_norm, MC::head_dim);
            aw.k_norm = read_1d_f32(cq, lw.attn_k_norm, MC::head_dim);
            attn_idx++;
        }
    }
}

// Upload host bf16 data to a device as a weight Tensor [M, K] in tiled BF16 layout
static Tensor upload_bf16_weight(MeshDevice* device, const uint16_t* bf16_data,
                                  uint32_t M, uint32_t K) {
    // Convert bf16 (uint16) to float for from_span
    size_t n = (size_t)M * K;
    std::vector<float> f32(n);
    const bfloat16* src = reinterpret_cast<const bfloat16*>(bf16_data);
    for (size_t i = 0; i < n; i++) f32[i] = static_cast<float>(src[i]);

    TensorSpec spec(
        Shape({1, 1, M, K}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));
    return Tensor::from_span<float>(std::span<const float>(f32.data(), n), spec, device);
}

// Create Tensor wrappers for all on-device weight MeshBuffers
static void create_weight_tensors() {
    // Chip 0: layer weights (already on device via GGUF loader)
    int attn_idx = 0, ssm_idx = 0;
    for (int layer = 0; layer < MC::n_layers; layer++) {
        if (MC::is_recurrent(layer)) {
            auto& lw = g_model.ssm_layers[ssm_idx];
            uint32_t combined_rows = MC::ssm_conv_channels + MC::ssm_d_inner
                                   + MC::ssm_dt_rank + MC::ssm_dt_rank;
            g_wt.ssm_w_combined[ssm_idx] = wrap_weight(lw.w_combined, combined_rows, MC::n_embd);
            g_wt.ssm_ffn_gate_up[ssm_idx] = wrap_weight(lw.ffn_gate_up, 2 * MC::n_ff, MC::n_embd);
            g_wt.ssm_ffn_down[ssm_idx] = wrap_weight(lw.ffn_down, MC::n_embd, MC::n_ff);
            ssm_idx++;
        } else {
            auto& lw = g_model.attn_layers[attn_idx];
            int q_dim = MC::n_head * MC::head_dim * 2;
            int kv_dim_one = MC::n_head_kv * MC::head_dim;
            int qkv_rows = q_dim + 2 * kv_dim_one;
            g_wt.attn_wqkv[attn_idx] = wrap_weight(lw.wqkv, qkv_rows, MC::n_embd);
            g_wt.attn_ffn_gate_up[attn_idx] = wrap_weight(lw.ffn_gate_up, 2 * MC::n_ff, MC::n_embd);
            g_wt.attn_ffn_down[attn_idx] = wrap_weight(lw.ffn_down, MC::n_embd, MC::n_ff);
            attn_idx++;
        }
    }
    printf("Created %d attention + %d SSM weight tensor wrappers on chip 0.\n",
           attn_idx, ssm_idx);

    // Chip 1: wo, ssm_out, LM head (uploaded from host bf16 vectors)
    printf("Uploading output projections to chip 1...\n");
    for (int i = 0; i < 8; i++) {
        auto& lw = g_model.attn_layers[i];
        g_wt.attn_wo[i] = upload_bf16_weight(g_mesh2.get(), lw.wo_host.data(),
                                              MC::n_embd, MC::n_head * MC::head_dim);
        printf("  wo[%d] uploaded\n", i);
    }
    for (int i = 0; i < 24; i++) {
        auto& lw = g_model.ssm_layers[i];
        g_wt.ssm_out[i] = upload_bf16_weight(g_mesh2.get(), lw.ssm_out_host.data(),
                                              MC::n_embd, MC::ssm_d_inner);
        printf("  ssm_out[%d] uploaded\n", i);
    }
    g_wt.lm_head = upload_bf16_weight(g_mesh2.get(), g_model.output_host.data(),
                                       MC::n_vocab, MC::n_embd);
    printf("  lm_head uploaded\n");

    // Free host bf16 vectors now that weights are on chip 1
    for (int i = 0; i < 8; i++)
        g_model.attn_layers[i].wo_host.clear();
    for (int i = 0; i < 24; i++)
        g_model.ssm_layers[i].ssm_out_host.clear();
    g_model.output_host.clear();
    printf("Freed host bf16 vectors (wo, ssm_out, output) — saved ~2.9 GB host RAM\n");
}

// ============================================================================
// Pre-allocated scratch buffers for forward_decode (avoid heap allocs per token)
// ============================================================================
static std::vector<float> g_norm_out(MC::n_embd);
static std::vector<float> g_residual(MC::n_embd);
static constexpr int g_combined_rows = MC::ssm_conv_channels + MC::ssm_d_inner
                                     + MC::ssm_dt_rank + MC::ssm_dt_rank;
static std::vector<float> g_proj(g_combined_rows);
static std::vector<float> g_conv_out(MC::ssm_conv_channels);
static std::vector<float> g_delta_out(MC::ssm_d_inner);
static std::vector<float> g_ssm_proj_in(MC::ssm_d_inner);
static std::vector<float> g_layer_out(MC::n_embd);
static std::vector<float> g_ffn_buf(2 * MC::n_ff);
static std::vector<float> g_ffn_act(MC::n_ff);
static std::vector<float> g_ffn_out(MC::n_embd);
static constexpr int g_qkv_rows = MC::n_head * MC::head_dim * 2 + 2 * MC::n_head_kv * MC::head_dim;
static std::vector<float> g_qkv(g_qkv_rows);
static std::vector<float> g_q_heads(MC::n_head * MC::head_dim);
static std::vector<float> g_gate_heads(MC::n_head * MC::head_dim);
static std::vector<float> g_attn_out(MC::n_head * MC::head_dim);
static std::vector<float> g_acc(MC::head_dim);
static std::vector<float> g_logits(MC::n_vocab);


// ============================================================================
// Pre-computed RoPE cos/sin tables (avoid trig calls per token)
// ============================================================================
static std::vector<float> g_rope_cos;  // [max_ctx * rope_dim/2]
static std::vector<float> g_rope_sin;  // [max_ctx * rope_dim/2]

static void precompute_rope(int max_ctx) {
    constexpr int half = MC::rope_dim / 2;
    g_rope_cos.resize((size_t)max_ctx * half);
    g_rope_sin.resize((size_t)max_ctx * half);
    for (int pos = 0; pos < max_ctx; pos++) {
        for (int p = 0; p < half; p++) {
            float freq = 1.0f / powf(MC::rope_freq_base, 2.0f * p / MC::rope_dim);
            float theta = pos * freq;
            g_rope_cos[pos * half + p] = cosf(theta);
            g_rope_sin[pos * half + p] = sinf(theta);
        }
    }
}

static void apply_rope_cached(float* head, int pos) {
    constexpr int half = MC::rope_dim / 2;
    const float* cos_t = g_rope_cos.data() + pos * half;
    const float* sin_t = g_rope_sin.data() + pos * half;
    for (int p = 0; p < half; p++) {
        int i0 = p, i1 = p + half;
        float x0 = head[i0], x1 = head[i1];
        head[i0] = x0 * cos_t[p] - x1 * sin_t[p];
        head[i1] = x1 * cos_t[p] + x0 * sin_t[p];
    }
}

// ============================================================================
// Forward pass: single decode token
// Large matmuls on device via ttnn::matmul, small ops on host CPU.
// Returns pointer to static g_logits buffer (valid until next call).
// ============================================================================
static int g_decode_count = 0;
static double g_time_gemv = 0, g_time_ffn = 0, g_time_ssm = 0, g_time_attn = 0, g_time_lmhead = 0;

static float* forward_decode() {
    using Clock = std::chrono::high_resolution_clock;
    int pos = g_pos;
    float* norm_out = g_norm_out.data();
    float* residual = g_residual.data();

    int attn_idx = 0, ssm_idx = 0;
    for (int layer = 0; layer < MC::n_layers; layer++) {
        auto& ln = g_layer_norms[layer];

        // Save residual
        memcpy(residual, g_hidden_f32.data(), MC::n_embd * sizeof(float));

        // Pre-norm
        rmsnorm(g_hidden_f32.data(), ln.attn_norm.data(), norm_out, MC::n_embd);

        if (MC::is_recurrent(layer)) {
            // ======== SSM (Delta-Net) Layer ========
            auto& lw = g_model.ssm_layers[ssm_idx];

            // 1. Combined projection via DEVICE matmul (trace idx 0-23)
            auto t_gemv0 = Clock::now();
            device_gemv(g_mesh.get(), g_wt.ssm_w_combined[ssm_idx], g_combined_rows, MC::n_embd,
                       norm_out, g_proj.data(), ssm_idx);
            g_time_gemv += std::chrono::duration<double, std::milli>(Clock::now() - t_gemv0).count();

            float* qkv_raw = g_proj.data();
            float* z_raw = g_proj.data() + MC::ssm_conv_channels;
            float* alpha_raw = z_raw + MC::ssm_d_inner;
            float* beta_raw = alpha_raw + MC::ssm_dt_rank;

            // 2. Conv1d + SiLU
            auto& cs = g_conv_state[ssm_idx];
            float* conv_out = g_conv_out.data();
            for (int ch = 0; ch < MC::ssm_conv_channels; ch++) {
                float sum = 0;
                for (int k = 0; k < MC::ssm_conv_kernel; k++) {
                    float val;
                    if (k < conv_state_len)
                        val = cs[k * MC::ssm_conv_channels + ch];
                    else
                        val = qkv_raw[ch];
                    sum += val * lw.ssm_conv1d_host[ch * MC::ssm_conv_kernel + k];
                }
                conv_out[ch] = sum / (1.0f + expf(-sum));  // SiLU

                // Shift state
                for (int i = 0; i < conv_state_len - 1; i++)
                    cs[i * MC::ssm_conv_channels + ch] = cs[(i + 1) * MC::ssm_conv_channels + ch];
                cs[(conv_state_len - 1) * MC::ssm_conv_channels + ch] = qkv_raw[ch];
            }

            // 3. Split conv output: Q[2048] | K[2048] | V[4096]
            constexpr int num_k_heads = MC::ssm_n_group;
            constexpr int head_k = MC::ssm_d_state;
            constexpr int num_v = ssm_n_v_heads;
            constexpr int head_v = ssm_head_v_dim_c;

            float* conv_q = conv_out;
            float* conv_k = conv_out + num_k_heads * head_k;
            float* conv_v = conv_out + 2 * num_k_heads * head_k;

            // 4. Delta-net recurrence
            auto& state = g_ssm_state[ssm_idx];
            float* delta_out = g_delta_out.data();
            constexpr float ssm_scale = 1.0f / 11.3137f;  // 1/sqrt(128)

            for (int vh = 0; vh < num_v; vh++) {
                int kh = vh % num_k_heads;
                float q[head_k], k[head_k], v[head_v];
                memcpy(q, conv_q + kh * head_k, head_k * sizeof(float));
                memcpy(k, conv_k + kh * head_k, head_k * sizeof(float));
                memcpy(v, conv_v + vh * head_v, head_v * sizeof(float));

                // L2 normalize
                float qn = 0, kn = 0;
                for (int d = 0; d < head_k; d++) { qn += q[d]*q[d]; kn += k[d]*k[d]; }
                qn = 1.0f / sqrtf(qn + MC::rms_norm_eps);
                kn = 1.0f / sqrtf(kn + MC::rms_norm_eps);
                for (int d = 0; d < head_k; d++) { q[d] *= qn; k[d] *= kn; }

                // Gate
                float biased = alpha_raw[vh] + lw.ssm_dt_bias_host[vh];
                float sp = (biased > 20.0f) ? biased : logf(1.0f + expf(biased));
                float gate = sp * lw.ssm_a_host[vh];
                float decay = expf(gate);

                // Beta
                float beta = 1.0f / (1.0f + expf(-beta_raw[vh]));

                // State layout: [head_v, head_k] for contiguous inner-loop access
                float* sh = state.data() + vh * head_v * head_k;

                // Decay
                for (int j = 0; j < head_v * head_k; j++) sh[j] *= decay;

                // Delta update + output (inner loops now access contiguous memory)
                for (int i = 0; i < head_v; i++) {
                    float* row = sh + i * head_k;
                    float sk = 0;
                    for (int j = 0; j < head_k; j++) sk += row[j] * k[j];
                    float d = beta * (v[i] - sk);
                    for (int j = 0; j < head_k; j++) row[j] += k[j] * d;
                    float out = 0;
                    for (int j = 0; j < head_k; j++) out += row[j] * q[j];
                    delta_out[vh * head_v + i] = out * ssm_scale;
                }
            }

            // 5. Gated RMSNorm (ssm_norm is [128], broadcast across all v_heads)
            float* ssm_proj_in = g_ssm_proj_in.data();
            for (int vh = 0; vh < num_v; vh++) {
                float sum_sq = 0;
                for (int d = 0; d < head_v; d++) {
                    float val = delta_out[vh * head_v + d];
                    sum_sq += val * val;
                }
                float rms = 1.0f / sqrtf(sum_sq / head_v + MC::rms_norm_eps);
                for (int d = 0; d < head_v; d++) {
                    int idx = vh * head_v + d;
                    float normalized = delta_out[idx] * rms * lw.ssm_norm_host[d];
                    float z = z_raw[idx];
                    float silu_z = z / (1.0f + expf(-z));
                    ssm_proj_in[idx] = normalized * silu_z;
                }
            }

            // 6. Output projection (DEVICE matmul on chip 1, trace idx 24-47)
            auto t_gemv1 = Clock::now();
            float* layer_out = g_layer_out.data();
            device_gemv(g_mesh2.get(), g_wt.ssm_out[ssm_idx], MC::n_embd, MC::ssm_d_inner,
                       ssm_proj_in, layer_out, 24 + ssm_idx);
            g_time_gemv += std::chrono::duration<double, std::milli>(Clock::now() - t_gemv1).count();

            // 7. Residual
            for (int i = 0; i < MC::n_embd; i++)
                g_hidden_f32[i] = residual[i] + layer_out[i];

            memcpy(residual, g_hidden_f32.data(), MC::n_embd * sizeof(float));

            // 8. Post-norm + FFN (chained on device)
            rmsnorm(g_hidden_f32.data(), ln.post_norm.data(), norm_out, MC::n_embd);

            auto t_ffn0 = Clock::now();
            float* ffn_out = g_ffn_out.data();
            device_ffn(g_mesh.get(), layer, g_wt.ssm_ffn_gate_up[ssm_idx], g_wt.ssm_ffn_down[ssm_idx],
                      norm_out, ffn_out);
            g_time_ffn += std::chrono::duration<double, std::milli>(Clock::now() - t_ffn0).count();

            for (int i = 0; i < MC::n_embd; i++)
                g_hidden_f32[i] = residual[i] + ffn_out[i];

            ssm_idx++;
        } else {
            // ======== Full Attention Layer ========
            auto& lw = g_model.attn_layers[attn_idx];
            auto& aw = g_attn_small[attn_idx];

            // 1. QKV projection (DEVICE matmul, trace idx 48-55)
            auto t_qkv = Clock::now();
            constexpr int q_dim = MC::n_head * MC::head_dim * 2;
            constexpr int kv_dim_one = MC::n_head_kv * MC::head_dim;
            float* qkv = g_qkv.data();
            device_gemv(g_mesh.get(), g_wt.attn_wqkv[attn_idx], g_qkv_rows, MC::n_embd,
                       norm_out, qkv, 48 + attn_idx);
            g_time_gemv += std::chrono::duration<double, std::milli>(Clock::now() - t_qkv).count();

            // 2. Deinterleave Q and gate
            float* q_heads = g_q_heads.data();
            float* gate_heads = g_gate_heads.data();
            for (int h = 0; h < MC::n_head; h++) {
                for (int d = 0; d < MC::head_dim; d++) {
                    q_heads[h * MC::head_dim + d] = qkv[h * MC::head_dim * 2 + d];
                    gate_heads[h * MC::head_dim + d] = qkv[h * MC::head_dim * 2 + MC::head_dim + d];
                }
            }
            float* k_proj = qkv + q_dim;
            float* v_proj = k_proj + kv_dim_one;

            // 3. Per-head Q/K RMSNorm
            for (int h = 0; h < MC::n_head; h++) {
                float* qh = q_heads + h * MC::head_dim;
                float ss = 0;
                for (int d = 0; d < MC::head_dim; d++) ss += qh[d] * qh[d];
                float rms = 1.0f / sqrtf(ss / MC::head_dim + MC::rms_norm_eps);
                for (int d = 0; d < MC::head_dim; d++)
                    qh[d] = qh[d] * rms * aw.q_norm[d];
            }
            for (int h = 0; h < MC::n_head_kv; h++) {
                float* kh = k_proj + h * MC::head_dim;
                float ss = 0;
                for (int d = 0; d < MC::head_dim; d++) ss += kh[d] * kh[d];
                float rms = 1.0f / sqrtf(ss / MC::head_dim + MC::rms_norm_eps);
                for (int d = 0; d < MC::head_dim; d++)
                    kh[d] = kh[d] * rms * aw.k_norm[d];
            }

            // 4. RoPE (using pre-computed tables)
            for (int h = 0; h < MC::n_head; h++)
                apply_rope_cached(q_heads + h * MC::head_dim, pos);
            for (int h = 0; h < MC::n_head_kv; h++)
                apply_rope_cached(k_proj + h * MC::head_dim, pos);

            // 5. KV cache
            memcpy(g_k_cache[attn_idx].data() + (size_t)pos * kv_dim,
                   k_proj, kv_dim * sizeof(float));
            memcpy(g_v_cache[attn_idx].data() + (size_t)pos * kv_dim,
                   v_proj, kv_dim * sizeof(float));
            int kv_len = pos + 1;

            // 6. Attention (online softmax)
            float* attn_out = g_attn_out.data();
            memset(attn_out, 0, MC::n_head * MC::head_dim * sizeof(float));
            for (int h = 0; h < MC::n_head; h++) {
                int kv_h = h / (MC::n_head / MC::n_head_kv);
                float* qh = q_heads + h * MC::head_dim;
                float* out = attn_out + h * MC::head_dim;
                float max_score = -FLT_MAX, sum_exp = 0;
                float* acc = g_acc.data();
                memset(acc, 0, MC::head_dim * sizeof(float));

                for (int kp = 0; kp < kv_len; kp++) {
                    float* kh = g_k_cache[attn_idx].data() + (size_t)kp * kv_dim + kv_h * MC::head_dim;
                    float dot = 0;
                    for (int d = 0; d < MC::head_dim; d++) dot += qh[d] * kh[d];
                    float score = dot * MC::attn_scale;

                    float new_max = std::max(max_score, score);
                    float exp_s = expf(score - new_max);
                    float corr = expf(max_score - new_max);
                    sum_exp = sum_exp * corr + exp_s;

                    float* vh = g_v_cache[attn_idx].data() + (size_t)kp * kv_dim + kv_h * MC::head_dim;
                    for (int d = 0; d < MC::head_dim; d++)
                        acc[d] = acc[d] * corr + exp_s * vh[d];
                    max_score = new_max;
                }
                for (int d = 0; d < MC::head_dim; d++) out[d] = acc[d] / sum_exp;
            }

            // 7. Sigmoid gating
            for (int i = 0; i < MC::n_head * MC::head_dim; i++)
                attn_out[i] *= 1.0f / (1.0f + expf(-gate_heads[i]));

            // 8. Output projection (DEVICE matmul on chip 1, trace idx 56-63)
            auto t_wo = Clock::now();
            float* layer_out = g_layer_out.data();
            device_gemv(g_mesh2.get(), g_wt.attn_wo[attn_idx], MC::n_embd, MC::n_head * MC::head_dim,
                       attn_out, layer_out, 56 + attn_idx);
            g_time_gemv += std::chrono::duration<double, std::milli>(Clock::now() - t_wo).count();

            // 9. Residual
            for (int i = 0; i < MC::n_embd; i++)
                g_hidden_f32[i] = residual[i] + layer_out[i];

            memcpy(residual, g_hidden_f32.data(), MC::n_embd * sizeof(float));

            // 10. Post-norm + FFN (chained on device)
            rmsnorm(g_hidden_f32.data(), ln.post_norm.data(), norm_out, MC::n_embd);

            auto t_ffn1 = Clock::now();
            float* ffn_out = g_ffn_out.data();
            device_ffn(g_mesh.get(), layer, g_wt.attn_ffn_gate_up[attn_idx], g_wt.attn_ffn_down[attn_idx],
                      norm_out, ffn_out);
            g_time_ffn += std::chrono::duration<double, std::milli>(Clock::now() - t_ffn1).count();

            for (int i = 0; i < MC::n_embd; i++)
                g_hidden_f32[i] = residual[i] + ffn_out[i];

            attn_idx++;
        }
    }

    // Final norm
    rmsnorm(g_hidden_f32.data(), g_output_norm.data(), norm_out, MC::n_embd);

    // LM head (DEVICE matmul on chip 1, trace idx 64)
    auto t_lm = Clock::now();
    float* logits = g_logits.data();
    device_gemv(g_mesh2.get(), g_wt.lm_head, MC::n_vocab, MC::n_embd,
               norm_out, logits, 64);
    g_time_lmhead += std::chrono::duration<double, std::milli>(Clock::now() - t_lm).count();

    g_decode_count++;
    if (g_decode_count % 10 == 0) {
        int calls = g_gemv_count > 0 ? g_gemv_count : 1;
        printf("  [profile @%d] gemv=%.0f ffn=%.0f lmhead=%.0f ms/tok\n",
               g_decode_count,
               g_time_gemv / g_decode_count,
               g_time_ffn / g_decode_count,
               g_time_lmhead / g_decode_count);
        printf("    gemv: tilize=%.2f enq=%.2f disp=%.2f read=%.2f host=%.2f ms/call (%d calls)\n",
               g_t_tilize / calls, g_t_enqueue / calls, g_t_dispatch / calls, g_t_read / calls, g_t_host / calls, calls);
    }

    return logits;
}

// ============================================================================
// Public API
// ============================================================================

bool load_model_and_tokenizer(const char* model_path, int max_ctx) {
    printf("Loading model from %s (max_ctx=%d)...\n", model_path, max_ctx);

    // Open both N300 chips: chip 0 for layer weights, chip 1 for output projections
    auto meshes = MeshDevice::create_unit_meshes({0, 1});
    g_mesh = meshes[0];
    g_mesh2 = meshes[1];

    auto grid = g_mesh->compute_with_storage_grid_size();
    printf("Chip 0 opened: compute grid %zux%zu (%zu cores)\n",
           grid.x, grid.y, grid.x * grid.y);
    auto grid2 = g_mesh2->compute_with_storage_grid_size();
    printf("Chip 1 opened: compute grid %zux%zu (%zu cores)\n",
           grid2.x, grid2.y, grid2.x * grid2.y);

    // Enable program cache for faster repeated matmul dispatch
    g_mesh->enable_program_cache();
    g_mesh2->enable_program_cache();

    MeshCommandQueue& cq = g_mesh->mesh_command_queue();
    g_max_ctx = max_ctx;

    if (!g_tokenizer.load(model_path)) {
        fprintf(stderr, "Failed to load tokenizer\n");
        return false;
    }

    if (!load_gguf_weights(model_path, g_model, g_mesh.get(), cq)) {
        fprintf(stderr, "Failed to load weights\n");
        return false;
    }
    Finish(cq);

    // Allocate KV caches
    for (int i = 0; i < 8; i++) {
        g_k_cache[i].resize((size_t)max_ctx * kv_dim, 0.0f);
        g_v_cache[i].resize((size_t)max_ctx * kv_dim, 0.0f);
    }
    for (int i = 0; i < 24; i++) {
        g_ssm_state[i].resize(ssm_n_v_heads * ssm_head_k_dim * ssm_head_v_dim_c, 0.0f);
        g_conv_state[i].resize(conv_state_len * MC::ssm_conv_channels, 0.0f);
    }

    // Cache small weights (norms, SSM params) — tiny, < 50 MB total
    printf("Caching small weights...\n");
    cache_small_weights(cq);

    // Pre-compute RoPE cos/sin tables
    printf("Pre-computing RoPE tables for %d positions...\n", max_ctx);
    precompute_rope(max_ctx);

    // Create Tensor wrappers for on-device weight matrices
    printf("Creating weight tensor wrappers...\n");
    create_weight_tensors();

    printf("Ready.\n");

    g_loaded = true;
    g_pos = 0;
    return true;
}

int generate(const std::vector<int>& prompt_tokens, int max_tokens,
             float temperature, TokenCallback cb, StopReason* stop_reason) {
    if (!g_loaded) {
        fprintf(stderr, "Model not loaded\n");
        return 0;
    }

    int total_generated = 0;
    int next_token = -1;

    auto t_start = std::chrono::high_resolution_clock::now();

    // Process all prompt tokens
    for (int i = 0; i < (int)prompt_tokens.size(); i++) {
        int token = prompt_tokens[i];
        printf("  [prefill token %d/%d: %d]\n", i + 1, (int)prompt_tokens.size(), token);

        // Bulk bf16→f32 embedding lookup using raw bit ops
        const uint16_t* emb = g_model.tok_embd_host.data() + (size_t)token * MC::n_embd;
        uint32_t* hbits = reinterpret_cast<uint32_t*>(g_hidden_f32.data());
        for (int j = 0; j < MC::n_embd; j++)
            hbits[j] = static_cast<uint32_t>(emb[j]) << 16;

        const float* logits = forward_decode();
        g_pos++;

        // After last prompt token, sample first output
        if (i == (int)prompt_tokens.size() - 1) {
            float max_l = -FLT_MAX;
            for (int v = 0; v < MC::n_vocab; v++) {
                if (logits[v] > max_l) { max_l = logits[v]; next_token = v; }
            }
        }
    }

    auto t_prefill = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill - t_start).count();
    printf("  [prefill: %.1f ms for %zu tokens (%.1f ms/tok)]\n",
           prefill_ms, prompt_tokens.size(), prefill_ms / prompt_tokens.size());

    // Generate tokens
    while (total_generated < max_tokens) {
        total_generated++;
        if (cb) {
            std::string text = g_tokenizer.decode(next_token);
            if (!cb(next_token, text)) {
                if (stop_reason) *stop_reason = STOP_CALLBACK;
                return total_generated;
            }
        }
        if (next_token == g_tokenizer.eos_token_id()) {
            if (stop_reason) *stop_reason = STOP_EOS;
            return total_generated;
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        // Forward pass with generated token
        const bfloat16* emb = reinterpret_cast<const bfloat16*>(
            g_model.tok_embd_host.data() + (size_t)next_token * MC::n_embd);
        for (int j = 0; j < MC::n_embd; j++)
            g_hidden_f32[j] = static_cast<float>(emb[j]);

        const float* logits = forward_decode();
        g_pos++;

        auto t1 = std::chrono::high_resolution_clock::now();
        double tok_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("  [decode: %.0f ms]\n", tok_ms);

        float max_l = -FLT_MAX;
        int best = 0;
        for (int v = 0; v < MC::n_vocab; v++) {
            if (logits[v] > max_l) { max_l = logits[v]; best = v; }
        }
        next_token = best;
    }

    if (stop_reason) *stop_reason = STOP_LENGTH;
    return total_generated;
}

void reset_state() {
    g_pos = 0;
    std::fill(g_hidden_f32.begin(), g_hidden_f32.end(), 0.0f);
    for (int i = 0; i < 8; i++) {
        std::fill(g_k_cache[i].begin(), g_k_cache[i].end(), 0.0f);
        std::fill(g_v_cache[i].begin(), g_v_cache[i].end(), 0.0f);
    }
    for (int i = 0; i < 24; i++) {
        std::fill(g_ssm_state[i].begin(), g_ssm_state[i].end(), 0.0f);
        std::fill(g_conv_state[i].begin(), g_conv_state[i].end(), 0.0f);
    }
}

const Tokenizer& get_tokenizer() {
    return g_tokenizer;
}

void shutdown() {
    if (!g_loaded) return;
    g_loaded = false;

    // Release FFN traces
    for (int i = 0; i < 32; i++) {
        if (g_ffn_traces_valid[i]) {
            ttnn::operations::trace::release_trace(g_mesh.get(), g_ffn_traces[i]);
            g_ffn_traces_valid[i] = false;
        }
    }
    // Release gemv traces (chip 0: idx 0-23,48-55; chip 1: idx 24-47,56-64)
    for (int i = 0; i < 128; i++) {
        if (g_gemv_traces[i].valid) {
            MeshDevice* dev = (i >= 24 && i <= 47) || (i >= 56) ? g_mesh2.get() : g_mesh.get();
            ttnn::operations::trace::release_trace(dev, g_gemv_traces[i].trace_id);
            g_gemv_traces[i].valid = false;
        }
    }

    // Clear pre-allocated GEMV and FFN buffers
    g_ffn_bufs.clear();
    g_gemv_bufs.clear();

    // Clear weight tensor wrappers — chip 0
    for (auto& t : g_wt.attn_wqkv) t = Tensor();
    for (auto& t : g_wt.attn_ffn_gate_up) t = Tensor();
    for (auto& t : g_wt.attn_ffn_down) t = Tensor();
    for (auto& t : g_wt.ssm_w_combined) t = Tensor();
    for (auto& t : g_wt.ssm_ffn_gate_up) t = Tensor();
    for (auto& t : g_wt.ssm_ffn_down) t = Tensor();
    // Clear weight tensor wrappers — chip 1
    for (auto& t : g_wt.attn_wo) t = Tensor();
    for (auto& t : g_wt.ssm_out) t = Tensor();
    g_wt.lm_head = Tensor();

    g_model.output_norm.reset();
    for (auto& l : g_model.attn_layers) {
        l.attn_norm.reset(); l.wqkv.reset(); l.attn_q_norm.reset();
        l.attn_k_norm.reset(); l.post_attn_norm.reset();
        l.ffn_gate_up.reset(); l.ffn_down.reset();
    }
    for (auto& l : g_model.ssm_layers) {
        l.attn_norm.reset(); l.w_combined.reset();
        l.post_attn_norm.reset(); l.ffn_gate_up.reset(); l.ffn_down.reset();
    }

    if (g_mesh2) {
        g_mesh2->close();
        g_mesh2.reset();
    }
    if (g_mesh) {
        g_mesh->close();
        g_mesh.reset();
    }
}
