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
static std::shared_ptr<MeshDevice> g_mesh;   // chip 0: layer weights + output projections (BFP8)
static std::shared_ptr<MeshDevice> g_mesh2;  // chip 1: LM head only
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
    Tensor attn_ffn_gate[8];     // [n_ff, n_embd] with fused SiLU
    Tensor attn_ffn_up[8];       // [n_ff, n_embd]
    Tensor attn_ffn_down[8];

    // Chip 0: SSM layers (24)
    Tensor ssm_w_combined[24];
    Tensor ssm_ffn_gate[24];     // [n_ff, n_embd] with fused SiLU
    Tensor ssm_ffn_up[24];       // [n_ff, n_embd]
    Tensor ssm_ffn_down[24];

    // Output projections (chip 0, BFP8) + LM head (chip 1, BFP8)
    Tensor attn_wo[8];           // [n_embd, n_head*head_dim] on chip 0
    Tensor ssm_out[24];          // [n_embd, ssm_d_inner] on chip 0
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
// Per-layer trace captures for on-device operations
// ============================================================================
// Trace for: rms_norm(hidden) → matmul(norm, weight) → result in gemv out buffer
static ttnn::MeshTraceId g_norm_matmul_traces[32];
static bool g_norm_matmul_traces_valid[32] = {};

// Trace for: add_(hidden, residual) → rms_norm → FFN chain → add_(hidden, ffn_out)
static ttnn::MeshTraceId g_ffn_chain_traces[32];
static bool g_ffn_chain_traces_valid[32] = {};

// Run norm + matmul on device (g_hidden_dev → norm → matmul → gemv out buffer)
static void norm_matmul_ops(const Tensor& norm_weight, const Tensor& weight,
                            uint32_t M, uint32_t K) {
    auto norm_out = ttnn::rms_norm(g_hidden_dev, MC::rms_norm_eps, norm_weight);
    auto& gb = get_gemv_buf(g_mesh.get(), M, K);
    ttnn::matmul(norm_out, weight,
                 false, true, ttnn::DRAM_MEMORY_CONFIG,
                 std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                 std::nullopt, std::nullopt, gb.out_tensor);
}

// Run outproj matmul + residual add + norm + FFN chain + residual add on device.
// Input: g_residual_dev contains ssm_proj_in or attn_out (written by host).
// Output: g_hidden_dev updated with outproj residual + FFN residual.
static void outproj_ffn_chain_ops(const Tensor& outproj_weight, uint32_t outproj_M, uint32_t outproj_K,
                                   const Tensor& norm_weight,
                                   const Tensor& gate_weight, const Tensor& up_weight,
                                   const Tensor& down_weight) {
    // 1. Output projection: matmul(residual, outproj_weight)
    auto& gb_op = get_gemv_buf(g_mesh.get(), outproj_M, outproj_K);
    ttnn::matmul(g_residual_dev, outproj_weight,
                 false, true, ttnn::DRAM_MEMORY_CONFIG,
                 std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                 std::nullopt, std::nullopt, gb_op.out_tensor);

    // 2. Residual add: hidden += outproj result
    ttnn::add_(g_hidden_dev, gb_op.out_tensor);

    // 3. RMSNorm
    auto norm_out = ttnn::rms_norm(g_hidden_dev, MC::rms_norm_eps, norm_weight);

    // 4. FFN: gate matmul (fused SiLU) + up matmul + multiply + down matmul
    auto& fb = get_ffn_buf(g_mesh.get());
    auto& gb_dn = get_gemv_buf(g_mesh.get(), MC::n_embd, MC::n_ff);

    // Gate with fused SiLU activation (eliminates separate slice + silu ops)
    ttnn::matmul(norm_out, gate_weight,
                 false, true, ttnn::DRAM_MEMORY_CONFIG,
                 std::nullopt, std::nullopt, std::string("silu"), std::nullopt,
                 std::nullopt, std::nullopt, fb.gate_tensor);

    // Up projection (no activation)
    ttnn::matmul(norm_out, up_weight,
                 false, true, ttnn::DRAM_MEMORY_CONFIG,
                 std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                 std::nullopt, std::nullopt, fb.up_tensor);

    // Gate * Up
    ttnn::multiply(fb.gate_tensor, fb.up_tensor, std::nullopt, ttnn::DRAM_MEMORY_CONFIG,
                   fb.act_tensor);

    // Down projection
    ttnn::matmul(fb.act_tensor, down_weight,
                 false, true, ttnn::DRAM_MEMORY_CONFIG,
                 std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                 std::nullopt, std::nullopt, gb_dn.out_tensor);

    // 5. Residual add: hidden += ffn_out
    ttnn::add_(g_hidden_dev, gb_dn.out_tensor);
}

// Separate tiled host buffer for write_f32_to_buf (declared before use)
static std::vector<bfloat16> g_write_host_tiled;

// Write f32 vector to a device MeshBuffer (tilize + enqueue on chip 0)
static void write_f32_to_buf(std::shared_ptr<MeshBuffer> buf, const float* data,
                              uint32_t len) {
    auto& cq = g_mesh->mesh_command_queue();
    uint32_t padded = ((len + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;
    size_t needed = (size_t)(padded / TILE_WIDTH) * TILE_HEIGHT * TILE_WIDTH;
    if (g_write_host_tiled.size() < needed)
        g_write_host_tiled.resize(needed, bfloat16(0.0f));

    uint16_t* scratch = g_bf16_scratch.data();
    const uint32_t* bits = reinterpret_cast<const uint32_t*>(data);
    for (uint32_t i = 0; i < len; i++)
        scratch[i] = static_cast<uint16_t>(bits[i] >> 16);

    uint16_t* ht = reinterpret_cast<uint16_t*>(g_write_host_tiled.data());
    uint32_t num_tile_cols = padded / TILE_WIDTH;
    for (uint32_t tc = 0; tc < num_tile_cols; tc++) {
        uint32_t tile_off = tc * TILE_HEIGHT * TILE_WIDTH;
        uint32_t base = tc * TILE_WIDTH;
        memcpy(ht + tile_off, scratch + base, 16 * sizeof(uint16_t));
        memcpy(ht + tile_off + 256, scratch + base + 16, 16 * sizeof(uint16_t));
    }
    EnqueueWriteMeshBuffer(cq, buf, g_write_host_tiled, false);
}

// Read gemv output buffer to host f32 (untilize + convert)
static void read_gemv_to_f32(GemvBuf& gb, float* out, uint32_t M) {
    uint16_t* scratch = g_bf16_scratch.data();
    const uint16_t* oht = reinterpret_cast<const uint16_t*>(gb.out_host_tiled.data());
    uint32_t out_tile_cols = gb.M_padded / TILE_WIDTH;
    for (uint32_t tc = 0; tc < out_tile_cols; tc++) {
        uint32_t tile_off = tc * TILE_HEIGHT * TILE_WIDTH;
        uint32_t base = tc * TILE_WIDTH;
        memcpy(scratch + base, oht + tile_off, 16 * sizeof(uint16_t));
        memcpy(scratch + base + 16, oht + tile_off + 256, 16 * sizeof(uint16_t));
    }
    uint32_t* ybits = reinterpret_cast<uint32_t*>(out);
    for (uint32_t i = 0; i < M; i++)
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

// Upload host bf16 data to a device as a weight Tensor [M, K] in tiled layout
static Tensor upload_bf16_weight(MeshDevice* device, const uint16_t* bf16_data,
                                  uint32_t M, uint32_t K, DataType dtype = DataType::BFLOAT8_B) {
    // Convert bf16 (uint16) to float for from_span
    size_t n = (size_t)M * K;
    std::vector<float> f32(n);
    const bfloat16* src = reinterpret_cast<const bfloat16*>(bf16_data);
    for (size_t i = 0; i < n; i++) f32[i] = static_cast<float>(src[i]);

    TensorSpec spec(
        Shape({1, 1, M, K}),
        TensorLayout(dtype, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));
    return Tensor::from_span<float>(std::span<const float>(f32.data(), n), spec, device);
}

// Read a BF16 tiled MeshBuffer back to host as flat f32 vector [M * K]
static std::vector<float> read_bf16_buf_to_f32(MeshCommandQueue& cq,
    std::shared_ptr<MeshBuffer> buf, uint32_t M, uint32_t K) {
    std::vector<bfloat16> tiled;
    EnqueueReadMeshBuffer(cq, tiled, buf, true);
    uint32_t M_pad = ((M + TILE_HEIGHT - 1) / TILE_HEIGHT) * TILE_HEIGHT;
    uint32_t K_pad = ((K + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;
    auto flat = untilize_nfaces(tiled, M_pad, K_pad);
    std::vector<float> f32(M * K);
    for (uint32_t r = 0; r < M; r++)
        for (uint32_t c = 0; c < K; c++)
            f32[r * K + c] = static_cast<float>(flat[r * K_pad + c]);
    return f32;
}

// Re-upload a BF16 device MeshBuffer as a BFLOAT8_B Tensor (halves DRAM usage)
static Tensor convert_to_bfp8(MeshDevice* device, MeshCommandQueue& cq,
                                std::shared_ptr<MeshBuffer>& buf, uint32_t M, uint32_t K) {
    auto f32 = read_bf16_buf_to_f32(cq, buf, M, K);
    buf.reset();  // free old BF16 buffer
    TensorSpec spec(
        Shape({1, 1, M, K}),
        TensorLayout(DataType::BFLOAT8_B, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));
    return Tensor::from_span<float>(std::span<const float>(f32.data(), f32.size()), spec, device);
}

// Split a combined [2*M, K] BF16 MeshBuffer into two BFP8 tensors: [M, K] each
static std::pair<Tensor, Tensor> split_gate_up_to_bfp8(
    MeshDevice* device, MeshCommandQueue& cq,
    std::shared_ptr<MeshBuffer>& buf, uint32_t M, uint32_t K) {
    auto f32 = read_bf16_buf_to_f32(cq, buf, 2 * M, K);
    buf.reset();  // free old BF16 buffer
    TensorSpec spec(Shape({1, 1, M, K}),
        TensorLayout(DataType::BFLOAT8_B, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));
    // Gate is first M rows, Up is last M rows
    auto gate = Tensor::from_span<float>(std::span<const float>(f32.data(), (size_t)M * K), spec, device);
    auto up = Tensor::from_span<float>(std::span<const float>(f32.data() + (size_t)M * K, (size_t)M * K), spec, device);
    return {std::move(gate), std::move(up)};
}

// Create Tensor wrappers for all on-device weight MeshBuffers
static void create_weight_tensors() {
    auto& cq0 = g_mesh->mesh_command_queue();

    // Re-upload chip 0 weights as BFLOAT8_B (halves DRAM reads, ~2x bandwidth)
    printf("Converting chip 0 weights to BFLOAT8_B...\n");
    int attn_idx = 0, ssm_idx = 0;
    for (int layer = 0; layer < MC::n_layers; layer++) {
        if (MC::is_recurrent(layer)) {
            auto& lw = g_model.ssm_layers[ssm_idx];
            uint32_t combined_rows = MC::ssm_conv_channels + MC::ssm_d_inner
                                   + MC::ssm_dt_rank + MC::ssm_dt_rank;
            g_wt.ssm_w_combined[ssm_idx] = convert_to_bfp8(g_mesh.get(), cq0,
                                                             lw.w_combined, combined_rows, MC::n_embd);
            auto [sg, su] = split_gate_up_to_bfp8(g_mesh.get(), cq0,
                                                      lw.ffn_gate_up, MC::n_ff, MC::n_embd);
            g_wt.ssm_ffn_gate[ssm_idx] = std::move(sg);
            g_wt.ssm_ffn_up[ssm_idx] = std::move(su);
            g_wt.ssm_ffn_down[ssm_idx] = convert_to_bfp8(g_mesh.get(), cq0,
                                                            lw.ffn_down, MC::n_embd, MC::n_ff);
            if ((ssm_idx + 1) % 6 == 0) printf("  SSM layers 0-%d converted\n", ssm_idx);
            ssm_idx++;
        } else {
            auto& lw = g_model.attn_layers[attn_idx];
            int q_dim = MC::n_head * MC::head_dim * 2;
            int kv_dim_one = MC::n_head_kv * MC::head_dim;
            int qkv_rows = q_dim + 2 * kv_dim_one;
            g_wt.attn_wqkv[attn_idx] = convert_to_bfp8(g_mesh.get(), cq0,
                                                          lw.wqkv, qkv_rows, MC::n_embd);
            auto [ag, au] = split_gate_up_to_bfp8(g_mesh.get(), cq0,
                                                      lw.ffn_gate_up, MC::n_ff, MC::n_embd);
            g_wt.attn_ffn_gate[attn_idx] = std::move(ag);
            g_wt.attn_ffn_up[attn_idx] = std::move(au);
            g_wt.attn_ffn_down[attn_idx] = convert_to_bfp8(g_mesh.get(), cq0,
                                                              lw.ffn_down, MC::n_embd, MC::n_ff);
            printf("  Attn layer %d converted\n", attn_idx);
            attn_idx++;
        }
    }
    printf("Converted %d attention + %d SSM weight tensors to BFLOAT8_B on chip 0.\n",
           attn_idx, ssm_idx);

    // All output projections + LM head now on chip 0 too (BFP8 fits!)
    printf("Uploading output projections + LM head to chip 0 as BFLOAT8_B...\n");
    for (int i = 0; i < 8; i++) {
        auto& lw = g_model.attn_layers[i];
        g_wt.attn_wo[i] = upload_bf16_weight(g_mesh.get(), lw.wo_host.data(),
                                              MC::n_embd, MC::n_head * MC::head_dim);
        printf("  wo[%d] uploaded\n", i);
    }
    for (int i = 0; i < 24; i++) {
        auto& lw = g_model.ssm_layers[i];
        g_wt.ssm_out[i] = upload_bf16_weight(g_mesh.get(), lw.ssm_out_host.data(),
                                              MC::n_embd, MC::ssm_d_inner);
        printf("  ssm_out[%d] uploaded\n", i);
    }
    g_wt.lm_head = upload_bf16_weight(g_mesh2.get(), g_model.output_host.data(),
                                       MC::n_vocab, MC::n_embd);
    printf("  lm_head uploaded\n");

    // Free host bf16 vectors now that weights are on device
    for (int i = 0; i < 8; i++)
        g_model.attn_layers[i].wo_host.clear();
    for (int i = 0; i < 24; i++)
        g_model.ssm_layers[i].ssm_out_host.clear();
    g_model.output_host.clear();
    printf("Freed host bf16 vectors — saved ~2.9 GB host RAM\n");

    // Wrap norm weights as device Tensors for on-device RMSNorm
    printf("Creating norm weight tensors...\n");
    auto wrap_1d = [](std::shared_ptr<MeshBuffer> buf, uint32_t len) -> Tensor {
        uint32_t padded = ((len + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;
        DeviceStorage storage(buf, {MeshCoordinate(0, 0)});
        TensorSpec spec(
            Shape({1, 1, 1, padded}),
            TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));
        return Tensor(std::move(storage), spec, TensorTopology{});
    };

    attn_idx = 0; ssm_idx = 0;
    for (int layer = 0; layer < MC::n_layers; layer++) {
        if (MC::is_recurrent(layer)) {
            auto& lw = g_model.ssm_layers[ssm_idx];
            g_wt.attn_norm[layer] = wrap_1d(lw.attn_norm, MC::n_embd);
            g_wt.post_norm[layer] = wrap_1d(lw.post_attn_norm, MC::n_embd);
            ssm_idx++;
        } else {
            auto& lw = g_model.attn_layers[attn_idx];
            g_wt.attn_norm[layer] = wrap_1d(lw.attn_norm, MC::n_embd);
            g_wt.post_norm[layer] = wrap_1d(lw.post_attn_norm, MC::n_embd);
            attn_idx++;
        }
    }
    g_wt.output_norm_tensor = wrap_1d(g_model.output_norm, MC::n_embd);

    // Create persistent hidden state + residual + norm output buffers on chip 0
    printf("Creating persistent device buffers...\n");
    uint32_t n_embd_padded = ((MC::n_embd + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;
    uint32_t embd_tiles = n_embd_padded / TILE_WIDTH;
    uint32_t tile_bytes = TILE_HEIGHT * TILE_WIDTH * sizeof(bfloat16);
    DeviceLocalBufferConfig dram_cfg{.page_size = tile_bytes, .buffer_type = BufferType::DRAM};

    g_hidden_dev_buf = MeshBuffer::create(ReplicatedBufferConfig{.size = embd_tiles * tile_bytes},
                                           dram_cfg, g_mesh.get());
    g_residual_dev_buf = MeshBuffer::create(ReplicatedBufferConfig{.size = embd_tiles * tile_bytes},
                                             dram_cfg, g_mesh.get());
    g_norm_dev_buf = MeshBuffer::create(ReplicatedBufferConfig{.size = embd_tiles * tile_bytes},
                                         dram_cfg, g_mesh.get());

    TensorSpec embd_spec(
        Shape({1, 1, 1, n_embd_padded}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));
    g_hidden_dev = Tensor(DeviceStorage(g_hidden_dev_buf, {MeshCoordinate(0, 0)}),
                           embd_spec, TensorTopology{});
    g_residual_dev = Tensor(DeviceStorage(g_residual_dev_buf, {MeshCoordinate(0, 0)}),
                             embd_spec, TensorTopology{});
    g_norm_dev = Tensor(DeviceStorage(g_norm_dev_buf, {MeshCoordinate(0, 0)}),
                         embd_spec, TensorTopology{});
    printf("Persistent device buffers created.\n");
}

// ============================================================================
// Device hidden state write/read helpers
// ============================================================================
static std::vector<bfloat16> g_dev_host_tiled;  // for PCIe transfers of [1, n_embd] tiled

static void write_hidden_to_device(const float* f32_data) {
    auto& cq = g_mesh->mesh_command_queue();
    uint32_t n_embd_padded = ((MC::n_embd + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;
    if (g_dev_host_tiled.empty())
        g_dev_host_tiled.resize(n_embd_padded / TILE_WIDTH * TILE_HEIGHT * TILE_WIDTH, bfloat16(0.0f));

    // Convert f32→bf16 and tilize
    uint16_t* scratch = g_bf16_scratch.data();
    const uint32_t* bits = reinterpret_cast<const uint32_t*>(f32_data);
    for (uint32_t i = 0; i < (uint32_t)MC::n_embd; i++)
        scratch[i] = static_cast<uint16_t>(bits[i] >> 16);

    uint16_t* ht = reinterpret_cast<uint16_t*>(g_dev_host_tiled.data());
    uint32_t num_tile_cols = n_embd_padded / TILE_WIDTH;
    for (uint32_t tc = 0; tc < num_tile_cols; tc++) {
        uint32_t tile_off = tc * TILE_HEIGHT * TILE_WIDTH;
        uint32_t base = tc * TILE_WIDTH;
        memcpy(ht + tile_off, scratch + base, 16 * sizeof(uint16_t));
        memcpy(ht + tile_off + 256, scratch + base + 16, 16 * sizeof(uint16_t));
    }
    EnqueueWriteMeshBuffer(cq, g_hidden_dev_buf, g_dev_host_tiled, false);
}

// Read a device buffer to host f32 (untilize + convert)
static void read_device_to_f32(std::shared_ptr<MeshBuffer> buf, float* out, uint32_t len,
                                MeshCommandQueue& cq) {
    uint32_t padded = ((len + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;
    if (g_dev_host_tiled.size() < padded / TILE_WIDTH * TILE_HEIGHT * TILE_WIDTH)
        g_dev_host_tiled.resize(padded / TILE_WIDTH * TILE_HEIGHT * TILE_WIDTH, bfloat16(0.0f));

    EnqueueReadMeshBuffer(cq, g_dev_host_tiled, buf, true);

    uint16_t* scratch = g_bf16_scratch.data();
    const uint16_t* oht = reinterpret_cast<const uint16_t*>(g_dev_host_tiled.data());
    uint32_t out_tile_cols = padded / TILE_WIDTH;
    for (uint32_t tc = 0; tc < out_tile_cols; tc++) {
        uint32_t tile_off = tc * TILE_HEIGHT * TILE_WIDTH;
        uint32_t base = tc * TILE_WIDTH;
        memcpy(scratch + base, oht + tile_off, 16 * sizeof(uint16_t));
        memcpy(scratch + base + 16, oht + tile_off + 256, 16 * sizeof(uint16_t));
    }
    uint32_t* ybits = reinterpret_cast<uint32_t*>(out);
    for (uint32_t i = 0; i < len; i++)
        ybits[i] = static_cast<uint32_t>(scratch[i]) << 16;
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
static double g_time_outproj = 0, g_time_reswrite = 0, g_time_host = 0, g_time_norm_mm = 0;

static float* forward_decode() {
    using Clock = std::chrono::high_resolution_clock;
    int pos = g_pos;
    auto& cq0 = g_mesh->mesh_command_queue();

    // Write hidden state to device (chip 0) — stays on device through all layers
    write_hidden_to_device(g_hidden_f32.data());

    int attn_idx = 0, ssm_idx = 0;
    for (int layer = 0; layer < MC::n_layers; layer++) {

        if (MC::is_recurrent(layer)) {
            // ======== SSM (Delta-Net) Layer ========
            auto& lw = g_model.ssm_layers[ssm_idx];

            // 1. Norm + combined matmul ON DEVICE (eliminates norm_out PCIe write)
            auto& gb_comb = get_gemv_buf(g_mesh.get(), g_combined_rows, MC::n_embd);
            auto t0 = Clock::now();

            if (!g_norm_matmul_traces_valid[layer]) {
                // Warmup
                norm_matmul_ops(g_wt.attn_norm[layer], g_wt.ssm_w_combined[ssm_idx],
                               g_combined_rows, MC::n_embd);
                EnqueueReadMeshBuffer(cq0, gb_comb.out_host_tiled, gb_comb.out_buf, true);
                // Capture trace
                auto tid = ttnn::operations::trace::begin_trace_capture(g_mesh.get(), std::nullopt);
                norm_matmul_ops(g_wt.attn_norm[layer], g_wt.ssm_w_combined[ssm_idx],
                               g_combined_rows, MC::n_embd);
                ttnn::operations::trace::end_trace_capture(g_mesh.get(), tid, std::nullopt);
                g_norm_matmul_traces[layer] = tid;
                g_norm_matmul_traces_valid[layer] = true;
            } else {
                ttnn::operations::trace::execute_trace(g_mesh.get(), g_norm_matmul_traces[layer],
                                                        std::nullopt, false);
            }
            EnqueueReadMeshBuffer(cq0, gb_comb.out_host_tiled, gb_comb.out_buf, true);

            g_time_norm_mm += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

            // Untilize combined result
            read_gemv_to_f32(gb_comb, g_proj.data(), g_combined_rows);

            auto t_host0 = Clock::now();
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

            // 4. Delta-net recurrence + 5. Gated RMSNorm (parallelized across v-heads)
            auto& state = g_ssm_state[ssm_idx];
            float* ssm_proj_in = g_ssm_proj_in.data();
            constexpr float ssm_scale = 1.0f / 11.3137f;


            for (int vh = 0; vh < num_v; vh++) {
                int kh = vh % num_k_heads;
                float q[head_k], k[head_k], v[head_v];
                memcpy(q, conv_q + kh * head_k, head_k * sizeof(float));
                memcpy(k, conv_k + kh * head_k, head_k * sizeof(float));
                memcpy(v, conv_v + vh * head_v, head_v * sizeof(float));
                float qn = 0, kn = 0;
                for (int d = 0; d < head_k; d++) { qn += q[d]*q[d]; kn += k[d]*k[d]; }
                qn = 1.0f / sqrtf(qn + MC::rms_norm_eps);
                kn = 1.0f / sqrtf(kn + MC::rms_norm_eps);
                for (int d = 0; d < head_k; d++) { q[d] *= qn; k[d] *= kn; }
                float biased = alpha_raw[vh] + lw.ssm_dt_bias_host[vh];
                float sp = (biased > 20.0f) ? biased : logf(1.0f + expf(biased));
                float gate = sp * lw.ssm_a_host[vh];
                float decay = expf(gate);
                float beta_val = 1.0f / (1.0f + expf(-beta_raw[vh]));
                float* sh = state.data() + vh * head_v * head_k;
                for (int j = 0; j < head_v * head_k; j++) sh[j] *= decay;
                for (int i = 0; i < head_v; i++) {
                    float* row = sh + i * head_k;
                    float sk = 0;
                    for (int j = 0; j < head_k; j++) sk += row[j] * k[j];
                    float dd = beta_val * (v[i] - sk);
                    for (int j = 0; j < head_k; j++) row[j] += k[j] * dd;
                    float out = 0;
                    for (int j = 0; j < head_k; j++) out += row[j] * q[j];
                    // Fused gated RMSNorm accumulation
                    ssm_proj_in[vh * head_v + i] = out * ssm_scale;
                }

                // Gated RMSNorm for this v-head
                float sum_sq = 0;
                float* vout = ssm_proj_in + vh * head_v;
                for (int d = 0; d < head_v; d++) sum_sq += vout[d] * vout[d];
                float rms = 1.0f / sqrtf(sum_sq / head_v + MC::rms_norm_eps);
                for (int d = 0; d < head_v; d++) {
                    float normalized = vout[d] * rms * lw.ssm_norm_host[d];
                    float z = z_raw[vh * head_v + d];
                    float silu_z = z / (1.0f + expf(-z));
                    vout[d] = normalized * silu_z;
                }
            }

            g_time_host += std::chrono::duration<double, std::milli>(Clock::now() - t_host0).count();

            // 6. Write ssm_proj_in to g_residual_dev, then outproj+FFN chain ON DEVICE
            auto t_rw = Clock::now();
            write_f32_to_buf(g_residual_dev_buf, ssm_proj_in, MC::ssm_d_inner);
            g_time_reswrite += std::chrono::duration<double, std::milli>(Clock::now() - t_rw).count();

            auto t2 = Clock::now();
            if (!g_ffn_chain_traces_valid[layer]) {
                outproj_ffn_chain_ops(g_wt.ssm_out[ssm_idx], MC::n_embd, MC::ssm_d_inner,
                                      g_wt.post_norm[layer],
                                      g_wt.ssm_ffn_gate[ssm_idx], g_wt.ssm_ffn_up[ssm_idx],
                                      g_wt.ssm_ffn_down[ssm_idx]);
                Finish(cq0);
                auto tid = ttnn::operations::trace::begin_trace_capture(g_mesh.get(), std::nullopt);
                outproj_ffn_chain_ops(g_wt.ssm_out[ssm_idx], MC::n_embd, MC::ssm_d_inner,
                                      g_wt.post_norm[layer],
                                      g_wt.ssm_ffn_gate[ssm_idx], g_wt.ssm_ffn_up[ssm_idx],
                                      g_wt.ssm_ffn_down[ssm_idx]);
                ttnn::operations::trace::end_trace_capture(g_mesh.get(), tid, std::nullopt);
                g_ffn_chain_traces[layer] = tid;
                g_ffn_chain_traces_valid[layer] = true;
            } else {
                ttnn::operations::trace::execute_trace(g_mesh.get(), g_ffn_chain_traces[layer],
                                                        std::nullopt, false);
            }
            g_time_ffn += std::chrono::duration<double, std::milli>(Clock::now() - t2).count();
            // Hidden state stays on device — no read needed!

            ssm_idx++;
        } else {
            // ======== Full Attention Layer ========
            auto& lw = g_model.attn_layers[attn_idx];
            auto& aw = g_attn_small[attn_idx];

            // 1. Norm + QKV matmul ON DEVICE
            auto& gb_qkv = get_gemv_buf(g_mesh.get(), g_qkv_rows, MC::n_embd);
            auto t0 = Clock::now();

            if (!g_norm_matmul_traces_valid[layer]) {
                norm_matmul_ops(g_wt.attn_norm[layer], g_wt.attn_wqkv[attn_idx],
                               g_qkv_rows, MC::n_embd);
                EnqueueReadMeshBuffer(cq0, gb_qkv.out_host_tiled, gb_qkv.out_buf, true);
                auto tid = ttnn::operations::trace::begin_trace_capture(g_mesh.get(), std::nullopt);
                norm_matmul_ops(g_wt.attn_norm[layer], g_wt.attn_wqkv[attn_idx],
                               g_qkv_rows, MC::n_embd);
                ttnn::operations::trace::end_trace_capture(g_mesh.get(), tid, std::nullopt);
                g_norm_matmul_traces[layer] = tid;
                g_norm_matmul_traces_valid[layer] = true;
            } else {
                ttnn::operations::trace::execute_trace(g_mesh.get(), g_norm_matmul_traces[layer],
                                                        std::nullopt, false);
            }
            EnqueueReadMeshBuffer(cq0, gb_qkv.out_host_tiled, gb_qkv.out_buf, true);
            g_time_norm_mm += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

            // Untilize QKV result
            constexpr int q_dim = MC::n_head * MC::head_dim * 2;
            constexpr int kv_dim_one = MC::n_head_kv * MC::head_dim;
            float* qkv = g_qkv.data();
            read_gemv_to_f32(gb_qkv, qkv, g_qkv_rows);

            auto t_host1 = Clock::now();
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

            // 4. RoPE
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

            // 6. Attention (online softmax) + 7. Sigmoid gating — parallelized
            float* attn_out = g_attn_out.data();

            for (int h = 0; h < MC::n_head; h++) {
                int kv_h = h / (MC::n_head / MC::n_head_kv);
                float* qh = q_heads + h * MC::head_dim;
                float* out = attn_out + h * MC::head_dim;
                float acc[MC::head_dim];
                memset(acc, 0, MC::head_dim * sizeof(float));
                float max_score = -FLT_MAX, sum_exp = 0;
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
                // Output + sigmoid gating fused
                float* gh = gate_heads + h * MC::head_dim;
                for (int d = 0; d < MC::head_dim; d++)
                    out[d] = (acc[d] / sum_exp) * (1.0f / (1.0f + expf(-gh[d])));
            }
            g_time_host += std::chrono::duration<double, std::milli>(Clock::now() - t_host1).count();

            // 8. Write attn_out to g_residual_dev, then outproj+FFN chain ON DEVICE
            auto t_rw2 = Clock::now();
            write_f32_to_buf(g_residual_dev_buf, attn_out, MC::n_head * MC::head_dim);
            g_time_reswrite += std::chrono::duration<double, std::milli>(Clock::now() - t_rw2).count();

            auto t2 = Clock::now();
            if (!g_ffn_chain_traces_valid[layer]) {
                outproj_ffn_chain_ops(g_wt.attn_wo[attn_idx], MC::n_embd, MC::n_head * MC::head_dim,
                                      g_wt.post_norm[layer],
                                      g_wt.attn_ffn_gate[attn_idx], g_wt.attn_ffn_up[attn_idx],
                                      g_wt.attn_ffn_down[attn_idx]);
                Finish(cq0);
                auto tid = ttnn::operations::trace::begin_trace_capture(g_mesh.get(), std::nullopt);
                outproj_ffn_chain_ops(g_wt.attn_wo[attn_idx], MC::n_embd, MC::n_head * MC::head_dim,
                                      g_wt.post_norm[layer],
                                      g_wt.attn_ffn_gate[attn_idx], g_wt.attn_ffn_up[attn_idx],
                                      g_wt.attn_ffn_down[attn_idx]);
                ttnn::operations::trace::end_trace_capture(g_mesh.get(), tid, std::nullopt);
                g_ffn_chain_traces[layer] = tid;
                g_ffn_chain_traces_valid[layer] = true;
            } else {
                ttnn::operations::trace::execute_trace(g_mesh.get(), g_ffn_chain_traces[layer],
                                                        std::nullopt, false);
            }
            g_time_ffn += std::chrono::duration<double, std::milli>(Clock::now() - t2).count();

            attn_idx++;
        }
    }

    // Read hidden from device for final norm + LM head
    read_device_to_f32(g_hidden_dev_buf, g_hidden_f32.data(), MC::n_embd, cq0);

    // Final norm (host)
    float* norm_out = g_norm_out.data();
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
        int dc = g_decode_count;
        printf("  [profile @%d] norm_mm=%.0f outproj=%.0f ffn=%.0f host=%.0f reswr=%.0f lmhead=%.0f ms/tok\n",
               dc, g_time_norm_mm / dc, g_time_outproj / dc, g_time_ffn / dc,
               g_time_host / dc, g_time_reswrite / dc, g_time_lmhead / dc);
        printf("    outproj_gemv: tilize=%.2f enq=%.2f disp=%.2f read=%.2f host=%.2f ms/call (%d calls)\n",
               g_t_tilize / calls, g_t_enqueue / calls, g_t_dispatch / calls, g_t_read / calls, g_t_host / calls, calls);
    }

    return logits;
}

// ============================================================================
// Public API
// ============================================================================

bool load_model_and_tokenizer(const char* model_path, int max_ctx) {
    printf("Loading model from %s (max_ctx=%d)...\n", model_path, max_ctx);

    // Open chip 0 for layer weights + output projections, chip 1 for LM head
    auto meshes = MeshDevice::create_unit_meshes({0, 1});
    g_mesh = meshes[0];
    g_mesh2 = meshes[1];

    auto grid = g_mesh->compute_with_storage_grid_size();
    printf("Chip 0 opened: compute grid %zux%zu (%zu cores)\n",
           grid.x, grid.y, grid.x * grid.y);

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

    // Release norm+matmul traces (chip 0)
    for (int i = 0; i < 32; i++) {
        if (g_norm_matmul_traces_valid[i]) {
            ttnn::operations::trace::release_trace(g_mesh.get(), g_norm_matmul_traces[i]);
            g_norm_matmul_traces_valid[i] = false;
        }
    }
    // Release FFN chain traces (chip 0)
    for (int i = 0; i < 32; i++) {
        if (g_ffn_chain_traces_valid[i]) {
            ttnn::operations::trace::release_trace(g_mesh.get(), g_ffn_chain_traces[i]);
            g_ffn_chain_traces_valid[i] = false;
        }
    }
    // Release gemv traces (chip 0 for all except idx 64 = LM head on chip 1)
    for (int i = 0; i < 128; i++) {
        if (g_gemv_traces[i].valid) {
            MeshDevice* dev = (i == 64) ? g_mesh2.get() : g_mesh.get();
            ttnn::operations::trace::release_trace(dev, g_gemv_traces[i].trace_id);
            g_gemv_traces[i].valid = false;
        }
    }

    // Clear pre-allocated GEMV and FFN buffers
    g_ffn_bufs.clear();
    g_gemv_bufs.clear();

    // Clear weight tensor wrappers — chip 0
    for (auto& t : g_wt.attn_wqkv) t = Tensor();
    for (auto& t : g_wt.attn_ffn_gate) t = Tensor();
    for (auto& t : g_wt.attn_ffn_up) t = Tensor();
    for (auto& t : g_wt.attn_ffn_down) t = Tensor();
    for (auto& t : g_wt.ssm_w_combined) t = Tensor();
    for (auto& t : g_wt.ssm_ffn_gate) t = Tensor();
    for (auto& t : g_wt.ssm_ffn_up) t = Tensor();
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
