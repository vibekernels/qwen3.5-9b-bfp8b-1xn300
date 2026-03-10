// SPDX-License-Identifier: Apache-2.0
// Qwen3.5-9B inference engine for Tenstorrent N300 via tt-metal.
//
// All matmuls run on-device via custom DRAM-sharded GEMV kernels on Tensix cores.
// Small element-wise ops (RoPE, gating, SSM recurrence) remain on host CPU.
// No ttnn dependency — uses only tt-metalium APIs.

#include "engine.h"
#include "model_config.h"
#include "gguf_loader.h"
#include "tokenizer.h"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>

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
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <immintrin.h>
#include <sys/stat.h>

// Suppress deprecation warnings from tt-metal's transitional global-scope aliases
// (CoreCoord, CoreRange, CoreRangeSet, stl→ttsl) — we use the tt::tt_metal:: versions via using-namespace.
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

// blockfloat_common not installed in build; include via TT_METAL_SRC_PATH
#include "blockfloat_common.hpp"

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using MC = ModelConfig;

// ============================================================================
// Global state
// ============================================================================
static std::shared_ptr<MeshDevice> g_mesh;    // primary mesh for all current ops (1-chip submesh of g_mesh2)
static std::shared_ptr<MeshDevice> g_mesh1;   // chip 1 submesh (for future TP)
static std::shared_ptr<MeshDevice> g_mesh2;   // 2-chip parent mesh (1×2)
static ModelBuffers g_model;
static Tokenizer g_tokenizer;
static bool g_loaded = false;
static bool g_verbose = true;  // set false via QUIET=1 env var
static int g_max_ctx = 0;
static int g_pos = 0;

// DRAM bank topology for sharded GEMV
static uint32_t g_num_dram_banks = 0;
static std::vector<CoreCoord> g_dram_workers;  // optimal Tensix worker per DRAM bank (chip 0)
static uint32_t g_num_dram_banks_1 = 0;
static std::vector<CoreCoord> g_dram_workers_1;  // optimal Tensix worker per DRAM bank (chip 1)

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

// Prefix cache: tokens already processed (prompt + generated), for reuse across generate() calls
static std::vector<int> g_cached_tokens;

// ============================================================================
// Cached weight MeshBuffers (on-device, DRAM-sharded BFP8_B tiles)
// ============================================================================
struct WeightBuffers {
    // Chip 0: Attention layers (8) — MeshBuffers for custom kernel dispatch
    std::shared_ptr<MeshBuffer> attn_wqkv_buf[8];
    std::shared_ptr<MeshBuffer> attn_ffn_gate_buf[8];
    std::shared_ptr<MeshBuffer> attn_ffn_up_buf[8];
    std::shared_ptr<MeshBuffer> attn_ffn_down_buf[8];
    std::shared_ptr<MeshBuffer> attn_wo_buf[8];

    // Chip 0: SSM layers (24) — MeshBuffers for custom kernel dispatch
    std::shared_ptr<MeshBuffer> ssm_w_combined_buf[24];
    std::shared_ptr<MeshBuffer> ssm_ffn_gate_buf[24];
    std::shared_ptr<MeshBuffer> ssm_ffn_up_buf[24];
    std::shared_ptr<MeshBuffer> ssm_ffn_down_buf[24];
    std::shared_ptr<MeshBuffer> ssm_out_buf[24];

    // Chip 0: Combined gate+up weight buffers (fused GEMV for FFN)
    std::shared_ptr<MeshBuffer> ssm_ffn_gate_up_buf[24];
    std::shared_ptr<MeshBuffer> attn_ffn_gate_up_buf[8];

    // LM head
    std::shared_ptr<MeshBuffer> lm_head_buf;

    // Norm weight buffers (BF16 on device, for custom dispatch_rmsnorm)
    std::shared_ptr<MeshBuffer> attn_norm_buf[32];
    std::shared_ptr<MeshBuffer> post_norm_buf[32];
    std::shared_ptr<MeshBuffer> output_norm_buf;

    // Chip 1: pre-layer GEMV weight halves (column-parallel, bottom M/2 rows)
    std::shared_ptr<MeshBuffer> ssm_w_combined_buf_1[24];
    std::shared_ptr<MeshBuffer> attn_wqkv_buf_1[8];
    std::shared_ptr<MeshBuffer> lm_head_buf_1;

    // Split dimensions (tile counts for each half)
    uint32_t ssm_combined_Mt0[24], ssm_combined_Mt1[24];  // tile rows per chip
    uint32_t attn_qkv_Mt0[8], attn_qkv_Mt1[8];
    uint32_t lm_head_Mt0, lm_head_Mt1;

    // Chip 1: tensor-parallel FFN weight halves (second half of gate/up rows, second half of down cols)
    std::shared_ptr<MeshBuffer> ssm_ffn_gate_buf_1[24];
    std::shared_ptr<MeshBuffer> ssm_ffn_up_buf_1[24];
    std::shared_ptr<MeshBuffer> ssm_ffn_down_buf_1[24];
    std::shared_ptr<MeshBuffer> attn_ffn_gate_buf_1[8];
    std::shared_ptr<MeshBuffer> attn_ffn_up_buf_1[8];
    std::shared_ptr<MeshBuffer> attn_ffn_down_buf_1[8];
    // Chip 1: Combined gate+up weight buffers (fused GEMV for FFN)
    std::shared_ptr<MeshBuffer> ssm_ffn_gate_up_buf_1[24];
    std::shared_ptr<MeshBuffer> attn_ffn_gate_up_buf_1[8];
    // Chip 1: replicated outproj weights (full copy for independent outproj_resadd)
    std::shared_ptr<MeshBuffer> ssm_out_buf_1[24];
    std::shared_ptr<MeshBuffer> attn_wo_buf_1[8];
    // Chip 1: norm weights for FFN chain (post_norm for rmsnorm_fpu)
    std::shared_ptr<MeshBuffer> post_norm_buf_1[32];
    // Chip 1: pre-layer norm weights
    std::shared_ptr<MeshBuffer> attn_norm_buf_1[32];

};
static WeightBuffers g_wt;

// Persistent hidden state on device (avoids PCIe round-trips between layers)
static std::shared_ptr<MeshBuffer> g_hidden_dev_buf;
static std::shared_ptr<MeshBuffer> g_residual_dev_buf;
// Temp buffer for matmul input after rms_norm (on device)
static std::shared_ptr<MeshBuffer> g_norm_dev_buf;

// Chip 1 persistent device buffers
static std::shared_ptr<MeshBuffer> g_hidden_dev_buf_1;
static std::shared_ptr<MeshBuffer> g_residual_dev_buf_1;
static std::shared_ptr<MeshBuffer> g_norm_dev_buf_1;
static std::shared_ptr<MeshBuffer> g_partial_down_buf;     // chip 0 partial down output
static std::shared_ptr<MeshBuffer> g_partial_down_buf_1;   // chip 1 partial down output

// TP FFN: half-size intermediate buffers on each chip
static constexpr uint32_t n_ff_tp = MC::n_ff / 2;           // 6144
static constexpr uint32_t n_ff_tp_tiles = n_ff_tp / TILE_WIDTH;  // 192
struct TpFfnBuf {
    std::shared_ptr<MeshBuffer> gate_buf;   // [1, n_ff/2] tiled
    std::shared_ptr<MeshBuffer> up_buf;     // [1, n_ff/2] tiled
    std::shared_ptr<MeshBuffer> act_buf;    // [1, n_ff/2] tiled
    bool initialized = false;
};
static TpFfnBuf g_tp_ffn_0, g_tp_ffn_1;

// Host-side buffer for reading partial down from chip 1
static std::vector<float> g_partial_f32(MC::n_embd);
// Separate tiled host buffer for chip 1 reads (avoids contention with g_dev_host_tiled)
static std::vector<bfloat16> g_dev_host_tiled_1;

// ============================================================================
// Pre-allocated device buffers for fast GEMV (avoids per-call alloc/dealloc)
// ============================================================================
struct GemvBuf {
    std::shared_ptr<MeshBuffer> act_buf;    // device-side activation buffer
    std::vector<bfloat16> act_host_tiled;   // pre-allocated host tilized buffer
    uint32_t K_padded;

    std::shared_ptr<MeshBuffer> out_buf;    // device-side output buffer
    std::vector<bfloat16> out_host_tiled;   // pre-allocated host read buffer
    uint32_t M_padded;
};

// Map from (device_ptr, K, M) -> pre-allocated buffers
static std::map<std::tuple<MeshDevice*, uint32_t, uint32_t>, GemvBuf> g_gemv_bufs;

// Device-aware DRAM bank topology lookup (forward declarations)
static uint32_t get_num_dram_banks(MeshDevice* device);
static const std::vector<CoreCoord>& get_dram_workers(MeshDevice* device);

static GemvBuf& get_gemv_buf(MeshDevice* device, uint32_t M, uint32_t K) {
    auto key = std::make_tuple(device, K, M);
    auto it = g_gemv_bufs.find(key);
    if (it != g_gemv_bufs.end()) return it->second;

    // Create new pre-allocated buffers
    GemvBuf buf;
    uint32_t TW = TILE_WIDTH, TH = TILE_HEIGHT;
    buf.K_padded = ((K + TW - 1) / TW) * TW;
    // Pad M to next multiple of num_dram_banks for DRAM-sharded dispatch
    uint32_t Mt_raw = (M + TH - 1) / TH;
    uint32_t nbanks = (get_num_dram_banks(device) > 0) ? get_num_dram_banks(device) : 12;
    uint32_t Mt_padded = ((Mt_raw + nbanks - 1) / nbanks) * nbanks;
    buf.M_padded = Mt_padded * TH;

    uint32_t act_tiles = buf.K_padded / TW;
    uint32_t out_tiles = Mt_padded;
    uint32_t tile_bytes = TH * TW * sizeof(bfloat16);

    DeviceLocalBufferConfig dram_cfg{.page_size = tile_bytes, .buffer_type = BufferType::DRAM};

    buf.act_buf = MeshBuffer::create(ReplicatedBufferConfig{.size = act_tiles * tile_bytes},
                                      dram_cfg, device);
    buf.act_host_tiled.resize(act_tiles * TH * TW, bfloat16(0.0f));

    buf.out_buf = MeshBuffer::create(ReplicatedBufferConfig{.size = out_tiles * tile_bytes},
                                      dram_cfg, device);
    buf.out_host_tiled.resize(out_tiles * TH * TW, bfloat16(0.0f));

    auto [ins, _] = g_gemv_bufs.emplace(key, std::move(buf));
    printf("  [gemv_buf] allocated K=%u M=%u on device\n", K, M); fflush(stdout);
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

// ============================================================================
// Custom tt-metal kernel dispatch (replaces ttnn ops)
// ============================================================================
static std::string kernel_path(const char* rel) {
    return std::string(KERNEL_DIR) + "/" + rel;
}

// Device-aware DRAM bank topology lookup
static uint32_t get_num_dram_banks(MeshDevice* device) {
    return (device == g_mesh.get()) ? g_num_dram_banks : g_num_dram_banks_1;
}
static const std::vector<CoreCoord>& get_dram_workers(MeshDevice* device) {
    return (device == g_mesh.get()) ? g_dram_workers : g_dram_workers_1;
}

// Cached eltwise binary workload: created once, reused across calls (trace-compatible).
// Key = (op_type, src0_addr, src1_addr, dst_addr, n_tiles)
struct CachedEltwiseWorkload {
    MeshWorkload workload;
    bool valid = false;
};
static std::map<std::tuple<uintptr_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t>, CachedEltwiseWorkload> g_eltwise_cache;

// Dispatch an elementwise binary op (add or multiply) on device.
// op_type: 0 = add, 1 = multiply
// Result is written to dst_buf.
// MeshWorkload is cached and reused — trace-compatible after first warmup call.
static void dispatch_eltwise_binary(MeshDevice* device, uint32_t op_type,
                                     std::shared_ptr<MeshBuffer> src0_buf,
                                     std::shared_ptr<MeshBuffer> src1_buf,
                                     std::shared_ptr<MeshBuffer> dst_buf,
                                     uint32_t n_tiles) {
    auto& cq = device->mesh_command_queue();
    auto key = std::make_tuple((uintptr_t)device, op_type, (uint32_t)src0_buf->address(),
                               (uint32_t)src1_buf->address(), (uint32_t)dst_buf->address(), n_tiles);
    auto& cached = g_eltwise_cache[key];

    if (!cached.valid) {
        MeshCoordinateRange dev_range(device->shape());
        Program program = CreateProgram();
        tt::tt_metal::CoreCoord core = {0, 0};

        uint32_t tile_bytes = TILE_HEIGHT * TILE_WIDTH * sizeof(bfloat16);

        CircularBufferConfig cb0_cfg =
            CircularBufferConfig(2 * tile_bytes, {{CBIndex::c_0, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_0, tile_bytes);
        CreateCircularBuffer(program, core, cb0_cfg);

        CircularBufferConfig cb1_cfg =
            CircularBufferConfig(2 * tile_bytes, {{CBIndex::c_1, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_1, tile_bytes);
        CreateCircularBuffer(program, core, cb1_cfg);

        CircularBufferConfig cb_out_cfg =
            CircularBufferConfig(2 * tile_bytes, {{CBIndex::c_16, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_16, tile_bytes);
        CreateCircularBuffer(program, core, cb_out_cfg);

        std::vector<uint32_t> reader_ct_args = {(uint32_t)CBIndex::c_0, (uint32_t)CBIndex::c_1};
        TensorAccessorArgs(*src0_buf).append_to(reader_ct_args);
        TensorAccessorArgs(*src1_buf).append_to(reader_ct_args);

        auto reader_kid = CreateKernel(program,
            kernel_path("dataflow/reader_binary_tiles.cpp"), core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc = NOC::RISCV_1_default,
                               .compile_args = reader_ct_args});

        SetRuntimeArgs(program, reader_kid, core,
                       {(uint32_t)src0_buf->address(), (uint32_t)src1_buf->address(), n_tiles});

        std::vector<uint32_t> compute_ct_args = {n_tiles, op_type};
        CreateKernel(program,
            kernel_path("compute/eltwise_binary.cpp"), core,
            ComputeConfig{.math_fidelity = MathFidelity::LoFi, .compile_args = compute_ct_args});

        std::vector<uint32_t> writer_ct_args = {(uint32_t)CBIndex::c_16};
        TensorAccessorArgs(*dst_buf).append_to(writer_ct_args);

        auto writer_kid = CreateKernel(program,
            kernel_path("dataflow/writer_tiles.cpp"), core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                               .noc = NOC::RISCV_0_default,
                               .compile_args = writer_ct_args});

        SetRuntimeArgs(program, writer_kid, core,
                       {(uint32_t)dst_buf->address(), n_tiles});

        cached.workload = MeshWorkload();
        cached.workload.add_program(dev_range, std::move(program));
        EnqueueMeshWorkload(cq, cached.workload, false);
        cached.valid = true;
    } else {
        EnqueueMeshWorkload(cq, cached.workload, false);
    }
}

// Cached GEMV workload: created once per unique (M,K,weight) combo, reused across calls.
struct CachedGemvWorkload {
    MeshWorkload workload;
    bool valid = false;
};
static std::map<std::tuple<uintptr_t,uint32_t,uint32_t,uint32_t>, CachedGemvWorkload> g_gemv_cache;

// DRAM-sharded GEMV: y[1,M] = x[1,K] @ W[M,K]^T
// Uses 12 DRAM-optimal cores (one per bank) for maximum bandwidth.
// Weight buffer is DRAM-sharded: each bank stores Mt_per_bank × Kt contiguous tiles.
// Each core reads activations from interleaved buf, weights from its assigned bank.
static void dispatch_gemv(MeshDevice* device,
                           std::shared_ptr<MeshBuffer> act_buf,
                           std::shared_ptr<MeshBuffer> weight_buf,
                           std::shared_ptr<MeshBuffer> out_buf,
                           uint32_t M, uint32_t K,
                           tt::DataFormat weight_format = tt::DataFormat::Bfp8_b) {
    auto& cq = device->mesh_command_queue();
    auto key = std::make_tuple((uintptr_t)device, (uint32_t)act_buf->address(),
                               (uint32_t)weight_buf->address(), (uint32_t)out_buf->address());
    auto& cached = g_gemv_cache[key];

    if (!cached.valid) {
        MeshCoordinateRange dev_range(device->shape());

        uint32_t Kt = K / TILE_WIDTH;
        uint32_t Mt = M / TILE_HEIGHT;
        uint32_t num_banks = get_num_dram_banks(device);
        const auto& dram_workers = get_dram_workers(device);
        uint32_t Mt_per_bank = (Mt + num_banks - 1) / num_banks;

        // Build CoreRangeSet from optimal DRAM workers (non-rectangular)
        std::vector<CoreRange> core_ranges;
        for (uint32_t b = 0; b < num_banks; b++) {
            auto& c = dram_workers[b];
            core_ranges.push_back(CoreRange(c, c));
        }
        CoreRangeSet all_cores(core_ranges);

        Program program = CreateProgram();

        uint32_t bf16_tile_bytes = TILE_HEIGHT * TILE_WIDTH * sizeof(bfloat16);
        uint32_t weight_tile_bytes = tile_size(weight_format);

        // BLOCK=Kt: single contiguous read per row, minimal overhead.
        // Compute is negligible vs DRAM reads; double-buffered for row overlap.
        uint32_t effective_block = Kt;
        uint32_t weight_cb_tiles = effective_block * 2;  // double-buffered for row overlap

        // Activation CB: Kt tiles (loaded once, reused for all output rows)
        CircularBufferConfig cb_act_cfg =
            CircularBufferConfig(Kt * bf16_tile_bytes, {{CBIndex::c_0, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_0, bf16_tile_bytes);
        CreateCircularBuffer(program, all_cores, cb_act_cfg);

        // Weight CB: sized for double-buffered block reads
        CircularBufferConfig cb_weight_cfg =
            CircularBufferConfig(weight_cb_tiles * weight_tile_bytes, {{CBIndex::c_1, weight_format}})
                .set_page_size(CBIndex::c_1, weight_tile_bytes);
        CreateCircularBuffer(program, all_cores, cb_weight_cfg);

        // Output CB: 2 tiles
        CircularBufferConfig cb_out_cfg =
            CircularBufferConfig(2 * bf16_tile_bytes, {{CBIndex::c_16, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_16, bf16_tile_bytes);
        CreateCircularBuffer(program, all_cores, cb_out_cfg);

        // DRAM-sharded reader: reads act from interleaved buf, weight from assigned bank
        std::vector<uint32_t> reader_ct_args = {(uint32_t)CBIndex::c_0, (uint32_t)CBIndex::c_1, Kt, effective_block};
        TensorAccessorArgs(*act_buf).append_to(reader_ct_args);

        auto reader_kid = CreateKernel(program,
            kernel_path("dataflow/reader_gemv_dram_sharded.cpp"), all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc = NOC::RISCV_1_default,
                               .compile_args = reader_ct_args});

        // Compute kernel: Kt and BLOCK are compile-time, Mt_per_core is runtime
        std::vector<uint32_t> compute_ct_args = {Kt, effective_block};
        auto compute_kid = CreateKernel(program,
            kernel_path("compute/gemv.cpp"), all_cores,
            ComputeConfig{.math_fidelity = MathFidelity::LoFi, .compile_args = compute_ct_args});

        // Writer kernel: writes output tiles to interleaved out_buf
        std::vector<uint32_t> writer_ct_args = {(uint32_t)CBIndex::c_16};
        TensorAccessorArgs(*out_buf).append_to(writer_ct_args);

        auto writer_kid = CreateKernel(program,
            kernel_path("dataflow/writer_gemv_multicore.cpp"), all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                               .noc = NOC::RISCV_0_default,
                               .compile_args = writer_ct_args});

        // Set per-core runtime args (one core per DRAM bank)
        for (uint32_t b = 0; b < num_banks; b++) {
            CoreCoord core = dram_workers[b];
            uint32_t start_row = b * Mt_per_bank;
            uint32_t mt_this_core = (start_row >= Mt) ? 0 :
                                    std::min(Mt_per_bank, Mt - start_row);

            // Reader: [act_addr, weight_bank_addr, Mt_per_core, bank_id, weight_start_offset]
            SetRuntimeArgs(program, reader_kid, core,
                           {(uint32_t)act_buf->address(), (uint32_t)weight_buf->address(),
                            mt_this_core, b, 0u});

            // Compute: [Mt_per_core]
            SetRuntimeArgs(program, compute_kid, core,
                           {mt_this_core});

            // Writer: [dst_addr, Mt_per_core, out_start_tile]
            SetRuntimeArgs(program, writer_kid, core,
                           {(uint32_t)out_buf->address(), mt_this_core, start_row});
        }

        cached.workload = MeshWorkload();
        cached.workload.add_program(dev_range, std::move(program));
        EnqueueMeshWorkload(cq, cached.workload, false);
        cached.valid = true;
    } else {
        EnqueueMeshWorkload(cq, cached.workload, false);
    }
}

// Cached GEMV+ResAdd workload: GEMV output added directly to residual buffer.
struct CachedGemvResaddWorkload {
    MeshWorkload workload;
    bool valid = false;
};
static std::map<std::tuple<uintptr_t,uint32_t,uint32_t,uint32_t,uint32_t>, CachedGemvResaddWorkload> g_gemv_resadd_cache;

// Fused GEMV + Residual Add: residual[1,M] += x[1,K] @ W[M,K]^T
// Same as dispatch_gemv but the writer reads existing residual, adds GEMV output, writes back.
// Eliminates separate dispatch_eltwise_binary call.
static void dispatch_gemv_resadd(MeshDevice* device,
                                  std::shared_ptr<MeshBuffer> act_buf,
                                  std::shared_ptr<MeshBuffer> weight_buf,
                                  std::shared_ptr<MeshBuffer> residual_buf,
                                  uint32_t M, uint32_t K,
                                  tt::DataFormat weight_format = tt::DataFormat::Bfp8_b) {
    auto& cq = device->mesh_command_queue();
    auto key = std::make_tuple((uintptr_t)device, (uint32_t)act_buf->address(),
                               (uint32_t)weight_buf->address(),
                               (uint32_t)residual_buf->address(), M);
    auto& cached = g_gemv_resadd_cache[key];

    if (!cached.valid) {
        MeshCoordinateRange dev_range(device->shape());

        uint32_t Kt = K / TILE_WIDTH;
        uint32_t Mt = M / TILE_HEIGHT;
        uint32_t num_banks = get_num_dram_banks(device);
        const auto& dram_workers = get_dram_workers(device);
        uint32_t Mt_per_bank = (Mt + num_banks - 1) / num_banks;

        std::vector<CoreRange> core_ranges;
        for (uint32_t b = 0; b < num_banks; b++) {
            auto& c = dram_workers[b];
            core_ranges.push_back(CoreRange(c, c));
        }
        CoreRangeSet all_cores(core_ranges);

        Program program = CreateProgram();

        uint32_t bf16_tile_bytes = TILE_HEIGHT * TILE_WIDTH * sizeof(bfloat16);
        uint32_t weight_tile_bytes = tile_size(weight_format);

        // BLOCK=Kt: single contiguous read per row, minimal overhead.
        // Compute is negligible vs DRAM reads; double-buffered for row overlap.
        uint32_t effective_block = Kt;
        uint32_t weight_cb_tiles = effective_block * 2;  // double-buffered for row overlap

        // Activation CB (c_0)
        CircularBufferConfig cb_act_cfg =
            CircularBufferConfig(Kt * bf16_tile_bytes, {{CBIndex::c_0, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_0, bf16_tile_bytes);
        CreateCircularBuffer(program, all_cores, cb_act_cfg);

        // Weight CB (c_1)
        CircularBufferConfig cb_weight_cfg =
            CircularBufferConfig(weight_cb_tiles * weight_tile_bytes, {{CBIndex::c_1, weight_format}})
                .set_page_size(CBIndex::c_1, weight_tile_bytes);
        CreateCircularBuffer(program, all_cores, cb_weight_cfg);

        // Output CB (c_16)
        CircularBufferConfig cb_out_cfg =
            CircularBufferConfig(2 * bf16_tile_bytes, {{CBIndex::c_16, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_16, bf16_tile_bytes);
        CreateCircularBuffer(program, all_cores, cb_out_cfg);

        // Scratch CB (c_2) for writer to read residual tiles
        CircularBufferConfig cb_scratch_cfg =
            CircularBufferConfig(bf16_tile_bytes, {{CBIndex::c_2, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_2, bf16_tile_bytes);
        CreateCircularBuffer(program, all_cores, cb_scratch_cfg);

        // DRAM-sharded reader: reads act from interleaved buf, weight from assigned bank
        std::vector<uint32_t> reader_ct_args = {(uint32_t)CBIndex::c_0, (uint32_t)CBIndex::c_1, Kt, effective_block};
        TensorAccessorArgs(*act_buf).append_to(reader_ct_args);

        auto reader_kid = CreateKernel(program,
            kernel_path("dataflow/reader_gemv_dram_sharded.cpp"), all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc = NOC::RISCV_1_default,
                               .compile_args = reader_ct_args});

        // Compute: same as regular GEMV
        std::vector<uint32_t> compute_ct_args = {Kt, effective_block};
        auto compute_kid = CreateKernel(program,
            kernel_path("compute/gemv.cpp"), all_cores,
            ComputeConfig{.math_fidelity = MathFidelity::LoFi, .compile_args = compute_ct_args});

        // Writer: fused resadd writer (reads residual, adds output, writes back)
        std::vector<uint32_t> writer_ct_args = {(uint32_t)CBIndex::c_16};
        TensorAccessorArgs(*residual_buf).append_to(writer_ct_args);

        auto writer_kid = CreateKernel(program,
            kernel_path("dataflow/writer_gemv_resadd.cpp"), all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                               .noc = NOC::RISCV_0_default,
                               .compile_args = writer_ct_args});

        for (uint32_t b = 0; b < num_banks; b++) {
            CoreCoord core = dram_workers[b];
            uint32_t start_row = b * Mt_per_bank;
            uint32_t mt_this_core = (start_row >= Mt) ? 0 :
                                    std::min(Mt_per_bank, Mt - start_row);

            // Reader: [act_addr, weight_bank_addr, Mt_per_core, bank_id, weight_start_offset]
            SetRuntimeArgs(program, reader_kid, core,
                           {(uint32_t)act_buf->address(), (uint32_t)weight_buf->address(),
                            mt_this_core, b, 0u});
            SetRuntimeArgs(program, compute_kid, core, {mt_this_core});
            SetRuntimeArgs(program, writer_kid, core,
                           {(uint32_t)residual_buf->address(), mt_this_core, start_row});
        }

        cached.workload = MeshWorkload();
        cached.workload.add_program(dev_range, std::move(program));
        EnqueueMeshWorkload(cq, cached.workload, false);
        cached.valid = true;
    } else {
        EnqueueMeshWorkload(cq, cached.workload, false);
    }
}

// Cached GEMV with split writer: output tiles below split_tile go to dst0, above go to dst1.
// Used for fused gate+up GEMV where a single GEMV populates two separate output buffers.
struct CachedGemvSplitWorkload {
    MeshWorkload workload;
    bool valid = false;
};
static std::map<std::tuple<uintptr_t,uint32_t,uint32_t,uint32_t,uint32_t>, CachedGemvSplitWorkload> g_gemv_split_cache;

// Fused gate+up GEMV: y[1,2M] = x[1,K] @ W[2M,K]^T, output split to dst0[M] and dst1[M]
static void dispatch_gemv_split(MeshDevice* device,
                                 std::shared_ptr<MeshBuffer> act_buf,
                                 std::shared_ptr<MeshBuffer> weight_buf,
                                 std::shared_ptr<MeshBuffer> dst0_buf,
                                 std::shared_ptr<MeshBuffer> dst1_buf,
                                 uint32_t M_total, uint32_t K, uint32_t split_M,
                                 tt::DataFormat weight_format = tt::DataFormat::Bfp8_b) {
    auto& cq = device->mesh_command_queue();
    auto key = std::make_tuple((uintptr_t)device, (uint32_t)act_buf->address(),
                               (uint32_t)weight_buf->address(),
                               (uint32_t)dst0_buf->address(), (uint32_t)dst1_buf->address());
    auto& cached = g_gemv_split_cache[key];

    if (!cached.valid) {
        MeshCoordinateRange dev_range(device->shape());

        uint32_t Kt = K / TILE_WIDTH;
        uint32_t Mt = M_total / TILE_HEIGHT;
        uint32_t split_tile = split_M / TILE_HEIGHT;
        uint32_t num_banks = get_num_dram_banks(device);
        const auto& dram_workers = get_dram_workers(device);
        uint32_t Mt_per_bank = (Mt + num_banks - 1) / num_banks;

        std::vector<CoreRange> core_ranges;
        for (uint32_t b = 0; b < num_banks; b++) {
            auto& c = dram_workers[b];
            core_ranges.push_back(CoreRange(c, c));
        }
        CoreRangeSet all_cores(core_ranges);

        Program program = CreateProgram();

        uint32_t bf16_tile_bytes = TILE_HEIGHT * TILE_WIDTH * sizeof(bfloat16);
        uint32_t weight_tile_bytes = tile_size(weight_format);

        // BLOCK=Kt: single contiguous read per row, minimal overhead.
        // Compute is negligible vs DRAM reads; double-buffered for row overlap.
        uint32_t effective_block = Kt;
        uint32_t weight_cb_tiles = effective_block * 2;  // double-buffered for row overlap

        // Activation CB (c_0)
        CircularBufferConfig cb_act_cfg =
            CircularBufferConfig(Kt * bf16_tile_bytes, {{CBIndex::c_0, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_0, bf16_tile_bytes);
        CreateCircularBuffer(program, all_cores, cb_act_cfg);

        // Weight CB (c_1)
        CircularBufferConfig cb_weight_cfg =
            CircularBufferConfig(weight_cb_tiles * weight_tile_bytes, {{CBIndex::c_1, weight_format}})
                .set_page_size(CBIndex::c_1, weight_tile_bytes);
        CreateCircularBuffer(program, all_cores, cb_weight_cfg);

        // Output CB (c_16)
        CircularBufferConfig cb_out_cfg =
            CircularBufferConfig(2 * bf16_tile_bytes, {{CBIndex::c_16, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_16, bf16_tile_bytes);
        CreateCircularBuffer(program, all_cores, cb_out_cfg);

        // DRAM-sharded reader: reads act from interleaved buf, weight from assigned bank
        std::vector<uint32_t> reader_ct_args = {(uint32_t)CBIndex::c_0, (uint32_t)CBIndex::c_1, Kt, effective_block};
        TensorAccessorArgs(*act_buf).append_to(reader_ct_args);

        auto reader_kid = CreateKernel(program,
            kernel_path("dataflow/reader_gemv_dram_sharded.cpp"), all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc = NOC::RISCV_1_default,
                               .compile_args = reader_ct_args});

        // Compute: same as regular GEMV
        std::vector<uint32_t> compute_ct_args = {Kt, effective_block};
        auto compute_kid = CreateKernel(program,
            kernel_path("compute/gemv.cpp"), all_cores,
            ComputeConfig{.math_fidelity = MathFidelity::LoFi, .compile_args = compute_ct_args});

        // Writer: split writer (gate/up to separate buffers)
        std::vector<uint32_t> writer_ct_args = {(uint32_t)CBIndex::c_16};
        TensorAccessorArgs(*dst0_buf).append_to(writer_ct_args);
        TensorAccessorArgs(*dst1_buf).append_to(writer_ct_args);

        auto writer_kid = CreateKernel(program,
            kernel_path("dataflow/writer_gemv_split.cpp"), all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                               .noc = NOC::RISCV_0_default,
                               .compile_args = writer_ct_args});

        for (uint32_t b = 0; b < num_banks; b++) {
            CoreCoord core = dram_workers[b];
            uint32_t start_row = b * Mt_per_bank;
            uint32_t mt_this_core = (start_row >= Mt) ? 0 :
                                    std::min(Mt_per_bank, Mt - start_row);

            // Reader: [act_addr, weight_bank_addr, Mt_per_core, bank_id, weight_start_offset]
            SetRuntimeArgs(program, reader_kid, core,
                           {(uint32_t)act_buf->address(), (uint32_t)weight_buf->address(),
                            mt_this_core, b, 0u});
            SetRuntimeArgs(program, compute_kid, core, {mt_this_core});
            SetRuntimeArgs(program, writer_kid, core,
                           {(uint32_t)dst0_buf->address(), (uint32_t)dst1_buf->address(),
                            mt_this_core, start_row, split_tile});
        }

        cached.workload = MeshWorkload();
        cached.workload.add_program(dev_range, std::move(program));
        EnqueueMeshWorkload(cq, cached.workload, false);
        cached.valid = true;
    } else {
        EnqueueMeshWorkload(cq, cached.workload, false);
    }
}

// Cached GEMV with fused RMSNorm: each reader core independently computes full rmsnorm
// before doing GEMV weight reads. Eliminates host PCIe round-trip for rmsnorm.
struct CachedFusedNormGemvWorkload {
    MeshWorkload workload;
    bool valid = false;
};
static std::map<std::tuple<uintptr_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t>, CachedFusedNormGemvWorkload> g_fused_norm_gemv_cache;

// GEMV with fused RMSNorm: y[1,M] = rmsnorm(hidden, norm_weight) @ W[M,K]^T
// hidden_buf: [1, K_padded] BF16 on device (will be rmsnorm'd by each reader core)
// norm_weight_buf: [1, K_padded] BF16 on device (rmsnorm weights)
// weight_buf: DRAM-sharded weight matrix
// out_buf: [1, M_padded] BF16 output
static void dispatch_gemv_fused_norm(MeshDevice* device,
                                      std::shared_ptr<MeshBuffer> hidden_buf,
                                      std::shared_ptr<MeshBuffer> norm_weight_buf,
                                      std::shared_ptr<MeshBuffer> weight_buf,
                                      std::shared_ptr<MeshBuffer> out_buf,
                                      uint32_t M, uint32_t K, uint32_t n_elements,
                                      tt::DataFormat weight_format = tt::DataFormat::Bfp8_b) {
    auto& cq = device->mesh_command_queue();
    auto key = std::make_tuple((uintptr_t)device, (uint32_t)hidden_buf->address(),
                               (uint32_t)norm_weight_buf->address(),
                               (uint32_t)weight_buf->address(),
                               (uint32_t)out_buf->address(), M);
    auto& cached = g_fused_norm_gemv_cache[key];

    if (!cached.valid) {
        MeshCoordinateRange dev_range(device->shape());

        uint32_t Kt = K / TILE_WIDTH;
        uint32_t Mt = M / TILE_HEIGHT;
        uint32_t num_banks = get_num_dram_banks(device);
        const auto& dram_workers = get_dram_workers(device);
        uint32_t Mt_per_bank = (Mt + num_banks - 1) / num_banks;

        std::vector<CoreRange> core_ranges;
        for (uint32_t b = 0; b < num_banks; b++) {
            auto& c = dram_workers[b];
            core_ranges.push_back(CoreRange(c, c));
        }
        CoreRangeSet all_cores(core_ranges);

        Program program = CreateProgram();

        uint32_t bf16_tile_bytes = TILE_HEIGHT * TILE_WIDTH * sizeof(bfloat16);
        uint32_t weight_tile_bytes = tile_size(weight_format);

        uint32_t effective_block = Kt;
        uint32_t weight_cb_tiles = effective_block;

        // Activation CB (c_0): Kt tiles for hidden + normalized activations
        CircularBufferConfig cb_act_cfg =
            CircularBufferConfig(Kt * bf16_tile_bytes, {{CBIndex::c_0, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_0, bf16_tile_bytes);
        CreateCircularBuffer(program, all_cores, cb_act_cfg);

        // Weight CB (c_1): double-buffered weight reads
        CircularBufferConfig cb_weight_cfg =
            CircularBufferConfig(weight_cb_tiles * weight_tile_bytes, {{CBIndex::c_1, weight_format}})
                .set_page_size(CBIndex::c_1, weight_tile_bytes);
        CreateCircularBuffer(program, all_cores, cb_weight_cfg);

        // Norm weight CB (c_2): Kt tiles for rmsnorm weights
        CircularBufferConfig cb_norm_cfg =
            CircularBufferConfig(Kt * bf16_tile_bytes, {{CBIndex::c_2, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_2, bf16_tile_bytes);
        CreateCircularBuffer(program, all_cores, cb_norm_cfg);

        // Output CB (c_16): 2 tiles
        CircularBufferConfig cb_out_cfg =
            CircularBufferConfig(2 * bf16_tile_bytes, {{CBIndex::c_16, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_16, bf16_tile_bytes);
        CreateCircularBuffer(program, all_cores, cb_out_cfg);

        // Reader: fused rmsnorm + GEMV reader
        std::vector<uint32_t> reader_ct_args = {
            (uint32_t)CBIndex::c_0, (uint32_t)CBIndex::c_1, (uint32_t)CBIndex::c_2,
            Kt, effective_block
        };
        TensorAccessorArgs(*hidden_buf).append_to(reader_ct_args);
        TensorAccessorArgs(*norm_weight_buf).append_to(reader_ct_args);
        TensorAccessorArgs(*weight_buf).append_to(reader_ct_args);

        auto reader_kid = CreateKernel(program,
            kernel_path("dataflow/reader_gemv_fused_norm.cpp"), all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc = NOC::RISCV_1_default,
                               .compile_args = reader_ct_args});

        // Compute: same as regular GEMV
        std::vector<uint32_t> compute_ct_args = {Kt, effective_block};
        auto compute_kid = CreateKernel(program,
            kernel_path("compute/gemv.cpp"), all_cores,
            ComputeConfig{.math_fidelity = MathFidelity::LoFi, .compile_args = compute_ct_args});

        // Writer: standard output writer
        std::vector<uint32_t> writer_ct_args = {(uint32_t)CBIndex::c_16};
        TensorAccessorArgs(*out_buf).append_to(writer_ct_args);

        auto writer_kid = CreateKernel(program,
            kernel_path("dataflow/writer_gemv_multicore.cpp"), all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                               .noc = NOC::RISCV_0_default,
                               .compile_args = writer_ct_args});

        for (uint32_t b = 0; b < num_banks; b++) {
            CoreCoord core = dram_workers[b];
            uint32_t start_row = b * Mt_per_bank;
            uint32_t mt_this_core = (start_row >= Mt) ? 0 :
                                    std::min(Mt_per_bank, Mt - start_row);

            // Reader: [hidden_addr, norm_weight_addr, weight_addr, Mt_per_core, n_elements, weight_start_tile]
            SetRuntimeArgs(program, reader_kid, core,
                           {(uint32_t)hidden_buf->address(), (uint32_t)norm_weight_buf->address(),
                            (uint32_t)weight_buf->address(), mt_this_core, n_elements, start_row * Kt});

            SetRuntimeArgs(program, compute_kid, core, {mt_this_core});

            SetRuntimeArgs(program, writer_kid, core,
                           {(uint32_t)out_buf->address(), mt_this_core, start_row});
        }

        cached.workload = MeshWorkload();
        cached.workload.add_program(dev_range, std::move(program));
        EnqueueMeshWorkload(cq, cached.workload, false);
        cached.valid = true;
    } else {
        EnqueueMeshWorkload(cq, cached.workload, false);
    }
}

// Cached fused-norm GEMV split: rmsnorm(hidden, norm_w) → gate/up GEMV with split output
struct CachedFusedNormGemvSplitWorkload {
    MeshWorkload workload;
    bool valid = false;
};
static std::map<std::tuple<uintptr_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t>, CachedFusedNormGemvSplitWorkload> g_fused_norm_gemv_split_cache;

static void dispatch_gemv_fused_norm_split(MeshDevice* device,
                                            std::shared_ptr<MeshBuffer> hidden_buf,
                                            std::shared_ptr<MeshBuffer> norm_weight_buf,
                                            std::shared_ptr<MeshBuffer> weight_buf,
                                            std::shared_ptr<MeshBuffer> dst0_buf,
                                            std::shared_ptr<MeshBuffer> dst1_buf,
                                            uint32_t M_total, uint32_t K, uint32_t split_M,
                                            uint32_t n_elements,
                                            tt::DataFormat weight_format = tt::DataFormat::Bfp8_b) {
    auto& cq = device->mesh_command_queue();
    auto key = std::make_tuple((uintptr_t)device, (uint32_t)hidden_buf->address(),
                               (uint32_t)norm_weight_buf->address(),
                               (uint32_t)weight_buf->address(),
                               (uint32_t)dst0_buf->address(), (uint32_t)dst1_buf->address());
    auto& cached = g_fused_norm_gemv_split_cache[key];

    if (!cached.valid) {
        MeshCoordinateRange dev_range(device->shape());

        uint32_t Kt = K / TILE_WIDTH;
        uint32_t Mt = M_total / TILE_HEIGHT;
        uint32_t split_tile = split_M / TILE_HEIGHT;
        uint32_t num_banks = get_num_dram_banks(device);
        const auto& dram_workers = get_dram_workers(device);
        uint32_t Mt_per_bank = (Mt + num_banks - 1) / num_banks;

        std::vector<CoreRange> core_ranges;
        for (uint32_t b = 0; b < num_banks; b++) {
            auto& c = dram_workers[b];
            core_ranges.push_back(CoreRange(c, c));
        }
        CoreRangeSet all_cores(core_ranges);

        Program program = CreateProgram();

        uint32_t bf16_tile_bytes = TILE_HEIGHT * TILE_WIDTH * sizeof(bfloat16);
        uint32_t weight_tile_bytes = tile_size(weight_format);

        uint32_t effective_block = Kt;
        uint32_t weight_cb_tiles = effective_block;

        // Activation CB (c_0): Kt tiles for hidden + normalized activations
        CircularBufferConfig cb_act_cfg =
            CircularBufferConfig(Kt * bf16_tile_bytes, {{CBIndex::c_0, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_0, bf16_tile_bytes);
        CreateCircularBuffer(program, all_cores, cb_act_cfg);

        // Weight CB (c_1): double-buffered
        CircularBufferConfig cb_weight_cfg =
            CircularBufferConfig(weight_cb_tiles * weight_tile_bytes, {{CBIndex::c_1, weight_format}})
                .set_page_size(CBIndex::c_1, weight_tile_bytes);
        CreateCircularBuffer(program, all_cores, cb_weight_cfg);

        // Norm weight CB (c_2): Kt tiles
        CircularBufferConfig cb_norm_cfg =
            CircularBufferConfig(Kt * bf16_tile_bytes, {{CBIndex::c_2, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_2, bf16_tile_bytes);
        CreateCircularBuffer(program, all_cores, cb_norm_cfg);

        // Output CB (c_16): 2 tiles
        CircularBufferConfig cb_out_cfg =
            CircularBufferConfig(2 * bf16_tile_bytes, {{CBIndex::c_16, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_16, bf16_tile_bytes);
        CreateCircularBuffer(program, all_cores, cb_out_cfg);

        // Reader: fused rmsnorm + GEMV reader
        std::vector<uint32_t> reader_ct_args = {
            (uint32_t)CBIndex::c_0, (uint32_t)CBIndex::c_1, (uint32_t)CBIndex::c_2,
            Kt, effective_block
        };
        TensorAccessorArgs(*hidden_buf).append_to(reader_ct_args);
        TensorAccessorArgs(*norm_weight_buf).append_to(reader_ct_args);
        TensorAccessorArgs(*weight_buf).append_to(reader_ct_args);

        auto reader_kid = CreateKernel(program,
            kernel_path("dataflow/reader_gemv_fused_norm.cpp"), all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc = NOC::RISCV_1_default,
                               .compile_args = reader_ct_args});

        // Compute: same GEMV kernel
        std::vector<uint32_t> compute_ct_args = {Kt, effective_block};
        auto compute_kid = CreateKernel(program,
            kernel_path("compute/gemv.cpp"), all_cores,
            ComputeConfig{.math_fidelity = MathFidelity::LoFi, .compile_args = compute_ct_args});

        // Writer: split writer (gate/up to separate buffers)
        std::vector<uint32_t> writer_ct_args = {(uint32_t)CBIndex::c_16};
        TensorAccessorArgs(*dst0_buf).append_to(writer_ct_args);
        TensorAccessorArgs(*dst1_buf).append_to(writer_ct_args);

        auto writer_kid = CreateKernel(program,
            kernel_path("dataflow/writer_gemv_split.cpp"), all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                               .noc = NOC::RISCV_0_default,
                               .compile_args = writer_ct_args});

        for (uint32_t b = 0; b < num_banks; b++) {
            CoreCoord core = dram_workers[b];
            uint32_t start_row = b * Mt_per_bank;
            uint32_t mt_this_core = (start_row >= Mt) ? 0 :
                                    std::min(Mt_per_bank, Mt - start_row);

            // Reader: [hidden_addr, norm_weight_addr, weight_addr, Mt_per_core, n_elements, weight_start_tile]
            SetRuntimeArgs(program, reader_kid, core,
                           {(uint32_t)hidden_buf->address(), (uint32_t)norm_weight_buf->address(),
                            (uint32_t)weight_buf->address(), mt_this_core, n_elements, start_row * Kt});

            SetRuntimeArgs(program, compute_kid, core, {mt_this_core});

            SetRuntimeArgs(program, writer_kid, core,
                           {(uint32_t)dst0_buf->address(), (uint32_t)dst1_buf->address(),
                            mt_this_core, start_row, split_tile});
        }

        cached.workload = MeshWorkload();
        cached.workload.add_program(dev_range, std::move(program));
        EnqueueMeshWorkload(cq, cached.workload, false);
        cached.valid = true;
    } else {
        EnqueueMeshWorkload(cq, cached.workload, false);
    }
}

// Cached SwiGLU workload: SiLU(gate) * up
struct CachedSwigluWorkload {
    MeshWorkload workload;
    bool valid = false;
};
static std::map<std::tuple<uintptr_t,uint32_t,uint32_t,uint32_t>, CachedSwigluWorkload> g_swiglu_cache;

// Dispatch SwiGLU (multi-core): out = SiLU(gate) * up
// Uses 12 DRAM worker cores, each handling a slice of tiles.
static void dispatch_swiglu(MeshDevice* device,
                             std::shared_ptr<MeshBuffer> gate_buf,
                             std::shared_ptr<MeshBuffer> up_buf,
                             std::shared_ptr<MeshBuffer> out_buf,
                             uint32_t n_tiles) {
    auto& cq = device->mesh_command_queue();
    auto key = std::make_tuple((uintptr_t)device,
                               (uint32_t)gate_buf->address(),
                               (uint32_t)up_buf->address(), (uint32_t)out_buf->address());
    auto& cached = g_swiglu_cache[key];

    if (!cached.valid) {
        MeshCoordinateRange dev_range(device->shape());
        Program program = CreateProgram();

        uint32_t num_banks = get_num_dram_banks(device);
        const auto& dram_workers = get_dram_workers(device);
        uint32_t tiles_per_bank = (n_tiles + num_banks - 1) / num_banks;

        std::vector<CoreRange> core_ranges;
        for (uint32_t b = 0; b < num_banks; b++) {
            auto& c = dram_workers[b];
            core_ranges.push_back(CoreRange(c, c));
        }
        CoreRangeSet all_cores(core_ranges);

        uint32_t tile_bytes = TILE_HEIGHT * TILE_WIDTH * sizeof(bfloat16);

        // gate CB (c_0), up CB (c_1), intermediate SiLU CB (c_2), output CB (c_16)
        CircularBufferConfig cb_gate_cfg =
            CircularBufferConfig(2 * tile_bytes, {{CBIndex::c_0, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_0, tile_bytes);
        CreateCircularBuffer(program, all_cores, cb_gate_cfg);

        CircularBufferConfig cb_up_cfg =
            CircularBufferConfig(2 * tile_bytes, {{CBIndex::c_1, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_1, tile_bytes);
        CreateCircularBuffer(program, all_cores, cb_up_cfg);

        CircularBufferConfig cb_silu_cfg =
            CircularBufferConfig(2 * tile_bytes, {{CBIndex::c_2, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_2, tile_bytes);
        CreateCircularBuffer(program, all_cores, cb_silu_cfg);

        CircularBufferConfig cb_out_cfg =
            CircularBufferConfig(2 * tile_bytes, {{CBIndex::c_16, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_16, tile_bytes);
        CreateCircularBuffer(program, all_cores, cb_out_cfg);

        // Reader: reads gate and up tiles with start offset
        std::vector<uint32_t> reader_ct_args = {(uint32_t)CBIndex::c_0, (uint32_t)CBIndex::c_1};
        TensorAccessorArgs(*gate_buf).append_to(reader_ct_args);
        TensorAccessorArgs(*up_buf).append_to(reader_ct_args);

        auto reader_kid = CreateKernel(program,
            kernel_path("dataflow/reader_binary_tiles_multicore.cpp"), all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc = NOC::RISCV_1_default,
                               .compile_args = reader_ct_args});

        // Compute: SwiGLU with runtime num_tiles
        auto compute_kid = CreateKernel(program,
            kernel_path("compute/swiglu_multicore.cpp"), all_cores,
            ComputeConfig{.math_fidelity = MathFidelity::LoFi});

        // Writer: writes output tiles with start offset
        std::vector<uint32_t> writer_ct_args = {(uint32_t)CBIndex::c_16};
        TensorAccessorArgs(*out_buf).append_to(writer_ct_args);

        auto writer_kid = CreateKernel(program,
            kernel_path("dataflow/writer_gemv_multicore.cpp"), all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                               .noc = NOC::RISCV_0_default,
                               .compile_args = writer_ct_args});

        for (uint32_t b = 0; b < num_banks; b++) {
            CoreCoord core = dram_workers[b];
            uint32_t start_tile = b * tiles_per_bank;
            uint32_t my_tiles = (start_tile >= n_tiles) ? 0 :
                                std::min(tiles_per_bank, n_tiles - start_tile);

            SetRuntimeArgs(program, reader_kid, core,
                           {(uint32_t)gate_buf->address(), (uint32_t)up_buf->address(),
                            my_tiles, start_tile});
            SetRuntimeArgs(program, compute_kid, core, {my_tiles});
            SetRuntimeArgs(program, writer_kid, core,
                           {(uint32_t)out_buf->address(), my_tiles, start_tile});
        }

        cached.workload = MeshWorkload();
        cached.workload.add_program(dev_range, std::move(program));
        EnqueueMeshWorkload(cq, cached.workload, false);
        cached.valid = true;
    } else {
        EnqueueMeshWorkload(cq, cached.workload, false);
    }
}

// Cached RMSNorm workload: created once, reused across calls (trace-compatible).
struct CachedRmsnormWorkload {
    MeshWorkload workload;
    bool valid = false;
};
static std::map<std::tuple<uintptr_t,uint32_t,uint32_t,uint32_t>, CachedRmsnormWorkload> g_rmsnorm_cache;

// Dispatch custom RMSNorm: out = rms_norm(input) * weight
// All buffers are [1, n_embd_padded] BF16 on device.
// MeshWorkload is cached and reused — trace-compatible after first warmup call.
static void dispatch_rmsnorm(MeshDevice* device,
                              std::shared_ptr<MeshBuffer> in_buf,
                              std::shared_ptr<MeshBuffer> weight_buf,
                              std::shared_ptr<MeshBuffer> out_buf,
                              uint32_t n_elements, uint32_t n_tiles) {
    auto& cq = device->mesh_command_queue();
    auto key = std::make_tuple((uintptr_t)device,
                               (uint32_t)in_buf->address(),
                               (uint32_t)weight_buf->address(), (uint32_t)out_buf->address());
    auto& cached = g_rmsnorm_cache[key];

    if (!cached.valid) {
        MeshCoordinateRange dev_range(device->shape());
        Program program = CreateProgram();
        tt::tt_metal::CoreCoord core = {0, 0};

        uint32_t tile_bytes = TILE_HEIGHT * TILE_WIDTH * sizeof(bfloat16);

        // CB c_0: input tiles (all n_tiles at once for batched read)
        CircularBufferConfig cb_in_cfg =
            CircularBufferConfig(n_tiles * tile_bytes, {{CBIndex::c_0, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_0, tile_bytes);
        CreateCircularBuffer(program, core, cb_in_cfg);

        // CB c_1: weight tiles (all n_tiles at once for batched read)
        CircularBufferConfig cb_w_cfg =
            CircularBufferConfig(n_tiles * tile_bytes, {{CBIndex::c_1, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_1, tile_bytes);
        CreateCircularBuffer(program, core, cb_w_cfg);

        std::vector<uint32_t> ct_args = {n_tiles};
        TensorAccessorArgs(*in_buf).append_to(ct_args);
        TensorAccessorArgs(*weight_buf).append_to(ct_args);
        TensorAccessorArgs(*out_buf).append_to(ct_args);

        auto kernel_id = CreateKernel(program,
            kernel_path("dataflow/reader_rmsnorm.cpp"), core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc = NOC::RISCV_1_default,
                               .compile_args = ct_args});

        SetRuntimeArgs(program, kernel_id, core,
                       {(uint32_t)in_buf->address(), (uint32_t)weight_buf->address(),
                        (uint32_t)out_buf->address(), n_elements});

        cached.workload = MeshWorkload();
        cached.workload.add_program(dev_range, std::move(program));
        EnqueueMeshWorkload(cq, cached.workload, false);
        cached.valid = true;
    } else {
        EnqueueMeshWorkload(cq, cached.workload, false);
    }
}

// Multi-core RMSNorm: 12 DRAM worker cores each handle their local bank's tiles.
// Each core computes partial sum_sq, cross-core reduction via semaphores + NOC writes.
// ~10x faster than single-core (0.2ms vs 1.9ms) since computation is distributed.
struct CachedMulticoreRmsnormWorkload {
    MeshWorkload workload;
    bool valid = false;
};
static std::map<std::tuple<uintptr_t,uint32_t,uint32_t,uint32_t>, CachedMulticoreRmsnormWorkload> g_mc_rmsnorm_cache;

static void dispatch_rmsnorm_multicore(MeshDevice* device,
                                        std::shared_ptr<MeshBuffer> in_buf,
                                        std::shared_ptr<MeshBuffer> weight_buf,
                                        std::shared_ptr<MeshBuffer> out_buf,
                                        uint32_t n_elements, uint32_t n_tiles) {
    auto& cq = device->mesh_command_queue();
    auto key = std::make_tuple((uintptr_t)device,
                               (uint32_t)in_buf->address(),
                               (uint32_t)weight_buf->address(), (uint32_t)out_buf->address());
    auto& cached = g_mc_rmsnorm_cache[key];

    if (!cached.valid) {
        MeshCoordinateRange dev_range(device->shape());
        Program program = CreateProgram();
        uint32_t num_banks = get_num_dram_banks(device);
        const auto& dram_workers = get_dram_workers(device);

        // Build CoreRangeSet from DRAM workers
        std::vector<CoreRange> core_ranges;
        for (uint32_t b = 0; b < num_banks; b++) {
            auto& c = dram_workers[b];
            core_ranges.push_back(CoreRange(c, c));
        }
        CoreRangeSet all_cores(core_ranges);

        uint32_t tile_bytes = TILE_HEIGHT * TILE_WIDTH * sizeof(bfloat16);

        // Max tiles per core: ceil(n_tiles / num_banks)
        uint32_t max_tiles_per_core = (n_tiles + num_banks - 1) / num_banks;

        // CB c_0: local input tiles
        CircularBufferConfig cb_in_cfg =
            CircularBufferConfig(max_tiles_per_core * tile_bytes,
                                 {{CBIndex::c_0, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_0, tile_bytes);
        CreateCircularBuffer(program, all_cores, cb_in_cfg);

        // CB c_1: local weight tiles (reused for output)
        CircularBufferConfig cb_w_cfg =
            CircularBufferConfig(max_tiles_per_core * tile_bytes,
                                 {{CBIndex::c_1, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_1, tile_bytes);
        CreateCircularBuffer(program, all_cores, cb_w_cfg);

        // Semaphores: sem0 on all cores (core 0 uses it for partial count),
        //             sem1 on all cores (non-zero cores wait on it)
        auto sem0_id = CreateSemaphore(program, all_cores, 0);
        auto sem1_id = CreateSemaphore(program, all_cores, 0);

        // Compile-time args: [n_tiles, num_cores, acc_in, acc_weight, acc_out]
        std::vector<uint32_t> ct_args = {n_tiles, num_banks};
        TensorAccessorArgs(*in_buf).append_to(ct_args);
        TensorAccessorArgs(*weight_buf).append_to(ct_args);
        TensorAccessorArgs(*out_buf).append_to(ct_args);

        auto kernel_id = CreateKernel(program,
            kernel_path("dataflow/reader_rmsnorm_multicore.cpp"), all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc = NOC::RISCV_1_default,
                               .compile_args = ct_args});

        // Get virtual NOC coordinates for all worker cores
        // Use the first device in the mesh to convert logical→virtual
        auto* dev0 = device->get_device(0, 0);
        std::vector<CoreCoord> noc_coords(num_banks);
        for (uint32_t b = 0; b < num_banks; b++) {
            noc_coords[b] = dev0->worker_core_from_logical_core(dram_workers[b]);
        }

        // CB c_2: scratch (1 tile = 1024 bytes, plenty for 12 floats + norm_factor)
        CircularBufferConfig cb_scratch_cfg =
            CircularBufferConfig(tile_bytes, {{CBIndex::c_2, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_2, tile_bytes);
        auto cb_scratch = CreateCircularBuffer(program, all_cores, cb_scratch_cfg);

        // Set per-core runtime args
        for (uint32_t b = 0; b < num_banks; b++) {
            CoreCoord core = dram_workers[b];

            // Common args: [in_addr, weight_addr, out_addr, n_elements, core_id,
            //               sem0_addr, sem1_addr, scratch_addr,
            //               noc_x_0, noc_y_0, ..., noc_x_N-1, noc_y_N-1]
            std::vector<uint32_t> rt_args = {
                (uint32_t)in_buf->address(),
                (uint32_t)weight_buf->address(),
                (uint32_t)out_buf->address(),
                n_elements,
                b,  // core_id
                sem0_id,
                sem1_id,
                0,  // scratch_addr placeholder — will be filled by kernel using CB
            };

            // All cores get all NOC coordinates (needed for cross-core communication)
            for (uint32_t i = 0; i < num_banks; i++) {
                rt_args.push_back(noc_coords[i].x);
                rt_args.push_back(noc_coords[i].y);
            }

            SetRuntimeArgs(program, kernel_id, core, rt_args);
        }

        cached.workload = MeshWorkload();
        cached.workload.add_program(dev_range, std::move(program));
        EnqueueMeshWorkload(cq, cached.workload, false);
        cached.valid = true;
    } else {
        EnqueueMeshWorkload(cq, cached.workload, false);
    }
}

// FPU-based single-core RMSNorm using Tensix compute engine.
// Much faster than software-float multicore version (~0.1ms vs ~1.9ms).
struct CachedFpuRmsnormWorkload {
    MeshWorkload workload;
    bool valid = false;
};
static std::map<std::tuple<uintptr_t,uint32_t,uint32_t,uint32_t>, CachedFpuRmsnormWorkload> g_fpu_rmsnorm_cache;

static void dispatch_rmsnorm_fpu(MeshDevice* device,
                                  std::shared_ptr<MeshBuffer> in_buf,
                                  std::shared_ptr<MeshBuffer> weight_buf,
                                  std::shared_ptr<MeshBuffer> out_buf,
                                  uint32_t n_elements, uint32_t n_tiles) {
    auto& cq = device->mesh_command_queue();
    auto key = std::make_tuple((uintptr_t)device,
                               (uint32_t)in_buf->address(),
                               (uint32_t)weight_buf->address(), (uint32_t)out_buf->address());
    auto& cached = g_fpu_rmsnorm_cache[key];

    if (!cached.valid) {
        MeshCoordinateRange dev_range(device->shape());
        Program program = CreateProgram();

        // Use a single worker core (first DRAM worker)
        auto core = get_dram_workers(device)[0];
        CoreRange single_core(core, core);
        CoreRangeSet core_set({single_core});

        uint32_t Kt = n_tiles;
        uint32_t bf16_tile_bytes = TILE_HEIGHT * TILE_WIDTH * sizeof(bfloat16);

        // cb_hidden (c_0): Kt tiles — input only
        CircularBufferConfig cb_hidden_cfg =
            CircularBufferConfig(Kt * bf16_tile_bytes, {{CBIndex::c_0, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_0, bf16_tile_bytes);
        CreateCircularBuffer(program, core_set, cb_hidden_cfg);

        // cb_out (c_16): Kt tiles — packer output CB
        CircularBufferConfig cb_out_cfg =
            CircularBufferConfig(Kt * bf16_tile_bytes, {{CBIndex::c_16, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_16, bf16_tile_bytes);
        CreateCircularBuffer(program, core_set, cb_out_cfg);

        // cb_norm_w (c_2): Kt tiles
        CircularBufferConfig cb_norm_cfg =
            CircularBufferConfig(Kt * bf16_tile_bytes, {{CBIndex::c_2, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_2, bf16_tile_bytes);
        CreateCircularBuffer(program, core_set, cb_norm_cfg);

        // cb_x2/act (c_24): Kt tiles shared
        CircularBufferConfig cb_x2_cfg =
            CircularBufferConfig(Kt * bf16_tile_bytes, {{CBIndex::c_24, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_24, bf16_tile_bytes);
        CreateCircularBuffer(program, core_set, cb_x2_cfg);

        // cb_var (c_4): 1 tile
        CircularBufferConfig cb_var_cfg =
            CircularBufferConfig(bf16_tile_bytes, {{CBIndex::c_4, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_4, bf16_tile_bytes);
        CreateCircularBuffer(program, core_set, cb_var_cfg);

        // cb_scaler (c_5): 1 tile
        CircularBufferConfig cb_scaler_cfg =
            CircularBufferConfig(bf16_tile_bytes, {{CBIndex::c_5, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_5, bf16_tile_bytes);
        CreateCircularBuffer(program, core_set, cb_scaler_cfg);

        // cb_eps (c_6): 1 tile
        CircularBufferConfig cb_eps_cfg =
            CircularBufferConfig(bf16_tile_bytes, {{CBIndex::c_6, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_6, bf16_tile_bytes);
        CreateCircularBuffer(program, core_set, cb_eps_cfg);

        // cb_rsqrt (c_7): 1 tile
        CircularBufferConfig cb_rsqrt_cfg =
            CircularBufferConfig(bf16_tile_bytes, {{CBIndex::c_7, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_7, bf16_tile_bytes);
        CreateCircularBuffer(program, core_set, cb_rsqrt_cfg);

        // Reader kernel
        std::vector<uint32_t> reader_ct_args = {Kt};
        TensorAccessorArgs(*in_buf).append_to(reader_ct_args);
        TensorAccessorArgs(*weight_buf).append_to(reader_ct_args);

        auto reader_kid = CreateKernel(program,
            kernel_path("dataflow/reader_rmsnorm_fpu.cpp"), core_set,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc = NOC::RISCV_1_default,
                               .compile_args = reader_ct_args});

        // Writer kernel
        std::vector<uint32_t> writer_ct_args = {Kt};
        TensorAccessorArgs(*out_buf).append_to(writer_ct_args);

        auto writer_kid = CreateKernel(program,
            kernel_path("dataflow/writer_rmsnorm_fpu.cpp"), core_set,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                               .noc = NOC::RISCV_0_default,
                               .compile_args = writer_ct_args});

        // Compute kernel
        std::vector<uint32_t> compute_ct_args = {Kt};
        auto compute_kid = CreateKernel(program,
            kernel_path("compute/rmsnorm_fpu.cpp"), core_set,
            ComputeConfig{.math_fidelity = MathFidelity::LoFi, .compile_args = compute_ct_args});

        // Scaler for REDUCE_SCALAR: applied twice (element-wise + post-multiply),
        // so use sqrt(1/N) to get effective 1/N scaling
        float scaler_f32 = 1.0f / std::sqrt((float)n_elements);
        uint32_t scaler_bits;
        std::memcpy(&scaler_bits, &scaler_f32, 4);
        uint32_t scaler_bf16 = scaler_bits >> 16;

        // Runtime args
        SetRuntimeArgs(program, reader_kid, core,
                       {(uint32_t)in_buf->address(), (uint32_t)weight_buf->address(),
                        n_elements, scaler_bf16});
        SetRuntimeArgs(program, writer_kid, core,
                       {(uint32_t)out_buf->address()});

        cached.workload = MeshWorkload();
        cached.workload.add_program(dev_range, std::move(program));
        EnqueueMeshWorkload(cq, cached.workload, false);
        cached.valid = true;
    } else {
        EnqueueMeshWorkload(cq, cached.workload, false);
    }
}

// ============================================================================
// Pre-allocated intermediate buffers for on-device FFN
// ============================================================================
struct FfnBuf {
    // Intermediates for gate/up split + SiLU + multiply
    std::shared_ptr<MeshBuffer> gate_buf;   // [1, n_ff] tiled
    std::shared_ptr<MeshBuffer> up_buf;     // [1, n_ff] tiled
    std::shared_ptr<MeshBuffer> act_buf;    // [1, n_ff] tiled (silu result / multiply result)
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
    buf.initialized = true;

    auto [ins, _] = g_ffn_bufs.emplace(device, std::move(buf));
    printf("  [ffn_buf] allocated n_ff=%u intermediates on device\n", MC::n_ff);
    return ins->second;
}

// ============================================================================
// Per-layer trace captures for on-device operations
// ============================================================================
// Trace for: rms_norm(hidden) → matmul(norm, weight) → result in gemv out buffer
static MeshTraceId g_norm_matmul_traces[32];
static bool g_norm_matmul_traces_valid[32] = {};

// Trace for: add_(hidden, residual) → rms_norm → FFN chain → add_(hidden, ffn_out)
static MeshTraceId g_ffn_chain_traces[32];
static bool g_ffn_chain_traces_valid[32] = {};
// Chip 1 FFN chain traces (for tensor-parallel FFN)
static MeshTraceId g_ffn_chain_traces_1[32];
static bool g_ffn_chain_traces_valid_1[32] = {};
// Chip 1 norm+matmul traces (for TP combined_proj/QKV)
static MeshTraceId g_norm_matmul_traces_1[32];
static bool g_norm_matmul_traces_valid_1[32] = {};

// Two-pass warmup: first forward pass warms all dispatch caches (no traces),
// second pass captures traces (no new allocations, avoids allocator warning).
static bool g_all_caches_warm = false;

// Run rmsnorm on device + GEMV on device (no host round-trip for norm).
// Hidden must already be on device in g_hidden_dev_buf.
static void norm_matmul_ops(std::shared_ptr<MeshBuffer> norm_weight_buf,
                            std::shared_ptr<MeshBuffer> weight_buf,
                            uint32_t M, uint32_t K) {
    constexpr uint32_t embd_tiles = MC::n_embd / TILE_WIDTH;
    dispatch_rmsnorm_fpu(g_mesh.get(), g_hidden_dev_buf, norm_weight_buf,
                         g_norm_dev_buf, MC::n_embd, embd_tiles);
    auto& gb = get_gemv_buf(g_mesh.get(), M, K);
    dispatch_gemv(g_mesh.get(), g_norm_dev_buf, weight_buf, gb.out_buf, M, K);
}

// Chip 1 version: rmsnorm + GEMV on chip 1's hidden/norm buffers
static void norm_matmul_ops_1(std::shared_ptr<MeshBuffer> norm_weight_buf,
                               std::shared_ptr<MeshBuffer> weight_buf,
                               uint32_t M, uint32_t K) {
    constexpr uint32_t embd_tiles = MC::n_embd / TILE_WIDTH;
    dispatch_rmsnorm_fpu(g_mesh1.get(), g_hidden_dev_buf_1, norm_weight_buf,
                         g_norm_dev_buf_1, MC::n_embd, embd_tiles);
    auto& gb = get_gemv_buf(g_mesh1.get(), M, K);
    dispatch_gemv(g_mesh1.get(), g_norm_dev_buf_1, weight_buf, gb.out_buf, M, K);
}

// Run output norm + LM head matmul on device (custom kernels, no host round-trip)
static std::shared_ptr<MeshBuffer> g_output_norm_buf;
static std::shared_ptr<MeshBuffer> g_lm_head_buf;
static void norm_lmhead_ops() {
    constexpr uint32_t embd_tiles = MC::n_embd / TILE_WIDTH;
    dispatch_rmsnorm_fpu(g_mesh.get(), g_hidden_dev_buf, g_output_norm_buf,
                         g_norm_dev_buf, MC::n_embd, embd_tiles);
    auto& gb = get_gemv_buf(g_mesh.get(), MC::n_vocab, MC::n_embd);
    dispatch_gemv(g_mesh.get(), g_norm_dev_buf, g_lm_head_buf, gb.out_buf, MC::n_vocab, MC::n_embd);
}
static MeshTraceId g_lmhead_trace;
static MeshTraceId g_lmhead_trace_1;  // chip 1 LM head trace
static bool g_lmhead_trace_valid = false;

// Run outproj matmul + residual add + norm + FFN chain + residual add on device.
// All using custom kernels (dispatch_gemv, dispatch_rmsnorm, dispatch_swiglu, dispatch_eltwise).
static void outproj_ffn_chain_ops(std::shared_ptr<MeshBuffer> outproj_weight_buf,
                                   uint32_t outproj_M, uint32_t outproj_K,
                                   std::shared_ptr<MeshBuffer> norm_weight_buf,
                                   std::shared_ptr<MeshBuffer> gate_up_weight_buf,
                                   std::shared_ptr<MeshBuffer> down_weight_buf) {
    // 1. Output projection + residual add (fused): hidden += residual @ outproj_weight^T
    dispatch_gemv_resadd(g_mesh.get(), g_residual_dev_buf, outproj_weight_buf,
                         g_hidden_dev_buf, outproj_M, outproj_K);

    // 2. RMSNorm (FPU-based)
    constexpr uint32_t embd_tiles = MC::n_embd / TILE_WIDTH;
    dispatch_rmsnorm_fpu(g_mesh.get(), g_hidden_dev_buf, norm_weight_buf,
                         g_norm_dev_buf, MC::n_embd, embd_tiles);

    // 3. FFN: fused gate+up GEMV → SwiGLU → down matmul + residual add
    auto& fb = get_ffn_buf(g_mesh.get());
    dispatch_gemv_split(g_mesh.get(), g_norm_dev_buf, gate_up_weight_buf,
                        fb.gate_buf, fb.up_buf, MC::n_ff * 2, MC::n_embd, MC::n_ff);

    // SwiGLU: SiLU(gate) * up
    constexpr uint32_t ff_tiles = MC::n_ff / TILE_WIDTH;
    dispatch_swiglu(g_mesh.get(), fb.gate_buf, fb.up_buf, fb.act_buf, ff_tiles);

    // Down projection + residual add (fused): hidden += act @ down_weight^T
    dispatch_gemv_resadd(g_mesh.get(), fb.act_buf, down_weight_buf,
                         g_hidden_dev_buf, MC::n_embd, MC::n_ff);
}

// TP FFN chain on chip 0: outproj_resadd → rmsnorm → fused gate+up → swiglu → down_half_resadd
static void outproj_ffn_chain_ops_tp0(
    std::shared_ptr<MeshBuffer> outproj_weight_buf,
    uint32_t outproj_M, uint32_t outproj_K,
    std::shared_ptr<MeshBuffer> norm_weight_buf,
    std::shared_ptr<MeshBuffer> gate_up_weight_buf,
    std::shared_ptr<MeshBuffer> down_weight_buf) {

    dispatch_gemv_resadd(g_mesh.get(), g_residual_dev_buf, outproj_weight_buf,
                         g_hidden_dev_buf, outproj_M, outproj_K);
    constexpr uint32_t embd_tiles = MC::n_embd / TILE_WIDTH;
    dispatch_rmsnorm_fpu(g_mesh.get(), g_hidden_dev_buf, norm_weight_buf,
                         g_norm_dev_buf, MC::n_embd, embd_tiles);
    dispatch_gemv_split(g_mesh.get(), g_norm_dev_buf, gate_up_weight_buf,
                        g_tp_ffn_0.gate_buf, g_tp_ffn_0.up_buf,
                        n_ff_tp * 2, MC::n_embd, n_ff_tp);
    dispatch_swiglu(g_mesh.get(), g_tp_ffn_0.gate_buf, g_tp_ffn_0.up_buf,
                    g_tp_ffn_0.act_buf, n_ff_tp_tiles);
    dispatch_gemv_resadd(g_mesh.get(), g_tp_ffn_0.act_buf, down_weight_buf,
                         g_hidden_dev_buf, MC::n_embd, n_ff_tp);
}

// TP FFN chain on chip 1: outproj_resadd → rmsnorm → fused gate+up → swiglu → down_half → partial
static void outproj_ffn_chain_ops_tp1(
    std::shared_ptr<MeshBuffer> outproj_weight_buf,
    uint32_t outproj_M, uint32_t outproj_K,
    std::shared_ptr<MeshBuffer> norm_weight_buf,
    std::shared_ptr<MeshBuffer> gate_up_weight_buf,
    std::shared_ptr<MeshBuffer> down_weight_buf) {

    dispatch_gemv_resadd(g_mesh1.get(), g_residual_dev_buf_1, outproj_weight_buf,
                         g_hidden_dev_buf_1, outproj_M, outproj_K);
    constexpr uint32_t embd_tiles = MC::n_embd / TILE_WIDTH;
    dispatch_rmsnorm_fpu(g_mesh1.get(), g_hidden_dev_buf_1, norm_weight_buf,
                         g_norm_dev_buf_1, MC::n_embd, embd_tiles);
    dispatch_gemv_split(g_mesh1.get(), g_norm_dev_buf_1, gate_up_weight_buf,
                        g_tp_ffn_1.gate_buf, g_tp_ffn_1.up_buf,
                        n_ff_tp * 2, MC::n_embd, n_ff_tp);
    dispatch_swiglu(g_mesh1.get(), g_tp_ffn_1.gate_buf, g_tp_ffn_1.up_buf,
                    g_tp_ffn_1.act_buf, n_ff_tp_tiles);
    // Plain GEMV (not resadd) - output partial sum to dedicated buffer
    dispatch_gemv(g_mesh1.get(), g_tp_ffn_1.act_buf, down_weight_buf,
                  g_partial_down_buf_1, MC::n_embd, n_ff_tp);
}

// Separate tiled host buffer for write_f32_to_buf (declared before use)
static std::vector<bfloat16> g_write_host_tiled;

// Write f32 vector to a device MeshBuffer (tilize + enqueue)
static void write_f32_to_buf(std::shared_ptr<MeshBuffer> buf, const float* data,
                              uint32_t len, MeshDevice* device = nullptr) {
    auto& cq = (device ? device : g_mesh.get())->mesh_command_queue();
    uint32_t padded = ((len + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;
    size_t needed = (size_t)(padded / TILE_WIDTH) * TILE_HEIGHT * TILE_WIDTH;
    if (g_write_host_tiled.size() < needed)
        g_write_host_tiled.resize(needed, bfloat16(0.0f));

    uint16_t* ht = reinterpret_cast<uint16_t*>(g_write_host_tiled.data());
    uint32_t num_tile_cols = padded / TILE_WIDTH;
    const uint32_t* bits = reinterpret_cast<const uint32_t*>(data);
    for (uint32_t tc = 0; tc < num_tile_cols; tc++) {
        uint32_t tile_off = tc * TILE_HEIGHT * TILE_WIDTH;
        uint32_t base = tc * TILE_WIDTH;
        if (base + 32 <= len) {
            // AVX-512: fused f32→bf16 + tilize (load 16 f32, truncate to bf16, store to tile face)
            __m512i v0 = _mm512_loadu_si512(bits + base);
            __m512i v1 = _mm512_loadu_si512(bits + base + 16);
            __m256i bf0 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(v0, 16));
            __m256i bf1 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(v1, 16));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(ht + tile_off), bf0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(ht + tile_off + 256), bf1);
        } else {
            // Scalar fallback
            uint16_t* scratch = g_bf16_scratch.data();
            for (uint32_t i = base; i < len; i++)
                scratch[i] = static_cast<uint16_t>(bits[i] >> 16);
            uint32_t n0 = std::min(16u, len - base);
            uint32_t n1 = (base + 16 < len) ? std::min(16u, len - base - 16) : 0;
            memcpy(ht + tile_off, scratch + base, n0 * sizeof(uint16_t));
            if (n1) memcpy(ht + tile_off + 256, scratch + base + 16, n1 * sizeof(uint16_t));
        }
    }
    EnqueueWriteMeshBuffer(cq, buf, g_write_host_tiled, false);
}

// ============================================================================
// Background chip 1 writer — overlaps remote writes with chip 0 GEMV execution
// ============================================================================
static std::vector<bfloat16> g_write_host_tiled_bg;   // separate staging buffer for background writes
static std::vector<uint16_t> g_bf16_scratch_bg;        // separate scratch for background tilize

// Persistent worker thread for chip 1 writes
struct Chip1Writer {
    std::thread thread;
    std::mutex mtx;
    std::condition_variable cv_work;
    std::condition_variable cv_done;
    std::function<void()> task;
    bool has_work = false;
    bool done = true;
    bool shutdown = false;

    void start() {
        thread = std::thread([this]{
            std::unique_lock<std::mutex> lk(mtx);
            while (true) {
                cv_work.wait(lk, [this]{ return has_work || shutdown; });
                if (shutdown) break;
                auto fn = std::move(task);
                lk.unlock();
                fn();
                lk.lock();
                has_work = false;
                done = true;
                cv_done.notify_one();
            }
        });
    }

    void submit(std::function<void()> fn) {
        std::unique_lock<std::mutex> lk(mtx);
        cv_done.wait(lk, [this]{ return done; });  // wait for previous task to finish
        task = std::move(fn);
        done = false;
        has_work = true;
        cv_work.notify_one();
    }

    void wait() {
        std::unique_lock<std::mutex> lk(mtx);
        cv_done.wait(lk, [this]{ return done; });
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(mtx);
            shutdown = true;
        }
        cv_work.notify_one();
        if (thread.joinable()) thread.join();
    }
};

static Chip1Writer g_chip1_writer;

// Persistent thread pool for parallel-for work (deltanet v-head processing)
// Uses C++20 atomic wait/notify (futex on Linux) — no mutex needed on hot path.
struct WorkerPool {
    static constexpr int N_WORKERS = 3;  // + main thread = 4 total
    std::thread threads[N_WORKERS];
    alignas(64) std::atomic<int> phase{0};       // incremented to signal new work
    alignas(64) std::atomic<int> done_count{0};  // workers increment when done
    std::atomic<bool> running{true};
    std::function<void(int, int)> work_fn;
    int ranges[N_WORKERS][2];

    void start() {
        for (int w = 0; w < N_WORKERS; w++) {
            threads[w] = std::thread([this, w]{
                int last_phase = 0;
                while (true) {
                    phase.wait(last_phase, std::memory_order_acquire);
                    if (!running.load(std::memory_order_relaxed)) return;
                    last_phase = phase.load(std::memory_order_relaxed);
                    work_fn(ranges[w][0], ranges[w][1]);
                    if (done_count.fetch_add(1, std::memory_order_acq_rel) == N_WORKERS - 1) {
                        done_count.notify_one();
                    }
                }
            });
        }
    }

    // Execute fn(start, end) in parallel across N_WORKERS+1 threads
    void parallel_for(int total, std::function<void(int, int)> fn) {
        int chunk = total / (N_WORKERS + 1);
        int remainder = total % (N_WORKERS + 1);
        int offset = 0;
        for (int w = 0; w < N_WORKERS; w++) {
            int sz = chunk + (w < remainder ? 1 : 0);
            ranges[w][0] = offset;
            ranges[w][1] = offset + sz;
            offset += sz;
        }
        work_fn = std::move(fn);
        done_count.store(0, std::memory_order_release);
        phase.fetch_add(1, std::memory_order_release);
        phase.notify_all();

        // Main thread does its chunk
        work_fn(offset, total);

        // Wait for all workers (futex-based, no mutex)
        while (done_count.load(std::memory_order_acquire) < N_WORKERS) {
            done_count.wait(done_count.load(std::memory_order_relaxed), std::memory_order_acquire);
        }
    }

    void stop() {
        running.store(false, std::memory_order_release);
        phase.fetch_add(1, std::memory_order_release);
        phase.notify_all();
        for (int w = 0; w < N_WORKERS; w++) {
            if (threads[w].joinable()) threads[w].join();
        }
    }
};

static WorkerPool g_worker_pool;

// Tilize + write to chip 1 device buffer using background staging buffer
static void write_f32_to_chip1_bg(std::shared_ptr<MeshBuffer> buf, const float* data, uint32_t len) {
    auto& cq = g_mesh1->mesh_command_queue();
    uint32_t padded = ((len + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;
    size_t needed = (size_t)(padded / TILE_WIDTH) * TILE_HEIGHT * TILE_WIDTH;
    if (g_write_host_tiled_bg.size() < needed)
        g_write_host_tiled_bg.resize(needed, bfloat16(0.0f));

    uint16_t* ht = reinterpret_cast<uint16_t*>(g_write_host_tiled_bg.data());
    uint32_t num_tile_cols = padded / TILE_WIDTH;
    const uint32_t* bits = reinterpret_cast<const uint32_t*>(data);
    for (uint32_t tc = 0; tc < num_tile_cols; tc++) {
        uint32_t tile_off = tc * TILE_HEIGHT * TILE_WIDTH;
        uint32_t base = tc * TILE_WIDTH;
        if (base + 32 <= len) {
            __m512i v0 = _mm512_loadu_si512(bits + base);
            __m512i v1 = _mm512_loadu_si512(bits + base + 16);
            __m256i bf0 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(v0, 16));
            __m256i bf1 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(v1, 16));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(ht + tile_off), bf0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(ht + tile_off + 256), bf1);
        } else {
            if (g_bf16_scratch_bg.size() < (size_t)len)
                g_bf16_scratch_bg.resize(len);
            uint16_t* scratch = g_bf16_scratch_bg.data();
            for (uint32_t i = base; i < len; i++)
                scratch[i] = static_cast<uint16_t>(bits[i] >> 16);
            uint32_t n0 = std::min(16u, len - base);
            uint32_t n1 = (base + 16 < len) ? std::min(16u, len - base - 16) : 0;
            memcpy(ht + tile_off, scratch + base, n0 * sizeof(uint16_t));
            if (n1) memcpy(ht + tile_off + 256, scratch + base + 16, n1 * sizeof(uint16_t));
        }
    }
    EnqueueWriteMeshBuffer(cq, buf, g_write_host_tiled_bg, false);
}

// Read gemv output buffer to host f32 (fused untilize + bf16→f32 via AVX-512)
static void read_gemv_to_f32(GemvBuf& gb, float* out, uint32_t M) {
    const uint16_t* oht = reinterpret_cast<const uint16_t*>(gb.out_host_tiled.data());
    uint32_t out_tile_cols = (M + TILE_WIDTH - 1) / TILE_WIDTH;  // use actual M, not bank-padded M_padded
    uint32_t* ybits = reinterpret_cast<uint32_t*>(out);
    for (uint32_t tc = 0; tc < out_tile_cols; tc++) {
        uint32_t tile_off = tc * TILE_HEIGHT * TILE_WIDTH;
        uint32_t base = tc * TILE_WIDTH;
        if (base + 32 <= M) {
            // AVX-512: load 16 bf16 from face 0, zero-extend to 32-bit, shift left 16
            __m256i f0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(oht + tile_off));
            __m256i f2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(oht + tile_off + 256));
            _mm512_storeu_si512(ybits + base, _mm512_slli_epi32(_mm512_cvtepu16_epi32(f0), 16));
            _mm512_storeu_si512(ybits + base + 16, _mm512_slli_epi32(_mm512_cvtepu16_epi32(f2), 16));
        } else {
            // Scalar fallback for last partial tile
            for (uint32_t i = 0; i < 16 && base + i < M; i++)
                ybits[base + i] = static_cast<uint32_t>(oht[tile_off + i]) << 16;
            for (uint32_t i = 0; i < 16 && base + 16 + i < M; i++)
                ybits[base + 16 + i] = static_cast<uint32_t>(oht[tile_off + 256 + i]) << 16;
        }
    }
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

// Pack bf16 data as BFP8_B tiles (CPU-side, can be called from worker threads).
// Input: row-major [M, K] bf16 matrix. Must be tile-aligned (M%32==0, K%32==0).
// The pack_as_bfp_tiles function expects pre-tilized input: each contiguous 1024-element
// block = one 32×32 tile. We reorganize from flat row-major to tile-major order first.
static std::vector<uint32_t> pack_bf16_as_bfp8b(const uint16_t* bf16_data, uint32_t M, uint32_t K) {
    constexpr uint32_t TH = TILE_HEIGHT, TW = TILE_WIDTH;
    uint32_t Mt = M / TH, Kt = K / TW;
    uint32_t total_tiles = Mt * Kt;
    uint32_t tile_elems = TH * TW;  // 1024
    const bfloat16* src = reinterpret_cast<const bfloat16*>(bf16_data);

    // Rearrange: [M,K] row-major → tile-major (each tile is 32×32 row-major block)
    std::vector<bfloat16> tiled(total_tiles * tile_elems);
    for (uint32_t mt = 0; mt < Mt; mt++) {
        for (uint32_t kt = 0; kt < Kt; kt++) {
            bfloat16* dst = tiled.data() + (mt * Kt + kt) * tile_elems;
            for (uint32_t r = 0; r < TH; r++) {
                const bfloat16* row_src = src + (mt * TH + r) * K + kt * TW;
                memcpy(dst + r * TW, row_src, TW * sizeof(bfloat16));
            }
        }
    }

    ttsl::Span<const bfloat16> span(tiled.data(), tiled.size());
    return pack_as_bfp_tiles<tt::DataFormat::Bfp8_b>(span, /*row_major_input=*/true, /*is_exp_a=*/false);
}

// Split packed BFP8_B tiles by M dimension (row split).
// Layout: tiles are row-major [Mt × Kt], so first Mt0*Kt tiles go to half0.
// Returns {half0, half1} and sets Mt0_out, Mt1_out to the tile counts.
static std::pair<std::vector<uint32_t>, std::vector<uint32_t>>
split_packed_m(const std::vector<uint32_t>& packed, uint32_t Mt, uint32_t Kt,
               uint32_t& Mt0_out, uint32_t& Mt1_out) {
    constexpr uint32_t wpt = BFLOAT8_B_TILE_HW / sizeof(uint32_t);  // 272 words per tile
    uint32_t Mt0 = (Mt + 1) / 2;  // ceil
    uint32_t Mt1 = Mt - Mt0;
    Mt0_out = Mt0;
    Mt1_out = Mt1;
    uint32_t words0 = Mt0 * Kt * wpt;
    uint32_t words1 = Mt1 * Kt * wpt;
    return {
        std::vector<uint32_t>(packed.begin(), packed.begin() + words0),
        std::vector<uint32_t>(packed.begin() + words0, packed.begin() + words0 + words1)
    };
}

// Split packed BFP8_B tiles by K dimension (column split): first K/2 cols vs second K/2 cols
// Tiles in row [m]: tile(m,0)..tile(m,Kt-1). Split at Kt/2.
static std::pair<std::vector<uint32_t>, std::vector<uint32_t>>
split_packed_k(const std::vector<uint32_t>& packed, uint32_t Mt, uint32_t Kt) {
    constexpr uint32_t wpt = BFLOAT8_B_TILE_HW / sizeof(uint32_t);
    uint32_t Kt_half = Kt / 2;
    uint32_t half_tiles_per_row = Kt_half;
    std::vector<uint32_t> half0(Mt * Kt_half * wpt), half1(Mt * Kt_half * wpt);
    for (uint32_t m = 0; m < Mt; m++) {
        uint32_t src_off = m * Kt * wpt;
        uint32_t dst_off = m * Kt_half * wpt;
        memcpy(half0.data() + dst_off, packed.data() + src_off, Kt_half * wpt * sizeof(uint32_t));
        memcpy(half1.data() + dst_off, packed.data() + src_off + Kt_half * wpt, Kt_half * wpt * sizeof(uint32_t));
    }
    return {std::move(half0), std::move(half1)};
}

// Upload packed BFP8_B data to device as DRAM-sharded buffer.
// Each of the 12 DRAM banks stores Mt_per_bank * Kt contiguous tiles,
// enabling bank-local reads with TRID pipelining for maximum bandwidth.
static std::shared_ptr<MeshBuffer> upload_packed_bfp8b_buf(MeshDevice* device,
                                                            const std::vector<uint32_t>& packed,
                                                            uint32_t M, uint32_t K) {
    constexpr uint32_t TH = TILE_HEIGHT, TW = TILE_WIDTH;
    uint32_t Mt = M / TH, Kt = K / TW;
    constexpr uint32_t bfp8_tile_bytes = BFLOAT8_B_TILE_HW;  // 1088

    // Create DRAM-sharded buffer: each bank stores Mt_per_bank * Kt contiguous tiles
    uint32_t num_banks = device->num_dram_channels();
    uint32_t Mt_per_bank = (Mt + num_banks - 1) / num_banks;
    uint32_t Mt_padded = Mt_per_bank * num_banks;
    uint32_t total_pages = Mt_padded * Kt;
    uint32_t pages_per_bank = Mt_per_bank * Kt;
    uint32_t total_bytes = total_pages * bfp8_tile_bytes;

    // Pad packed data with zero tiles if Mt not divisible by num_banks
    constexpr uint32_t words_per_tile = bfp8_tile_bytes / sizeof(uint32_t);  // 272
    uint32_t needed_words = total_pages * words_per_tile;
    std::vector<uint32_t> padded_packed;
    if (packed.size() < needed_words) {
        padded_packed = packed;
        padded_packed.resize(needed_words, 0);
    }
    const auto& write_data = (packed.size() >= needed_words) ? packed : padded_packed;

    // DRAM bank coordinates: 1D range (x=bank_id, y=0) — as per tt-metal convention
    CoreRange dram_bank_range({0, 0}, {num_banks - 1, 0});

    BufferDistributionSpec shard_spec(
        Shape{1, total_pages},
        Shape{1, pages_per_bank},
        corerange_to_cores(CoreRangeSet(dram_bank_range)));

    DeviceLocalBufferConfig dram_cfg{
        .page_size = bfp8_tile_bytes,
        .buffer_type = BufferType::DRAM,
        .sharding_args = BufferShardingArgs(shard_spec)};

    Shape2D global_shape(1, total_pages);
    distributed::ShardedBufferConfig sharded_cfg{
        .global_size = total_bytes,
        .global_buffer_shape = global_shape,
        .shard_shape = global_shape,
        .shard_orientation = ShardOrientation::ROW_MAJOR,
    };

    auto buf = MeshBuffer::create(sharded_cfg, dram_cfg, device);
    auto& cq = device->mesh_command_queue();
    EnqueueWriteMeshBuffer(cq, buf, write_data, false);
    return buf;
}

// Copy a BF16 norm weight MeshBuffer from chip 0 to chip 1
static std::shared_ptr<MeshBuffer> copy_norm_buf_to_chip1(std::shared_ptr<MeshBuffer> src) {
    uint32_t embd_tiles = MC::n_embd / TILE_WIDTH;
    uint32_t tile_bytes = TILE_HEIGHT * TILE_WIDTH * sizeof(bfloat16);
    size_t n_elems = embd_tiles * TILE_HEIGHT * TILE_WIDTH;
    std::vector<bfloat16> host_data(n_elems, bfloat16(0.0f));
    auto& cq0 = g_mesh->mesh_command_queue();
    EnqueueReadMeshBuffer(cq0, host_data, src, true);
    DeviceLocalBufferConfig dram_cfg{.page_size = tile_bytes, .buffer_type = BufferType::DRAM};
    auto dst = MeshBuffer::create(ReplicatedBufferConfig{.size = embd_tiles * tile_bytes},
                                   dram_cfg, g_mesh1.get());
    auto& cq1 = g_mesh1->mesh_command_queue();
    EnqueueWriteMeshBuffer(cq1, dst, host_data, false);
    return dst;
}

// ============================================================================
// BFP8_B weight cache — skip CPU packing on subsequent runs
// ============================================================================

struct BFP8CacheHeader {
    char magic[4];       // "BFP8"
    uint32_t version;    // 1
    uint64_t gguf_size;  // for invalidation
    uint64_t gguf_mtime; // for invalidation
    uint32_t n_entries;  // 161 = 32 layers * 5 + 1 LM head
};

static bool check_bfp8_cache(const char* model_path, const std::string& cache_path) {
    struct stat model_st;
    if (stat(model_path, &model_st) != 0) return false;

    FILE* cf = fopen(cache_path.c_str(), "rb");
    if (!cf) return false;

    BFP8CacheHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, cf) != 1) { fclose(cf); return false; }
    fclose(cf);

    if (memcmp(hdr.magic, "BFP8", 4) != 0 || hdr.version != 1) return false;
    if (hdr.gguf_size != (uint64_t)model_st.st_size) return false;
    if (hdr.gguf_mtime != (uint64_t)model_st.st_mtime) return false;
    return true;
}

// Helper: write one packed buffer entry to cache file
static void cache_write_entry(FILE* cf, uint32_t M, uint32_t K,
                               const std::vector<uint32_t>& packed) {
    uint64_t data_words = packed.size();
    fwrite(&M, 4, 1, cf);
    fwrite(&K, 4, 1, cf);
    fwrite(&data_words, 8, 1, cf);
    fwrite(packed.data(), sizeof(uint32_t), data_words, cf);
}

// Helper: read one packed buffer entry from cache file
static std::vector<uint32_t> cache_read_entry(FILE* cf, uint32_t& M_out, uint32_t& K_out) {
    fread(&M_out, 4, 1, cf);
    fread(&K_out, 4, 1, cf);
    uint64_t data_words;
    fread(&data_words, 8, 1, cf);
    std::vector<uint32_t> packed(data_words);
    fread(packed.data(), sizeof(uint32_t), data_words, cf);
    return packed;
}

// Upload packed data with TP splitting (shared by both cache-write and cache-read paths)
static void upload_ssm_layer_packed(int ssm_idx,
    std::vector<uint32_t>& p_combined, uint32_t combined_rows,
    std::vector<uint32_t>& p_gate, std::vector<uint32_t>& p_up,
    std::vector<uint32_t>& p_down, std::vector<uint32_t>& p_out) {

    g_wt.ssm_w_combined_buf[ssm_idx] = upload_packed_bfp8b_buf(g_mesh.get(), p_combined, combined_rows, MC::n_embd);

    if (g_mesh1) {
        uint32_t Mt_ff = MC::n_ff / TILE_HEIGHT, Kt_embd = MC::n_embd / TILE_WIDTH;
        uint32_t Mt0_gate, Mt1_gate, Mt0_up, Mt1_up;
        auto [gate_h0, gate_h1] = split_packed_m(p_gate, Mt_ff, Kt_embd, Mt0_gate, Mt1_gate);
        auto [up_h0, up_h1] = split_packed_m(p_up, Mt_ff, Kt_embd, Mt0_up, Mt1_up);
        uint32_t Mt_embd = MC::n_embd / TILE_HEIGHT, Kt_ff = MC::n_ff / TILE_WIDTH;
        auto [down_h0, down_h1] = split_packed_k(p_down, Mt_embd, Kt_ff);
        g_wt.ssm_ffn_down_buf[ssm_idx] = upload_packed_bfp8b_buf(g_mesh.get(), down_h0, MC::n_embd, n_ff_tp);
        g_wt.ssm_ffn_down_buf_1[ssm_idx] = upload_packed_bfp8b_buf(g_mesh1.get(), down_h1, MC::n_embd, n_ff_tp);
        std::vector<uint32_t> gate_up_h0(gate_h0);
        gate_up_h0.insert(gate_up_h0.end(), up_h0.begin(), up_h0.end());
        g_wt.ssm_ffn_gate_up_buf[ssm_idx] = upload_packed_bfp8b_buf(g_mesh.get(), gate_up_h0,
            (Mt0_gate + Mt0_up) * TILE_HEIGHT, MC::n_embd);
        std::vector<uint32_t> gate_up_h1(gate_h1);
        gate_up_h1.insert(gate_up_h1.end(), up_h1.begin(), up_h1.end());
        g_wt.ssm_ffn_gate_up_buf_1[ssm_idx] = upload_packed_bfp8b_buf(g_mesh1.get(), gate_up_h1,
            (Mt1_gate + Mt1_up) * TILE_HEIGHT, MC::n_embd);
        g_wt.ssm_out_buf[ssm_idx] = upload_packed_bfp8b_buf(g_mesh.get(), p_out, MC::n_embd, MC::ssm_d_inner);
        g_wt.ssm_out_buf_1[ssm_idx] = upload_packed_bfp8b_buf(g_mesh1.get(), p_out, MC::n_embd, MC::ssm_d_inner);
    } else {
        std::vector<uint32_t> gate_up(p_gate);
        gate_up.insert(gate_up.end(), p_up.begin(), p_up.end());
        g_wt.ssm_ffn_gate_up_buf[ssm_idx] = upload_packed_bfp8b_buf(g_mesh.get(), gate_up, MC::n_ff * 2, MC::n_embd);
        g_wt.ssm_ffn_down_buf[ssm_idx] = upload_packed_bfp8b_buf(g_mesh.get(), p_down, MC::n_embd, MC::n_ff);
        g_wt.ssm_out_buf[ssm_idx] = upload_packed_bfp8b_buf(g_mesh.get(), p_out, MC::n_embd, MC::ssm_d_inner);
    }
}

static void upload_attn_layer_packed(int attn_idx,
    std::vector<uint32_t>& p_qkv, uint32_t qkv_rows,
    std::vector<uint32_t>& p_gate, std::vector<uint32_t>& p_up,
    std::vector<uint32_t>& p_down, std::vector<uint32_t>& p_wo) {

    g_wt.attn_wqkv_buf[attn_idx] = upload_packed_bfp8b_buf(g_mesh.get(), p_qkv, qkv_rows, MC::n_embd);

    if (g_mesh1) {
        uint32_t Mt_ff = MC::n_ff / TILE_HEIGHT, Kt_embd = MC::n_embd / TILE_WIDTH;
        uint32_t Mt0_gate, Mt1_gate, Mt0_up, Mt1_up;
        auto [gate_h0, gate_h1] = split_packed_m(p_gate, Mt_ff, Kt_embd, Mt0_gate, Mt1_gate);
        auto [up_h0, up_h1] = split_packed_m(p_up, Mt_ff, Kt_embd, Mt0_up, Mt1_up);
        uint32_t Mt_embd = MC::n_embd / TILE_HEIGHT, Kt_ff = MC::n_ff / TILE_WIDTH;
        auto [down_h0, down_h1] = split_packed_k(p_down, Mt_embd, Kt_ff);
        g_wt.attn_ffn_down_buf[attn_idx] = upload_packed_bfp8b_buf(g_mesh.get(), down_h0, MC::n_embd, n_ff_tp);
        g_wt.attn_ffn_down_buf_1[attn_idx] = upload_packed_bfp8b_buf(g_mesh1.get(), down_h1, MC::n_embd, n_ff_tp);
        std::vector<uint32_t> gate_up_h0(gate_h0);
        gate_up_h0.insert(gate_up_h0.end(), up_h0.begin(), up_h0.end());
        g_wt.attn_ffn_gate_up_buf[attn_idx] = upload_packed_bfp8b_buf(g_mesh.get(), gate_up_h0,
            (Mt0_gate + Mt0_up) * TILE_HEIGHT, MC::n_embd);
        std::vector<uint32_t> gate_up_h1(gate_h1);
        gate_up_h1.insert(gate_up_h1.end(), up_h1.begin(), up_h1.end());
        g_wt.attn_ffn_gate_up_buf_1[attn_idx] = upload_packed_bfp8b_buf(g_mesh1.get(), gate_up_h1,
            (Mt1_gate + Mt1_up) * TILE_HEIGHT, MC::n_embd);
        g_wt.attn_wo_buf[attn_idx] = upload_packed_bfp8b_buf(g_mesh.get(), p_wo, MC::n_embd, MC::n_head * MC::head_dim);
        g_wt.attn_wo_buf_1[attn_idx] = upload_packed_bfp8b_buf(g_mesh1.get(), p_wo, MC::n_embd, MC::n_head * MC::head_dim);
    } else {
        std::vector<uint32_t> gate_up(p_gate);
        gate_up.insert(gate_up.end(), p_up.begin(), p_up.end());
        g_wt.attn_ffn_gate_up_buf[attn_idx] = upload_packed_bfp8b_buf(g_mesh.get(), gate_up, MC::n_ff * 2, MC::n_embd);
        g_wt.attn_ffn_down_buf[attn_idx] = upload_packed_bfp8b_buf(g_mesh.get(), p_down, MC::n_embd, MC::n_ff);
        g_wt.attn_wo_buf[attn_idx] = upload_packed_bfp8b_buf(g_mesh.get(), p_wo, MC::n_embd, MC::n_head * MC::head_dim);
    }
}

// Load all packed weights from BFP8_B cache and upload to device.
static void create_weight_tensors_from_cache(const std::string& cache_path) {
    printf("Loading pre-packed BFP8_B weights from cache...\n");
    auto t0 = std::chrono::steady_clock::now();

    FILE* cf = fopen(cache_path.c_str(), "rb");
    BFP8CacheHeader hdr;
    fread(&hdr, sizeof(hdr), 1, cf);

    int attn_idx = 0, ssm_idx = 0;
    for (int layer = 0; layer < MC::n_layers; layer++) {
        uint32_t M0, K0, M1, K1, M2, K2, M3, K3, M4, K4;
        auto p0 = cache_read_entry(cf, M0, K0);
        auto p1 = cache_read_entry(cf, M1, K1);
        auto p2 = cache_read_entry(cf, M2, K2);
        auto p3 = cache_read_entry(cf, M3, K3);
        auto p4 = cache_read_entry(cf, M4, K4);

        if (MC::is_recurrent(layer)) {
            upload_ssm_layer_packed(ssm_idx, p0, M0, p1, p2, p3, p4);
            if ((ssm_idx + 1) % 6 == 0) printf("  SSM layers 0-%d uploaded\n", ssm_idx);
            ssm_idx++;
        } else {
            upload_attn_layer_packed(attn_idx, p0, M0, p1, p2, p3, p4);
            printf("  Attn layer %d uploaded\n", attn_idx);
            attn_idx++;
        }
    }

    // LM head
    uint32_t lm_M, lm_K;
    auto p_lm = cache_read_entry(cf, lm_M, lm_K);
    fclose(cf);

    if (g_mesh1) {
        uint32_t Mt_lm = lm_M / TILE_HEIGHT;
        uint32_t Kt_lm = lm_K / TILE_WIDTH;
        uint32_t Mt0_lm, Mt1_lm;
        auto [lm_half0, lm_half1] = split_packed_m(p_lm, Mt_lm, Kt_lm, Mt0_lm, Mt1_lm);
        g_wt.lm_head_Mt0 = Mt0_lm;
        g_wt.lm_head_Mt1 = Mt1_lm;
        g_wt.lm_head_buf = upload_packed_bfp8b_buf(g_mesh.get(), lm_half0, Mt0_lm * TILE_HEIGHT, MC::n_embd);
        g_wt.lm_head_buf_1 = upload_packed_bfp8b_buf(g_mesh1.get(), lm_half1, Mt1_lm * TILE_HEIGHT, MC::n_embd);
        g_lm_head_buf = g_wt.lm_head_buf;
        printf("  lm_head split: chip0 %u rows, chip1 %u rows\n",
               Mt0_lm * TILE_HEIGHT, Mt1_lm * TILE_HEIGHT);
    } else {
        g_wt.lm_head_buf = upload_packed_bfp8b_buf(g_mesh.get(), p_lm, lm_M, MC::n_embd);
        g_lm_head_buf = g_wt.lm_head_buf;
        printf("  lm_head uploaded (single chip)\n");
    }

    auto t1 = std::chrono::steady_clock::now();
    printf("Loaded %d attn + %d SSM weights from cache (%.1fs).\n",
           attn_idx, ssm_idx, std::chrono::duration<double>(t1 - t0).count());
}

// Create weight buffers — pack weights as BFP8_B (multi-threaded) then upload.
// Each weight's host bf16 is freed immediately after packing to minimize peak memory.
// If cache_path is non-empty, writes packed data to cache for next run.
static void create_weight_tensors(const std::string& cache_path = "") {
    printf("Packing and uploading weights as BFLOAT8_B (multi-threaded)...\n");
    auto t0 = std::chrono::steady_clock::now();

    // Open cache file for writing if path provided
    FILE* cf = nullptr;
    if (!cache_path.empty()) {
        std::string tmp_path = cache_path + ".tmp";
        cf = fopen(tmp_path.c_str(), "wb");
        if (cf) {
            // Write placeholder header — will be finalized at end
            BFP8CacheHeader hdr{};
            fwrite(&hdr, sizeof(hdr), 1, cf);
            printf("  Writing BFP8_B cache to %s\n", cache_path.c_str());
        }
    }

    int attn_idx = 0, ssm_idx = 0;
    for (int layer = 0; layer < MC::n_layers; layer++) {
        if (MC::is_recurrent(layer)) {
            auto& lw = g_model.ssm_layers[ssm_idx];
            uint32_t combined_rows = MC::ssm_conv_channels + MC::ssm_d_inner
                                   + MC::ssm_dt_rank + MC::ssm_dt_rank;

            // Pack all 5 SSM weights in parallel
            std::vector<uint32_t> p_combined, p_gate, p_up, p_down, p_out;
            {
                std::thread t1([&]{ p_combined = pack_bf16_as_bfp8b(lw.w_combined_host.data(), combined_rows, MC::n_embd); });
                std::thread t2([&]{ p_gate = pack_bf16_as_bfp8b(lw.ffn_gate_host.data(), MC::n_ff, MC::n_embd); });
                std::thread t3([&]{ p_up = pack_bf16_as_bfp8b(lw.ffn_up_host.data(), MC::n_ff, MC::n_embd); });
                std::thread t4([&]{ p_down = pack_bf16_as_bfp8b(lw.ffn_down_host.data(), MC::n_embd, MC::n_ff); });
                p_out = pack_bf16_as_bfp8b(lw.ssm_out_host.data(), MC::n_embd, MC::ssm_d_inner);
                t1.join(); t2.join(); t3.join(); t4.join();
            }

            // Free host bf16 data
            { std::vector<uint16_t>().swap(lw.w_combined_host); }
            { std::vector<uint16_t>().swap(lw.ffn_gate_host); }
            { std::vector<uint16_t>().swap(lw.ffn_up_host); }
            { std::vector<uint16_t>().swap(lw.ffn_down_host); }
            { std::vector<uint16_t>().swap(lw.ssm_out_host); }

            // Write to cache before upload
            if (cf) {
                cache_write_entry(cf, combined_rows, MC::n_embd, p_combined);
                cache_write_entry(cf, MC::n_ff, MC::n_embd, p_gate);
                cache_write_entry(cf, MC::n_ff, MC::n_embd, p_up);
                cache_write_entry(cf, MC::n_embd, MC::n_ff, p_down);
                cache_write_entry(cf, MC::n_embd, MC::ssm_d_inner, p_out);
            }

            // TP split + upload
            upload_ssm_layer_packed(ssm_idx, p_combined, combined_rows, p_gate, p_up, p_down, p_out);

            if ((ssm_idx + 1) % 6 == 0) printf("  SSM layers 0-%d uploaded\n", ssm_idx);
            ssm_idx++;
        } else {
            auto& lw = g_model.attn_layers[attn_idx];
            int q_dim = MC::n_head * MC::head_dim * 2;
            int kv_dim_one = MC::n_head_kv * MC::head_dim;
            int qkv_rows = q_dim + 2 * kv_dim_one;

            // Pack all 5 attention weights in parallel
            std::vector<uint32_t> p_qkv, p_gate, p_up, p_down, p_wo;
            {
                std::thread t1([&]{ p_qkv = pack_bf16_as_bfp8b(lw.wqkv_host.data(), qkv_rows, MC::n_embd); });
                std::thread t2([&]{ p_gate = pack_bf16_as_bfp8b(lw.ffn_gate_host.data(), MC::n_ff, MC::n_embd); });
                std::thread t3([&]{ p_up = pack_bf16_as_bfp8b(lw.ffn_up_host.data(), MC::n_ff, MC::n_embd); });
                std::thread t4([&]{ p_down = pack_bf16_as_bfp8b(lw.ffn_down_host.data(), MC::n_embd, MC::n_ff); });
                p_wo = pack_bf16_as_bfp8b(lw.wo_host.data(), MC::n_embd, MC::n_head * MC::head_dim);
                t1.join(); t2.join(); t3.join(); t4.join();
            }

            // Free host bf16 data
            { std::vector<uint16_t>().swap(lw.wqkv_host); }
            { std::vector<uint16_t>().swap(lw.ffn_gate_host); }
            { std::vector<uint16_t>().swap(lw.ffn_up_host); }
            { std::vector<uint16_t>().swap(lw.ffn_down_host); }
            { std::vector<uint16_t>().swap(lw.wo_host); }

            // Write to cache before upload
            if (cf) {
                cache_write_entry(cf, qkv_rows, MC::n_embd, p_qkv);
                cache_write_entry(cf, MC::n_ff, MC::n_embd, p_gate);
                cache_write_entry(cf, MC::n_ff, MC::n_embd, p_up);
                cache_write_entry(cf, MC::n_embd, MC::n_ff, p_down);
                cache_write_entry(cf, MC::n_embd, MC::n_head * MC::head_dim, p_wo);
            }

            // TP split + upload
            upload_attn_layer_packed(attn_idx, p_qkv, qkv_rows, p_gate, p_up, p_down, p_wo);

            printf("  Attn layer %d uploaded\n", attn_idx);
            attn_idx++;
        }
    }

    auto t1_time = std::chrono::steady_clock::now();
    double layer_sec = std::chrono::duration<double>(t1_time - t0).count();
    printf("Uploaded %d attention + %d SSM weight tensors as BFLOAT8_B (%.1fs).\n",
           attn_idx, ssm_idx, layer_sec);

    // LM head — split across 2 chips for column-parallel if available
    auto p_lm = pack_bf16_as_bfp8b(g_model.output_host.data(), MC::n_vocab, MC::n_embd);
    { std::vector<uint16_t>().swap(g_model.output_host); }

    if (cf) {
        cache_write_entry(cf, MC::n_vocab, MC::n_embd, p_lm);
    }

    if (g_mesh1) {
        uint32_t Mt_lm = MC::n_vocab / TILE_HEIGHT;
        uint32_t Kt_lm = MC::n_embd / TILE_WIDTH;
        uint32_t Mt0_lm, Mt1_lm;
        auto [lm_half0, lm_half1] = split_packed_m(p_lm, Mt_lm, Kt_lm, Mt0_lm, Mt1_lm);
        g_wt.lm_head_Mt0 = Mt0_lm;
        g_wt.lm_head_Mt1 = Mt1_lm;
        g_wt.lm_head_buf = upload_packed_bfp8b_buf(g_mesh.get(), lm_half0, Mt0_lm * TILE_HEIGHT, MC::n_embd);
        g_wt.lm_head_buf_1 = upload_packed_bfp8b_buf(g_mesh1.get(), lm_half1, Mt1_lm * TILE_HEIGHT, MC::n_embd);
        g_lm_head_buf = g_wt.lm_head_buf;
        printf("  lm_head split: chip0 %u rows, chip1 %u rows\n",
               Mt0_lm * TILE_HEIGHT, Mt1_lm * TILE_HEIGHT);
    } else {
        g_wt.lm_head_buf = upload_packed_bfp8b_buf(g_mesh.get(), p_lm, MC::n_vocab, MC::n_embd);
        g_lm_head_buf = g_wt.lm_head_buf;
        printf("  lm_head uploaded (single chip)\n");
    }

    // Finalize cache file
    if (cf) {
        fflush(cf);
        fclose(cf);
        std::string tmp_path = cache_path + ".tmp";
        // Now rewrite header with valid data
        cf = fopen(tmp_path.c_str(), "r+b");
        if (cf) {
            // Get GGUF file stats — cache_path is model_path + ".bfp8cache"
            // model_path = cache_path minus ".bfp8cache" suffix
            std::string model_path = cache_path.substr(0, cache_path.size() - 10);
            struct stat st;
            stat(model_path.c_str(), &st);
            BFP8CacheHeader hdr;
            memcpy(hdr.magic, "BFP8", 4);
            hdr.version = 1;
            hdr.gguf_size = st.st_size;
            hdr.gguf_mtime = st.st_mtime;
            hdr.n_entries = 32 * 5 + 1;  // 161
            fseek(cf, 0, SEEK_SET);
            fwrite(&hdr, sizeof(hdr), 1, cf);
            fclose(cf);
            rename(tmp_path.c_str(), cache_path.c_str());
            printf("BFP8_B cache written (%.1f GB).\n",
                   (double)st.st_size / (1024.0 * 1024.0 * 1024.0));
            // Print actual cache size
            struct stat cache_st;
            if (stat(cache_path.c_str(), &cache_st) == 0) {
                printf("  Cache file: %.1f GB\n", (double)cache_st.st_size / (1024.0 * 1024.0 * 1024.0));
            }
        }
    }

}

// Set up norm buffers, persistent device buffers, and pre-allocate GEMV buffers.
// Called after weight upload (from either pack or cache path).
static void setup_post_weight_buffers() {
    // Assign norm weight buffers for on-device RMSNorm
    printf("Setting up norm weight buffers...\n");
    int attn_idx = 0, ssm_idx = 0;
    for (int layer = 0; layer < MC::n_layers; layer++) {
        if (MC::is_recurrent(layer)) {
            auto& lw = g_model.ssm_layers[ssm_idx];
            g_wt.attn_norm_buf[layer] = lw.attn_norm;
            g_wt.post_norm_buf[layer] = lw.post_attn_norm;
            ssm_idx++;
        } else {
            auto& lw = g_model.attn_layers[attn_idx];
            g_wt.attn_norm_buf[layer] = lw.attn_norm;
            g_wt.post_norm_buf[layer] = lw.post_attn_norm;
            attn_idx++;
        }
    }
    g_wt.output_norm_buf = g_model.output_norm;
    g_output_norm_buf = g_model.output_norm;

    // Copy post_norm and attn_norm weights to chip 1 for TP FFN chain + TP combined_proj
    if (g_mesh1) {
        printf("Copying norm weights to chip 1 for TP...\n");
        for (int layer = 0; layer < MC::n_layers; layer++) {
            g_wt.post_norm_buf_1[layer] = copy_norm_buf_to_chip1(g_wt.post_norm_buf[layer]);
            g_wt.attn_norm_buf_1[layer] = copy_norm_buf_to_chip1(g_wt.attn_norm_buf[layer]);
        }
    }

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

    // Pre-allocate all GEMV and FFN intermediate buffers (required before trace capture)
    printf("Pre-allocating GEMV and FFN buffers...\n");
    constexpr int combined_rows = MC::ssm_conv_channels + MC::ssm_d_inner
                                + MC::ssm_dt_rank + MC::ssm_dt_rank;
    constexpr int qkv_rows = MC::n_head * MC::head_dim * 2 + 2 * MC::n_head_kv * MC::head_dim;
    get_gemv_buf(g_mesh.get(), combined_rows, MC::n_embd);  // SSM combined
    get_gemv_buf(g_mesh.get(), qkv_rows, MC::n_embd);       // Attention QKV
    get_gemv_buf(g_mesh.get(), MC::n_embd, MC::ssm_d_inner); // SSM outproj
    get_gemv_buf(g_mesh.get(), MC::n_embd, MC::n_embd);      // Attn outproj (same as ssm outproj if sizes match)
    get_gemv_buf(g_mesh.get(), MC::n_embd, MC::n_head * MC::head_dim); // Attn outproj
    if (g_mesh1) {
        // TP: half-size FFN GEMV buffers
        get_gemv_buf(g_mesh.get(), n_ff_tp, MC::n_embd);         // Gate/up half (chip 0)
        get_gemv_buf(g_mesh.get(), MC::n_embd, n_ff_tp);         // Down half (chip 0)
        get_gemv_buf(g_mesh1.get(), n_ff_tp, MC::n_embd);        // Gate/up half (chip 1)
        get_gemv_buf(g_mesh1.get(), MC::n_embd, n_ff_tp);        // Down half (chip 1)
        // LM head halves on each chip
        get_gemv_buf(g_mesh.get(), g_wt.lm_head_Mt0 * TILE_HEIGHT, MC::n_embd);
        get_gemv_buf(g_mesh1.get(), g_wt.lm_head_Mt1 * TILE_HEIGHT, MC::n_embd);
        // Chip 1 persistent device buffers
        g_hidden_dev_buf_1 = MeshBuffer::create(ReplicatedBufferConfig{.size = embd_tiles * tile_bytes},
                                                 dram_cfg, g_mesh1.get());
        g_residual_dev_buf_1 = MeshBuffer::create(ReplicatedBufferConfig{.size = embd_tiles * tile_bytes},
                                                   dram_cfg, g_mesh1.get());
        g_norm_dev_buf_1 = MeshBuffer::create(ReplicatedBufferConfig{.size = embd_tiles * tile_bytes},
                                               dram_cfg, g_mesh1.get());
        g_partial_down_buf_1 = MeshBuffer::create(ReplicatedBufferConfig{.size = embd_tiles * tile_bytes},
                                                   dram_cfg, g_mesh1.get());
        // TP FFN intermediate buffers (half-size n_ff/2)
        uint32_t ff_tp_tiles = n_ff_tp / TILE_WIDTH;
        auto alloc_tp_ffn = [&](TpFfnBuf& buf, MeshDevice* dev) {
            buf.gate_buf = MeshBuffer::create(ReplicatedBufferConfig{.size = ff_tp_tiles * tile_bytes}, dram_cfg, dev);
            buf.up_buf = MeshBuffer::create(ReplicatedBufferConfig{.size = ff_tp_tiles * tile_bytes}, dram_cfg, dev);
            buf.act_buf = MeshBuffer::create(ReplicatedBufferConfig{.size = ff_tp_tiles * tile_bytes}, dram_cfg, dev);
            buf.initialized = true;
        };
        alloc_tp_ffn(g_tp_ffn_0, g_mesh.get());
        alloc_tp_ffn(g_tp_ffn_1, g_mesh1.get());
        printf("  TP FFN buffers allocated (n_ff_tp=%u per chip)\n", n_ff_tp);
    } else {
        get_gemv_buf(g_mesh.get(), MC::n_ff, MC::n_embd);        // Gate/up projections
        get_gemv_buf(g_mesh.get(), MC::n_embd, MC::n_ff);        // Down projection
        get_gemv_buf(g_mesh.get(), MC::n_vocab, MC::n_embd);     // LM head (single chip)
    }
    if (!g_mesh1) get_ffn_buf(g_mesh.get());              // FFN intermediates (non-TP only)
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

    // Fused f32→bf16 + tilize via AVX-512
    uint16_t* ht = reinterpret_cast<uint16_t*>(g_dev_host_tiled.data());
    const uint32_t* bits = reinterpret_cast<const uint32_t*>(f32_data);
    uint32_t num_tile_cols = n_embd_padded / TILE_WIDTH;
    for (uint32_t tc = 0; tc < num_tile_cols; tc++) {
        uint32_t tile_off = tc * TILE_HEIGHT * TILE_WIDTH;
        uint32_t base = tc * TILE_WIDTH;
        __m512i v0 = _mm512_loadu_si512(bits + base);
        __m512i v1 = _mm512_loadu_si512(bits + base + 16);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ht + tile_off),
                            _mm512_cvtepi32_epi16(_mm512_srli_epi32(v0, 16)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ht + tile_off + 256),
                            _mm512_cvtepi32_epi16(_mm512_srli_epi32(v1, 16)));
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

// Read a device buffer from chip 1 to host f32 (uses separate host buffer to avoid contention)
static void read_device_to_f32_chip1(std::shared_ptr<MeshBuffer> buf, float* out, uint32_t len,
                                      MeshCommandQueue& cq) {
    uint32_t padded = ((len + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;
    size_t needed = padded / TILE_WIDTH * TILE_HEIGHT * TILE_WIDTH;
    if (g_dev_host_tiled_1.size() < needed)
        g_dev_host_tiled_1.resize(needed, bfloat16(0.0f));

    EnqueueReadMeshBuffer(cq, g_dev_host_tiled_1, buf, true);

    const uint16_t* oht = reinterpret_cast<const uint16_t*>(g_dev_host_tiled_1.data());
    uint32_t out_tile_cols = padded / TILE_WIDTH;
    uint32_t* ybits = reinterpret_cast<uint32_t*>(out);
    for (uint32_t tc = 0; tc < out_tile_cols; tc++) {
        uint32_t tile_off = tc * TILE_HEIGHT * TILE_WIDTH;
        uint32_t base = tc * TILE_WIDTH;
        // Untilize: face 0 (16 elements) then face 2 (16 elements)
        for (uint32_t i = 0; i < 16 && base + i < len; i++)
            ybits[base + i] = static_cast<uint32_t>(oht[tile_off + i]) << 16;
        for (uint32_t i = 0; i < 16 && base + 16 + i < len; i++)
            ybits[base + 16 + i] = static_cast<uint32_t>(oht[tile_off + 256 + i]) << 16;
    }
}

// Fast approximate exp() using Schraudolph's method (IEEE 754 trick)
// Max relative error ~1.7%, sufficient for SiLU activation and softplus
static inline float fast_expf(float x) {
    // Clamp to avoid overflow/underflow
    x = std::max(-88.0f, std::min(88.0f, x));
    // Schraudolph's approximation: exp(x) ≈ 2^(x/ln2) via float bit manipulation
    union { float f; int32_t i; } u;
    u.i = (int32_t)(12102203.0f * x + 1065353216.0f);
    return u.f;
}

static inline float fast_sigmoidf(float x) {
    return 1.0f / (1.0f + fast_expf(-x));
}

static inline float fast_siluf(float x) {
    return x * fast_sigmoidf(x);
}

// AVX-512 vectorized SiLU: x * sigmoid(x)
static inline __m512 fast_silu_avx512(__m512 x) {
    // Schraudolph's exp(-x): clamp, then bit manipulation
    __m512 neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x);
    neg_x = _mm512_max_ps(_mm512_set1_ps(-88.0f), _mm512_min_ps(_mm512_set1_ps(88.0f), neg_x));
    __m512i exp_i = _mm512_cvtps_epi32(_mm512_fmadd_ps(neg_x, _mm512_set1_ps(12102203.0f), _mm512_set1_ps(1065353216.0f)));
    __m512 exp_neg_x = _mm512_castsi512_ps(exp_i);
    // sigmoid(x) = 1 / (1 + exp(-x))
    __m512 sigmoid = _mm512_div_ps(_mm512_set1_ps(1.0f), _mm512_add_ps(_mm512_set1_ps(1.0f), exp_neg_x));
    return _mm512_mul_ps(x, sigmoid);
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
// Batched prefill: process N tokens (N<=32) through all layers using batch GEMV.
// Reads weights once per batch instead of once per token — big bandwidth win.
// Host-side RMSNorm, SwiGLU, residual add; SSM/attention still per-token.
// ============================================================================
static constexpr int PREFILL_BATCH = 32;

// Scratch buffers for batch prefill (allocated on first use)
static std::vector<float> g_batch_hidden;     // [32, n_embd]
static std::vector<float> g_batch_norm;       // [32, n_embd]
static std::vector<float> g_batch_proj;       // [32, max(combined_rows, qkv_rows)]
static std::vector<float> g_batch_outproj;    // [32, n_embd]
static std::vector<float> g_batch_ffn_out;    // [32, 2*n_ff or 2*n_ff_tp]
static std::vector<float> g_batch_ffn_act;    // [32, n_ff or n_ff_tp]
static std::vector<float> g_batch_ffn_down;   // [32, n_embd]
static std::vector<float> g_batch_ffn_down1;  // [32, n_embd] (chip 1 partial for TP)
static std::vector<bfloat16> g_batch_tiled;   // staging buffer for batch PCIe transfers
static bool g_prefill_bufs_inited = false;

static void init_prefill_bufs() {
    if (g_prefill_bufs_inited) return;
    int max_proj = std::max(g_combined_rows, g_qkv_rows);
    g_batch_hidden.resize(PREFILL_BATCH * MC::n_embd);
    g_batch_norm.resize(PREFILL_BATCH * MC::n_embd);
    g_batch_proj.resize(PREFILL_BATCH * max_proj);
    g_batch_outproj.resize(PREFILL_BATCH * MC::n_embd);
    int ffn_dim = g_mesh1 ? (int)n_ff_tp : MC::n_ff;
    g_batch_ffn_out.resize(PREFILL_BATCH * 2 * ffn_dim);
    g_batch_ffn_act.resize(PREFILL_BATCH * ffn_dim);
    g_batch_ffn_down.resize(PREFILL_BATCH * MC::n_embd);
    if (g_mesh1) g_batch_ffn_down1.resize(PREFILL_BATCH * MC::n_embd);
    g_prefill_bufs_inited = true;
}

// Write [N, dim] row-major f32 data into tiled BF16 format on device.
// N <= 32. Fills all N rows of each 32x32 tile.
static void write_batch_to_buf(std::shared_ptr<MeshBuffer> buf,
                                const float* data, int N, uint32_t dim,
                                MeshDevice* device = nullptr) {
    auto& cq = (device ? device : g_mesh.get())->mesh_command_queue();
    uint32_t dim_padded = ((dim + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;
    uint32_t num_tile_cols = dim_padded / TILE_WIDTH;
    size_t needed = (size_t)num_tile_cols * TILE_HEIGHT * TILE_WIDTH;
    if (g_batch_tiled.size() < needed)
        g_batch_tiled.resize(needed);
    memset(g_batch_tiled.data(), 0, needed * sizeof(bfloat16));

    uint16_t* ht = reinterpret_cast<uint16_t*>(g_batch_tiled.data());

    for (int r = 0; r < N; r++) {
        const uint32_t* row_bits = reinterpret_cast<const uint32_t*>(data + (size_t)r * dim);
        // Face layout: rows 0-15 → faces 0,2 (offsets 0, 256)
        //              rows 16-31 → faces 1,3 (offsets 512, 768)
        uint32_t face_base_lo = (r < 16) ? 0 : 512;       // face 0 or 1
        uint32_t face_base_hi = face_base_lo + 256;         // face 2 or 3
        uint32_t row_off = (r % 16) * 16;

        for (uint32_t tc = 0; tc < num_tile_cols; tc++) {
            uint32_t tile_off = tc * TILE_HEIGHT * TILE_WIDTH;
            uint32_t col_base = tc * TILE_WIDTH;
            uint32_t pos0 = tile_off + face_base_lo + row_off;
            uint32_t pos1 = tile_off + face_base_hi + row_off;

            if (col_base + 32 <= dim) {
                __m512i v0 = _mm512_loadu_si512(row_bits + col_base);
                __m512i v1 = _mm512_loadu_si512(row_bits + col_base + 16);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(ht + pos0),
                                    _mm512_cvtepi32_epi16(_mm512_srli_epi32(v0, 16)));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(ht + pos1),
                                    _mm512_cvtepi32_epi16(_mm512_srli_epi32(v1, 16)));
            } else {
                for (uint32_t c = 0; c < 16 && col_base + c < dim; c++)
                    ht[pos0 + c] = static_cast<uint16_t>(row_bits[col_base + c] >> 16);
                for (uint32_t c = 0; c < 16 && col_base + 16 + c < dim; c++)
                    ht[pos1 + c] = static_cast<uint16_t>(row_bits[col_base + 16 + c] >> 16);
            }
        }
    }
    EnqueueWriteMeshBuffer(cq, buf, g_batch_tiled, false);
}

// Read [N, dim] from tiled BF16 output tiles into row-major f32.
// tiled_data is the host-side copy of the output buffer.
static void read_batch_from_tiles(const bfloat16* tiled_data, float* out,
                                   int N, uint32_t dim) {
    uint32_t dim_padded = ((dim + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;
    uint32_t num_tile_cols = dim_padded / TILE_WIDTH;
    const uint16_t* ht = reinterpret_cast<const uint16_t*>(tiled_data);

    for (int r = 0; r < N; r++) {
        uint32_t* row_bits = reinterpret_cast<uint32_t*>(out + (size_t)r * dim);
        uint32_t face_base_lo = (r < 16) ? 0 : 512;
        uint32_t face_base_hi = face_base_lo + 256;
        uint32_t row_off = (r % 16) * 16;

        for (uint32_t tc = 0; tc < num_tile_cols; tc++) {
            uint32_t tile_off = tc * TILE_HEIGHT * TILE_WIDTH;
            uint32_t col_base = tc * TILE_WIDTH;
            uint32_t pos0 = tile_off + face_base_lo + row_off;
            uint32_t pos1 = tile_off + face_base_hi + row_off;

            if (col_base + 32 <= dim) {
                __m256i f0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ht + pos0));
                __m256i f1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ht + pos1));
                _mm512_storeu_si512(row_bits + col_base,
                    _mm512_slli_epi32(_mm512_cvtepu16_epi32(f0), 16));
                _mm512_storeu_si512(row_bits + col_base + 16,
                    _mm512_slli_epi32(_mm512_cvtepu16_epi32(f1), 16));
            } else {
                for (uint32_t c = 0; c < 16 && col_base + c < dim; c++)
                    row_bits[col_base + c] = static_cast<uint32_t>(ht[pos0 + c]) << 16;
                for (uint32_t c = 0; c < 16 && col_base + 16 + c < dim; c++)
                    row_bits[col_base + 16 + c] = static_cast<uint32_t>(ht[pos1 + c]) << 16;
            }
        }
    }
}

// Batch GEMV helper: write batch activation → dispatch_gemv → read batch output.
// Returns pointer to output in out_buf (caller provides host staging).
static void batch_gemv(MeshDevice* device,
                        const float* act, int N, uint32_t K,
                        std::shared_ptr<MeshBuffer> act_dev_buf,
                        std::shared_ptr<MeshBuffer> weight_buf,
                        uint32_t M,
                        float* out_host) {
    write_batch_to_buf(act_dev_buf, act, N, K, device);
    auto& gb = get_gemv_buf(device, M, K);
    dispatch_gemv(device, act_dev_buf, weight_buf, gb.out_buf, M, K);
    auto& cq = device->mesh_command_queue();
    EnqueueReadMeshBuffer(cq, gb.out_host_tiled, gb.out_buf, true);
    read_batch_from_tiles(gb.out_host_tiled.data(), out_host, N, M);
}

// Forward prefill: process N tokens (N<=32) through all 32 layers.
// Updates g_pos, SSM state, KV caches. Returns logits for last token.
static float* forward_prefill(const int* tokens, int N) {
    init_prefill_bufs();
    auto& cq0 = g_mesh->mesh_command_queue();

    // 1. Embed all tokens
    for (int i = 0; i < N; i++) {
        const uint16_t* emb = g_model.tok_embd_host.data() + (size_t)tokens[i] * MC::n_embd;
        uint32_t* hbits = reinterpret_cast<uint32_t*>(g_batch_hidden.data() + (size_t)i * MC::n_embd);
        for (int j = 0; j < MC::n_embd; j++)
            hbits[j] = static_cast<uint32_t>(emb[j]) << 16;
    }

    int attn_idx = 0, ssm_idx = 0;
    for (int layer = 0; layer < MC::n_layers; layer++) {
        float* hidden = g_batch_hidden.data();
        float* norm_buf = g_batch_norm.data();

        // Host RMSNorm for all N tokens
        const float* norm_w = g_layer_norms[layer].attn_norm.data();
        for (int i = 0; i < N; i++)
            rmsnorm(hidden + (size_t)i * MC::n_embd, norm_w,
                    norm_buf + (size_t)i * MC::n_embd, MC::n_embd);

        if (MC::is_recurrent(layer)) {
            // ======== SSM (Delta-Net) Layer ========
            auto& lw = g_model.ssm_layers[ssm_idx];

            // Batch combined projection GEMV
            batch_gemv(g_mesh.get(), norm_buf, N, MC::n_embd,
                       g_norm_dev_buf, g_wt.ssm_w_combined_buf[ssm_idx],
                       g_combined_rows, g_batch_proj.data());
            // Per-token SSM processing (sequential: conv1d + delta-net)
            for (int ti = 0; ti < N; ti++) {
                float* proj = g_batch_proj.data() + (size_t)ti * g_combined_rows;
                float* qkv_raw = proj;
                float* z_raw = proj + MC::ssm_conv_channels;
                float* alpha_raw = z_raw + MC::ssm_d_inner;
                float* beta_raw = alpha_raw + MC::ssm_dt_rank;

                // Conv1d + SiLU
                auto& cs = g_conv_state[ssm_idx];
                float* conv_out = g_conv_out.data();
                const float* w = lw.ssm_conv1d_host.data();
                constexpr int C = MC::ssm_conv_channels;
                const float* s0 = cs.data();
                const float* s1 = cs.data() + C;
                const float* s2 = cs.data() + 2 * C;
                for (int ch = 0; ch < C; ch += 16) {
                    __m512 vw0 = _mm512_loadu_ps(w + ch);
                    __m512 vw1 = _mm512_loadu_ps(w + C + ch);
                    __m512 vw2 = _mm512_loadu_ps(w + 2 * C + ch);
                    __m512 vw3 = _mm512_loadu_ps(w + 3 * C + ch);
                    __m512 vsum = _mm512_mul_ps(_mm512_loadu_ps(s0 + ch), vw0);
                    vsum = _mm512_fmadd_ps(_mm512_loadu_ps(s1 + ch), vw1, vsum);
                    vsum = _mm512_fmadd_ps(_mm512_loadu_ps(s2 + ch), vw2, vsum);
                    vsum = _mm512_fmadd_ps(_mm512_loadu_ps(qkv_raw + ch), vw3, vsum);
                    _mm512_storeu_ps(conv_out + ch, fast_silu_avx512(vsum));
                }
                memcpy(cs.data(), s1, C * sizeof(float));
                memcpy(cs.data() + C, s2, C * sizeof(float));
                memcpy(cs.data() + 2 * C, qkv_raw, C * sizeof(float));

                // Split conv output: Q | K | V
                constexpr int num_k_heads = MC::ssm_n_group;
                constexpr int head_k = MC::ssm_d_state;
                constexpr int num_v = ssm_n_v_heads;
                constexpr int head_v = ssm_head_v_dim_c;
                float* conv_q = conv_out;
                float* conv_k = conv_out + num_k_heads * head_k;
                float* conv_v = conv_out + 2 * num_k_heads * head_k;

                // Delta-net recurrence + gated RMSNorm
                auto& state = g_ssm_state[ssm_idx];
                float* ssm_proj_in = g_ssm_proj_in.data();
                constexpr float ssm_scale = 1.0f / 11.3137f;

                for (int vh = 0; vh < num_v; vh++) {
                    int kh = vh % num_k_heads;
                    alignas(64) float q[head_k], k_vec[head_k], v_vec[head_v];
                    memcpy(q, conv_q + kh * head_k, head_k * sizeof(float));
                    memcpy(k_vec, conv_k + kh * head_k, head_k * sizeof(float));
                    memcpy(v_vec, conv_v + vh * head_v, head_v * sizeof(float));

                    // RMSNorm Q and K
                    __m512 vqn = _mm512_setzero_ps(), vkn = _mm512_setzero_ps();
                    for (int d = 0; d < head_k; d += 16) {
                        __m512 vq = _mm512_load_ps(q + d);
                        __m512 vk = _mm512_load_ps(k_vec + d);
                        vqn = _mm512_fmadd_ps(vq, vq, vqn);
                        vkn = _mm512_fmadd_ps(vk, vk, vkn);
                    }
                    float qn_s = 1.0f / sqrtf(_mm512_reduce_add_ps(vqn) + MC::rms_norm_eps);
                    float kn_s = 1.0f / sqrtf(_mm512_reduce_add_ps(vkn) + MC::rms_norm_eps);
                    __m512 vqn_b = _mm512_set1_ps(qn_s);
                    __m512 vkn_b = _mm512_set1_ps(kn_s);
                    for (int d = 0; d < head_k; d += 16) {
                        _mm512_store_ps(q + d, _mm512_mul_ps(_mm512_load_ps(q + d), vqn_b));
                        _mm512_store_ps(k_vec + d, _mm512_mul_ps(_mm512_load_ps(k_vec + d), vkn_b));
                    }

                    float biased = alpha_raw[vh] + lw.ssm_dt_bias_host[vh];
                    float sp = (biased > 20.0f) ? biased : logf(1.0f + fast_expf(biased));
                    float gate_val = sp * lw.ssm_a_host[vh];
                    float decay = expf(gate_val);
                    float beta_val = fast_sigmoidf(beta_raw[vh]);
                    float* sh = state.data() + vh * head_v * head_k;

                    __m512 vdecay = _mm512_set1_ps(decay);
                    for (int i = 0; i < head_v; i++) {
                        float* row = sh + i * head_k;
                        __m512 vsk = _mm512_setzero_ps();
                        for (int j = 0; j < head_k; j += 16) {
                            __m512 vr = _mm512_mul_ps(_mm512_loadu_ps(row + j), vdecay);
                            _mm512_storeu_ps(row + j, vr);
                            vsk = _mm512_fmadd_ps(vr, _mm512_load_ps(k_vec + j), vsk);
                        }
                        float sk = _mm512_reduce_add_ps(vsk);
                        float dd = beta_val * (v_vec[i] - sk);
                        __m512 vdd = _mm512_set1_ps(dd);
                        __m512 vout_acc = _mm512_setzero_ps();
                        for (int j = 0; j < head_k; j += 16) {
                            __m512 vr = _mm512_loadu_ps(row + j);
                            __m512 vk = _mm512_load_ps(k_vec + j);
                            vr = _mm512_fmadd_ps(vk, vdd, vr);
                            _mm512_storeu_ps(row + j, vr);
                            vout_acc = _mm512_fmadd_ps(vr, _mm512_load_ps(q + j), vout_acc);
                        }
                        ssm_proj_in[vh * head_v + i] = _mm512_reduce_add_ps(vout_acc) * ssm_scale;
                    }

                    // Gated RMSNorm
                    float* vo = ssm_proj_in + vh * head_v;
                    __m512 vssq = _mm512_setzero_ps();
                    for (int d = 0; d < head_v; d += 16) {
                        __m512 vv = _mm512_loadu_ps(vo + d);
                        vssq = _mm512_fmadd_ps(vv, vv, vssq);
                    }
                    float rms = 1.0f / sqrtf(_mm512_reduce_add_ps(vssq) / head_v + MC::rms_norm_eps);
                    __m512 vrms = _mm512_set1_ps(rms);
                    for (int d = 0; d < head_v; d += 16) {
                        __m512 vv = _mm512_loadu_ps(vo + d);
                        __m512 vnorm = _mm512_mul_ps(_mm512_mul_ps(vv, vrms),
                                                      _mm512_loadu_ps(lw.ssm_norm_host.data() + d));
                        __m512 vz = _mm512_loadu_ps(z_raw + vh * head_v + d);
                        _mm512_storeu_ps(vo + d, _mm512_mul_ps(vnorm, fast_silu_avx512(vz)));
                    }
                }

                // Store SSM output for this token into batch outproj buffer
                memcpy(g_batch_outproj.data() + (size_t)ti * MC::n_embd,
                       ssm_proj_in, MC::ssm_d_inner * sizeof(float));
                // Zero-pad if ssm_d_inner < n_embd (they're equal for this model)
            }
            // Batch outproj GEMV: [N, ssm_d_inner] → [N, n_embd]
            batch_gemv(g_mesh.get(), g_batch_outproj.data(), N, MC::ssm_d_inner,
                       g_norm_dev_buf, g_wt.ssm_out_buf[ssm_idx],
                       MC::n_embd, g_batch_ffn_down.data());
            // Residual add on host
            for (int i = 0; i < N; i++)
                for (int j = 0; j < MC::n_embd; j++)
                    hidden[(size_t)i * MC::n_embd + j] += g_batch_ffn_down[(size_t)i * MC::n_embd + j];

            // FFN: norm → gate+up GEMV → SwiGLU → down GEMV → residual add
            const float* post_norm_w = g_layer_norms[layer].post_norm.data();
            for (int i = 0; i < N; i++)
                rmsnorm(hidden + (size_t)i * MC::n_embd, post_norm_w,
                        norm_buf + (size_t)i * MC::n_embd, MC::n_embd);

            if (g_mesh1) {
                // TP FFN: overlap chip 0 and chip 1 gate+up GEMVs
                auto& cq1 = g_mesh1->mesh_command_queue();

                // Write activations to both chips concurrently
                write_batch_to_buf(g_norm_dev_buf, norm_buf, N, MC::n_embd, g_mesh.get());
                write_batch_to_buf(g_norm_dev_buf_1, norm_buf, N, MC::n_embd, g_mesh1.get());

                // Dispatch gate+up on both chips (non-blocking)
                auto& gb0_gu = get_gemv_buf(g_mesh.get(), n_ff_tp * 2, MC::n_embd);
                dispatch_gemv(g_mesh.get(), g_norm_dev_buf, g_wt.ssm_ffn_gate_up_buf[ssm_idx],
                              gb0_gu.out_buf, n_ff_tp * 2, MC::n_embd);
                auto& gb1_gu = get_gemv_buf(g_mesh1.get(), n_ff_tp * 2, MC::n_embd);
                dispatch_gemv(g_mesh1.get(), g_norm_dev_buf_1, g_wt.ssm_ffn_gate_up_buf_1[ssm_idx],
                              gb1_gu.out_buf, n_ff_tp * 2, MC::n_embd);

                // Read both results (blocking)
                EnqueueReadMeshBuffer(cq0, gb0_gu.out_host_tiled, gb0_gu.out_buf, true);
                EnqueueReadMeshBuffer(cq1, gb1_gu.out_host_tiled, gb1_gu.out_buf, true);

                // Untilize both
                read_batch_from_tiles(gb0_gu.out_host_tiled.data(), g_batch_ffn_out.data(), N, n_ff_tp * 2);
                static std::vector<float> batch_gu1;
                if (batch_gu1.size() < (size_t)N * n_ff_tp * 2)
                    batch_gu1.resize((size_t)N * n_ff_tp * 2);
                read_batch_from_tiles(gb1_gu.out_host_tiled.data(), batch_gu1.data(), N, n_ff_tp * 2);

                // SwiGLU on both halves
                static std::vector<float> batch_act1;
                if (batch_act1.size() < (size_t)N * n_ff_tp)
                    batch_act1.resize((size_t)N * n_ff_tp);
                for (int i = 0; i < N; i++) {
                    float* gu0 = g_batch_ffn_out.data() + (size_t)i * n_ff_tp * 2;
                    float* act0 = g_batch_ffn_act.data() + (size_t)i * n_ff_tp;
                    float* gu1 = batch_gu1.data() + (size_t)i * n_ff_tp * 2;
                    float* act1 = batch_act1.data() + (size_t)i * n_ff_tp;
                    for (uint32_t j = 0; j < n_ff_tp; j += 16) {
                        __m512 vg0 = _mm512_loadu_ps(gu0 + j);
                        __m512 vu0 = _mm512_loadu_ps(gu0 + n_ff_tp + j);
                        _mm512_storeu_ps(act0 + j, _mm512_mul_ps(fast_silu_avx512(vg0), vu0));
                        __m512 vg1 = _mm512_loadu_ps(gu1 + j);
                        __m512 vu1 = _mm512_loadu_ps(gu1 + n_ff_tp + j);
                        _mm512_storeu_ps(act1 + j, _mm512_mul_ps(fast_silu_avx512(vg1), vu1));
                    }
                }

                // Overlap chip 0 and chip 1 down GEMVs
                write_batch_to_buf(g_tp_ffn_0.act_buf, g_batch_ffn_act.data(), N, n_ff_tp, g_mesh.get());
                write_batch_to_buf(g_tp_ffn_1.act_buf, batch_act1.data(), N, n_ff_tp, g_mesh1.get());

                auto& gb0_down = get_gemv_buf(g_mesh.get(), MC::n_embd, n_ff_tp);
                dispatch_gemv(g_mesh.get(), g_tp_ffn_0.act_buf, g_wt.ssm_ffn_down_buf[ssm_idx],
                              gb0_down.out_buf, MC::n_embd, n_ff_tp);
                auto& gb1_down = get_gemv_buf(g_mesh1.get(), MC::n_embd, n_ff_tp);
                dispatch_gemv(g_mesh1.get(), g_tp_ffn_1.act_buf, g_wt.ssm_ffn_down_buf_1[ssm_idx],
                              gb1_down.out_buf, MC::n_embd, n_ff_tp);

                EnqueueReadMeshBuffer(cq0, gb0_down.out_host_tiled, gb0_down.out_buf, true);
                EnqueueReadMeshBuffer(cq1, gb1_down.out_host_tiled, gb1_down.out_buf, true);

                read_batch_from_tiles(gb0_down.out_host_tiled.data(), g_batch_ffn_down.data(), N, MC::n_embd);
                read_batch_from_tiles(gb1_down.out_host_tiled.data(), g_batch_ffn_down1.data(), N, MC::n_embd);

                // Residual add: hidden += down_chip0 + down_chip1
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < MC::n_embd; j++)
                        hidden[(size_t)i * MC::n_embd + j] +=
                            g_batch_ffn_down[(size_t)i * MC::n_embd + j] +
                            g_batch_ffn_down1[(size_t)i * MC::n_embd + j];
            } else {
                // Non-TP FFN: full n_ff on chip 0
                batch_gemv(g_mesh.get(), norm_buf, N, MC::n_embd,
                           g_norm_dev_buf, g_wt.ssm_ffn_gate_up_buf[ssm_idx],
                           MC::n_ff * 2, g_batch_ffn_out.data());
                // SwiGLU
                for (int i = 0; i < N; i++) {
                    float* gu = g_batch_ffn_out.data() + (size_t)i * MC::n_ff * 2;
                    float* act = g_batch_ffn_act.data() + (size_t)i * MC::n_ff;
                    for (int j = 0; j < MC::n_ff; j += 16) {
                        __m512 vg = _mm512_loadu_ps(gu + j);
                        __m512 vu = _mm512_loadu_ps(gu + MC::n_ff + j);
                        _mm512_storeu_ps(act + j, _mm512_mul_ps(fast_silu_avx512(vg), vu));
                    }
                }
                // Down GEMV
                batch_gemv(g_mesh.get(), g_batch_ffn_act.data(), N, MC::n_ff,
                           get_ffn_buf(g_mesh.get()).act_buf, g_wt.ssm_ffn_down_buf[ssm_idx],
                           MC::n_embd, g_batch_ffn_down.data());
                // Residual add
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < MC::n_embd; j++)
                        hidden[(size_t)i * MC::n_embd + j] += g_batch_ffn_down[(size_t)i * MC::n_embd + j];
            }
            ssm_idx++;
        } else {
            // ======== Full Attention Layer ========
            auto& lw = g_model.attn_layers[attn_idx];
            auto& aw = g_attn_small[attn_idx];

            // Batch QKV projection GEMV
            batch_gemv(g_mesh.get(), norm_buf, N, MC::n_embd,
                       g_norm_dev_buf, g_wt.attn_wqkv_buf[attn_idx],
                       g_qkv_rows, g_batch_proj.data());

            // Per-token attention processing
            for (int ti = 0; ti < N; ti++) {
                int pos = g_pos + ti;
                float* qkv = g_batch_proj.data() + (size_t)ti * g_qkv_rows;

                // Deinterleave Q and gate
                constexpr int q_dim = MC::n_head * MC::head_dim * 2;
                constexpr int kv_dim_one = MC::n_head_kv * MC::head_dim;
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

                // Per-head Q/K RMSNorm
                for (int h = 0; h < MC::n_head; h++) {
                    float* qh = q_heads + h * MC::head_dim;
                    float ss = 0;
                    for (int d = 0; d < MC::head_dim; d++) ss += qh[d] * qh[d];
                    float rms_val = 1.0f / sqrtf(ss / MC::head_dim + MC::rms_norm_eps);
                    for (int d = 0; d < MC::head_dim; d++)
                        qh[d] = qh[d] * rms_val * aw.q_norm[d];
                }
                for (int h = 0; h < MC::n_head_kv; h++) {
                    float* kh = k_proj + h * MC::head_dim;
                    float ss = 0;
                    for (int d = 0; d < MC::head_dim; d++) ss += kh[d] * kh[d];
                    float rms_val = 1.0f / sqrtf(ss / MC::head_dim + MC::rms_norm_eps);
                    for (int d = 0; d < MC::head_dim; d++)
                        kh[d] = kh[d] * rms_val * aw.k_norm[d];
                }

                // RoPE
                for (int h = 0; h < MC::n_head; h++)
                    apply_rope_cached(q_heads + h * MC::head_dim, pos);
                for (int h = 0; h < MC::n_head_kv; h++)
                    apply_rope_cached(k_proj + h * MC::head_dim, pos);

                // KV cache
                memcpy(g_k_cache[attn_idx].data() + (size_t)pos * kv_dim,
                       k_proj, kv_dim * sizeof(float));
                memcpy(g_v_cache[attn_idx].data() + (size_t)pos * kv_dim,
                       v_proj, kv_dim * sizeof(float));
                int kv_len = pos + 1;

                // Attention (online softmax) + sigmoid gating
                float* attn_out = g_attn_out.data();
                for (int h = 0; h < MC::n_head; h++) {
                    int kv_h = h / (MC::n_head / MC::n_head_kv);
                    float* qh = q_heads + h * MC::head_dim;
                    float* out = attn_out + h * MC::head_dim;
                    alignas(64) float acc[MC::head_dim];
                    memset(acc, 0, MC::head_dim * sizeof(float));
                    float max_score = -FLT_MAX, sum_exp = 0;
                    for (int kp = 0; kp < kv_len; kp++) {
                        float* kh = g_k_cache[attn_idx].data() + (size_t)kp * kv_dim + kv_h * MC::head_dim;
                        __m512 vdot = _mm512_setzero_ps();
                        for (int d = 0; d < MC::head_dim; d += 16)
                            vdot = _mm512_fmadd_ps(_mm512_loadu_ps(qh + d), _mm512_loadu_ps(kh + d), vdot);
                        float score = _mm512_reduce_add_ps(vdot) * MC::attn_scale;
                        float new_max = std::max(max_score, score);
                        float exp_s = expf(score - new_max);
                        float corr = expf(max_score - new_max);
                        sum_exp = sum_exp * corr + exp_s;
                        float* vh = g_v_cache[attn_idx].data() + (size_t)kp * kv_dim + kv_h * MC::head_dim;
                        __m512 vcorr = _mm512_set1_ps(corr);
                        __m512 vexp = _mm512_set1_ps(exp_s);
                        for (int d = 0; d < MC::head_dim; d += 16) {
                            __m512 va = _mm512_load_ps(acc + d);
                            va = _mm512_fmadd_ps(vexp, _mm512_loadu_ps(vh + d), _mm512_mul_ps(va, vcorr));
                            _mm512_store_ps(acc + d, va);
                        }
                        max_score = new_max;
                    }
                    float* gh = gate_heads + h * MC::head_dim;
                    for (int d = 0; d < MC::head_dim; d++)
                        out[d] = (acc[d] / sum_exp) * fast_sigmoidf(gh[d]);
                }

                // Store attention output for this token
                memcpy(g_batch_outproj.data() + (size_t)ti * MC::n_embd,
                       attn_out, MC::n_head * MC::head_dim * sizeof(float));
            }

            // Batch outproj GEMV: [N, n_head*head_dim] → [N, n_embd]
            batch_gemv(g_mesh.get(), g_batch_outproj.data(), N, MC::n_head * MC::head_dim,
                       g_norm_dev_buf, g_wt.attn_wo_buf[attn_idx],
                       MC::n_embd, g_batch_ffn_down.data());

            // Residual add
            for (int i = 0; i < N; i++)
                for (int j = 0; j < MC::n_embd; j++)
                    hidden[(size_t)i * MC::n_embd + j] += g_batch_ffn_down[(size_t)i * MC::n_embd + j];

            // FFN: norm → gate+up → SwiGLU → down → residual
            const float* post_norm_w = g_layer_norms[layer].post_norm.data();
            for (int i = 0; i < N; i++)
                rmsnorm(hidden + (size_t)i * MC::n_embd, post_norm_w,
                        norm_buf + (size_t)i * MC::n_embd, MC::n_embd);

            if (g_mesh1) {
                // TP FFN: overlap chip 0 and chip 1 gate+up GEMVs
                auto& cq1 = g_mesh1->mesh_command_queue();

                write_batch_to_buf(g_norm_dev_buf, norm_buf, N, MC::n_embd, g_mesh.get());
                write_batch_to_buf(g_norm_dev_buf_1, norm_buf, N, MC::n_embd, g_mesh1.get());

                auto& gb0_gu = get_gemv_buf(g_mesh.get(), n_ff_tp * 2, MC::n_embd);
                dispatch_gemv(g_mesh.get(), g_norm_dev_buf, g_wt.attn_ffn_gate_up_buf[attn_idx],
                              gb0_gu.out_buf, n_ff_tp * 2, MC::n_embd);
                auto& gb1_gu = get_gemv_buf(g_mesh1.get(), n_ff_tp * 2, MC::n_embd);
                dispatch_gemv(g_mesh1.get(), g_norm_dev_buf_1, g_wt.attn_ffn_gate_up_buf_1[attn_idx],
                              gb1_gu.out_buf, n_ff_tp * 2, MC::n_embd);

                EnqueueReadMeshBuffer(cq0, gb0_gu.out_host_tiled, gb0_gu.out_buf, true);
                EnqueueReadMeshBuffer(cq1, gb1_gu.out_host_tiled, gb1_gu.out_buf, true);

                read_batch_from_tiles(gb0_gu.out_host_tiled.data(), g_batch_ffn_out.data(), N, n_ff_tp * 2);
                static std::vector<float> batch_gu1_attn;
                if (batch_gu1_attn.size() < (size_t)N * n_ff_tp * 2)
                    batch_gu1_attn.resize((size_t)N * n_ff_tp * 2);
                read_batch_from_tiles(gb1_gu.out_host_tiled.data(), batch_gu1_attn.data(), N, n_ff_tp * 2);

                // SwiGLU on both halves
                static std::vector<float> batch_act1_attn;
                if (batch_act1_attn.size() < (size_t)N * n_ff_tp)
                    batch_act1_attn.resize((size_t)N * n_ff_tp);
                for (int i = 0; i < N; i++) {
                    float* gu0 = g_batch_ffn_out.data() + (size_t)i * n_ff_tp * 2;
                    float* act0 = g_batch_ffn_act.data() + (size_t)i * n_ff_tp;
                    float* gu1 = batch_gu1_attn.data() + (size_t)i * n_ff_tp * 2;
                    float* act1 = batch_act1_attn.data() + (size_t)i * n_ff_tp;
                    for (uint32_t j = 0; j < n_ff_tp; j += 16) {
                        __m512 vg0 = _mm512_loadu_ps(gu0 + j);
                        __m512 vu0 = _mm512_loadu_ps(gu0 + n_ff_tp + j);
                        _mm512_storeu_ps(act0 + j, _mm512_mul_ps(fast_silu_avx512(vg0), vu0));
                        __m512 vg1 = _mm512_loadu_ps(gu1 + j);
                        __m512 vu1 = _mm512_loadu_ps(gu1 + n_ff_tp + j);
                        _mm512_storeu_ps(act1 + j, _mm512_mul_ps(fast_silu_avx512(vg1), vu1));
                    }
                }

                // Overlap chip 0 and chip 1 down GEMVs
                write_batch_to_buf(g_tp_ffn_0.act_buf, g_batch_ffn_act.data(), N, n_ff_tp, g_mesh.get());
                write_batch_to_buf(g_tp_ffn_1.act_buf, batch_act1_attn.data(), N, n_ff_tp, g_mesh1.get());

                auto& gb0_down = get_gemv_buf(g_mesh.get(), MC::n_embd, n_ff_tp);
                dispatch_gemv(g_mesh.get(), g_tp_ffn_0.act_buf, g_wt.attn_ffn_down_buf[attn_idx],
                              gb0_down.out_buf, MC::n_embd, n_ff_tp);
                auto& gb1_down = get_gemv_buf(g_mesh1.get(), MC::n_embd, n_ff_tp);
                dispatch_gemv(g_mesh1.get(), g_tp_ffn_1.act_buf, g_wt.attn_ffn_down_buf_1[attn_idx],
                              gb1_down.out_buf, MC::n_embd, n_ff_tp);

                EnqueueReadMeshBuffer(cq0, gb0_down.out_host_tiled, gb0_down.out_buf, true);
                EnqueueReadMeshBuffer(cq1, gb1_down.out_host_tiled, gb1_down.out_buf, true);

                read_batch_from_tiles(gb0_down.out_host_tiled.data(), g_batch_ffn_down.data(), N, MC::n_embd);
                read_batch_from_tiles(gb1_down.out_host_tiled.data(), g_batch_ffn_down1.data(), N, MC::n_embd);

                for (int i = 0; i < N; i++)
                    for (int j = 0; j < MC::n_embd; j++)
                        hidden[(size_t)i * MC::n_embd + j] +=
                            g_batch_ffn_down[(size_t)i * MC::n_embd + j] +
                            g_batch_ffn_down1[(size_t)i * MC::n_embd + j];
            } else {
                batch_gemv(g_mesh.get(), norm_buf, N, MC::n_embd,
                           g_norm_dev_buf, g_wt.attn_ffn_gate_up_buf[attn_idx],
                           MC::n_ff * 2, g_batch_ffn_out.data());
                for (int i = 0; i < N; i++) {
                    float* gu = g_batch_ffn_out.data() + (size_t)i * MC::n_ff * 2;
                    float* act = g_batch_ffn_act.data() + (size_t)i * MC::n_ff;
                    for (int j = 0; j < MC::n_ff; j += 16) {
                        __m512 vg = _mm512_loadu_ps(gu + j);
                        __m512 vu = _mm512_loadu_ps(gu + MC::n_ff + j);
                        _mm512_storeu_ps(act + j, _mm512_mul_ps(fast_silu_avx512(vg), vu));
                    }
                }
                batch_gemv(g_mesh.get(), g_batch_ffn_act.data(), N, MC::n_ff,
                           get_ffn_buf(g_mesh.get()).act_buf, g_wt.attn_ffn_down_buf[attn_idx],
                           MC::n_embd, g_batch_ffn_down.data());
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < MC::n_embd; j++)
                        hidden[(size_t)i * MC::n_embd + j] += g_batch_ffn_down[(size_t)i * MC::n_embd + j];
            }

            attn_idx++;
        }
    }

    // Copy last token's hidden state to g_hidden_f32 for subsequent decode
    memcpy(g_hidden_f32.data(), g_batch_hidden.data() + (size_t)(N - 1) * MC::n_embd,
           MC::n_embd * sizeof(float));
    g_pos += N;

    // LM head: output norm + GEMV for last token only (reuse single-token path)
    rmsnorm(g_hidden_f32.data(), g_output_norm.data(), g_norm_out.data(), MC::n_embd);
    write_f32_to_buf(g_norm_dev_buf, g_norm_out.data(), MC::n_embd);

    float* logits = g_logits.data();
    if (g_mesh1) {
        uint32_t M0 = g_wt.lm_head_Mt0 * TILE_HEIGHT;
        uint32_t M1 = g_wt.lm_head_Mt1 * TILE_HEIGHT;
        auto& gb_lm0 = get_gemv_buf(g_mesh.get(), M0, MC::n_embd);
        auto& gb_lm1 = get_gemv_buf(g_mesh1.get(), M1, MC::n_embd);
        write_f32_to_buf(g_norm_dev_buf_1, g_norm_out.data(), MC::n_embd, g_mesh1.get());
        dispatch_gemv(g_mesh.get(), g_norm_dev_buf, g_wt.lm_head_buf, gb_lm0.out_buf, M0, MC::n_embd);
        dispatch_gemv(g_mesh1.get(), g_norm_dev_buf_1, g_wt.lm_head_buf_1, gb_lm1.out_buf, M1, MC::n_embd);
        auto& cq1 = g_mesh1->mesh_command_queue();
        g_chip1_writer.submit([&]{
            EnqueueReadMeshBuffer(cq1, gb_lm1.out_host_tiled, gb_lm1.out_buf, true);
        });
        EnqueueReadMeshBuffer(cq0, gb_lm0.out_host_tiled, gb_lm0.out_buf, true);
        g_chip1_writer.wait();
        read_gemv_to_f32(gb_lm0, logits, M0);
        read_gemv_to_f32(gb_lm1, logits + M0, M1);
    } else {
        auto& gb_lm = get_gemv_buf(g_mesh.get(), MC::n_vocab, MC::n_embd);
        dispatch_gemv(g_mesh.get(), g_norm_dev_buf, g_lm_head_buf, gb_lm.out_buf,
                      MC::n_vocab, MC::n_embd);
        EnqueueReadMeshBuffer(cq0, gb_lm.out_host_tiled, gb_lm.out_buf, true);
        read_gemv_to_f32(gb_lm, logits, MC::n_vocab);
    }

    return logits;
}

// ============================================================================
// Forward pass: single decode token
// Large matmuls on device via custom GEMV kernels, small ops on host CPU.
// Returns pointer to static g_logits buffer (valid until next call).
// ============================================================================
static int g_decode_count = 0;
static double g_time_gemv = 0, g_time_ffn = 0, g_time_ssm = 0, g_time_attn = 0, g_time_lmhead = 0;
static double g_time_outproj = 0, g_time_reswrite = 0, g_time_host = 0, g_time_norm_mm = 0;
static double g_time_conv1d = 0, g_time_deltanet = 0, g_time_untilize = 0, g_time_attn_host = 0;
static double g_time_ffn_wait = 0, g_time_rmsnorm_write = 0, g_time_gemv_read = 0;
static double g_time_tp_reduce = 0, g_time_ffn_device = 0;

static float* forward_decode() {
    using Clock = std::chrono::high_resolution_clock;
    int pos = g_pos;
    auto& cq0 = g_mesh->mesh_command_queue();

    // Write hidden state to device (chip 0) — stays on device through all layers
    write_hidden_to_device(g_hidden_f32.data());

    // One-time micro-benchmark: measure per-program overhead at decode 5
    // Only uses functions already warmed in the forward pass to avoid
    // allocating new kernel binaries while traces are active.
    if (g_decode_count == 5 && g_verbose) {
        Finish(cq0);
        auto& gb_test = get_gemv_buf(g_mesh.get(), g_combined_rows, MC::n_embd);

        // Test 1: Time 10x single GEMV dispatch+Finish
        auto ta = Clock::now();
        for (int r = 0; r < 10; r++) {
            dispatch_gemv(g_mesh.get(), g_norm_dev_buf, g_wt.ssm_w_combined_buf[0],
                          gb_test.out_buf, g_combined_rows, MC::n_embd);
            Finish(cq0);
        }
        auto tb = Clock::now();
        double t_single = std::chrono::duration<double, std::milli>(tb - ta).count() / 10.0;

        // Test 2: FFN chain trace replay
        auto tg = Clock::now();
        for (int r = 0; r < 10; r++) {
            g_mesh->replay_mesh_trace(0, g_ffn_chain_traces[0], false);
            Finish(cq0);
        }
        auto th = Clock::now();
        double t_ffn_trace = std::chrono::duration<double, std::milli>(th - tg).count() / 10.0;

        // Test 3: rmsnorm alone + Finish
        auto t6a = Clock::now();
        for (int r = 0; r < 10; r++) {
            dispatch_rmsnorm_fpu(g_mesh.get(), g_hidden_dev_buf, g_wt.attn_norm_buf[0],
                                 g_norm_dev_buf, MC::n_embd, MC::n_embd / TILE_WIDTH);
            Finish(cq0);
        }
        auto t6b = Clock::now();
        double t_norm_only = std::chrono::duration<double, std::milli>(t6b - t6a).count() / 10.0;

        printf("=== Per-program overhead benchmark ===\n");
        printf("  Single GEMV + Finish:     %.3f ms\n", t_single);
        printf("  FFN chain trace + Finish: %.3f ms\n", t_ffn_trace);
        printf("  rmsnorm alone + Finish:   %.3f ms\n", t_norm_only);
        fflush(stdout);
    }

    int attn_idx = 0, ssm_idx = 0;
    for (int layer = 0; layer < MC::n_layers; layer++) {
        if (MC::is_recurrent(layer)) {
            // ======== SSM (Delta-Net) Layer ========
            auto& lw = g_model.ssm_layers[ssm_idx];

            // 1. On-device rmsnorm + combined GEMV (traced)
            auto t0 = Clock::now();

            // TP reduction: parallel reads from both chips, combine, write back
            if (layer > 0 && g_mesh1) {
                // Start chip 1 read on bg thread (overlaps with chip 0 read)
                g_chip1_writer.submit([&]{
                    auto& cq1 = g_mesh1->mesh_command_queue();
                    read_device_to_f32_chip1(g_partial_down_buf_1, g_partial_f32.data(), MC::n_embd, cq1);
                });
                read_device_to_f32(g_hidden_dev_buf, g_hidden_f32.data(), MC::n_embd, cq0);
                auto t_dev_done = Clock::now();
                g_time_ffn_device += std::chrono::duration<double, std::milli>(t_dev_done - t0).count();
                g_chip1_writer.wait();
                for (int i = 0; i < MC::n_embd; i++) g_hidden_f32[i] += g_partial_f32[i];
                write_f32_to_buf(g_hidden_dev_buf, g_hidden_f32.data(), MC::n_embd);
                g_time_tp_reduce += std::chrono::duration<double, std::milli>(Clock::now() - t_dev_done).count();
            } else if (layer > 0) {
                Finish(cq0);
            }

            auto t_after_read = Clock::now();
            g_time_ffn_wait += std::chrono::duration<double, std::milli>(t_after_read - t0).count();

            // Start async write of hidden to chip 1 (overlaps with GEMV on chip 0)
            if (g_mesh1) {
                g_chip1_writer.submit([&]{
                    write_f32_to_chip1_bg(g_hidden_dev_buf_1, g_hidden_f32.data(), MC::n_embd);
                });
            }

            // On-device rmsnorm + GEMV (traced after caches warm)
            if (g_norm_matmul_traces_valid[layer]) {
                g_mesh->replay_mesh_trace(0, g_norm_matmul_traces[layer], false);
            } else if (g_all_caches_warm) {
                norm_matmul_ops(g_wt.attn_norm_buf[layer], g_wt.ssm_w_combined_buf[ssm_idx],
                                g_combined_rows, MC::n_embd);
                Finish(cq0);
                auto tid = g_mesh->begin_mesh_trace(0);
                norm_matmul_ops(g_wt.attn_norm_buf[layer], g_wt.ssm_w_combined_buf[ssm_idx],
                                g_combined_rows, MC::n_embd);
                g_mesh->end_mesh_trace(0, tid);
                g_norm_matmul_traces[layer] = tid;
                g_norm_matmul_traces_valid[layer] = true;
            } else {
                norm_matmul_ops(g_wt.attn_norm_buf[layer], g_wt.ssm_w_combined_buf[ssm_idx],
                                g_combined_rows, MC::n_embd);
            }

            auto& gb_comb = get_gemv_buf(g_mesh.get(), g_combined_rows, MC::n_embd);
            EnqueueReadMeshBuffer(cq0, gb_comb.out_host_tiled, gb_comb.out_buf, true);

            g_time_gemv_read += std::chrono::duration<double, std::milli>(Clock::now() - t_after_read).count();
            g_time_norm_mm += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

            // Untilize combined result
            read_gemv_to_f32(gb_comb, g_proj.data(), g_combined_rows);

            auto t_host0 = Clock::now();
            float* qkv_raw = g_proj.data();
            float* z_raw = g_proj.data() + MC::ssm_conv_channels;
            float* alpha_raw = z_raw + MC::ssm_d_inner;
            float* beta_raw = alpha_raw + MC::ssm_dt_rank;

            // 2. Conv1d + SiLU (vectorization-friendly: separate conv + state update)
            auto t_conv = Clock::now();
            auto& cs = g_conv_state[ssm_idx];
            float* conv_out = g_conv_out.data();
            // Conv1d: sum = state[0..2] · weights[0..2] + input · weights[3]
            // Layout: cs = [3 rows × 8192 channels], weights = [4 × 8192] (transposed for contiguous loads)
            const float* w = lw.ssm_conv1d_host.data();
            constexpr int C = MC::ssm_conv_channels;
            const float* s0 = cs.data();
            const float* s1 = cs.data() + C;
            const float* s2 = cs.data() + 2 * C;
            // AVX-512 vectorized conv1d + SiLU (transposed weights: w[tap*C + ch])
            for (int ch = 0; ch < C; ch += 16) {
                __m512 vw0 = _mm512_loadu_ps(w + ch);
                __m512 vw1 = _mm512_loadu_ps(w + C + ch);
                __m512 vw2 = _mm512_loadu_ps(w + 2 * C + ch);
                __m512 vw3 = _mm512_loadu_ps(w + 3 * C + ch);

                __m512 vsum = _mm512_mul_ps(_mm512_loadu_ps(s0 + ch), vw0);
                vsum = _mm512_fmadd_ps(_mm512_loadu_ps(s1 + ch), vw1, vsum);
                vsum = _mm512_fmadd_ps(_mm512_loadu_ps(s2 + ch), vw2, vsum);
                vsum = _mm512_fmadd_ps(_mm512_loadu_ps(qkv_raw + ch), vw3, vsum);

                _mm512_storeu_ps(conv_out + ch, fast_silu_avx512(vsum));
            }
            // State update: shift rows and append new input
            memcpy(cs.data(), s1, MC::ssm_conv_channels * sizeof(float));
            memcpy(cs.data() + MC::ssm_conv_channels, s2, MC::ssm_conv_channels * sizeof(float));
            memcpy(cs.data() + 2 * MC::ssm_conv_channels, qkv_raw, MC::ssm_conv_channels * sizeof(float));

            g_time_conv1d += std::chrono::duration<double, std::milli>(Clock::now() - t_conv).count();

            // 3. Split conv output: Q[2048] | K[2048] | V[4096]
            constexpr int num_k_heads = MC::ssm_n_group;
            constexpr int head_k = MC::ssm_d_state;
            constexpr int num_v = ssm_n_v_heads;
            constexpr int head_v = ssm_head_v_dim_c;
            float* conv_q = conv_out;
            float* conv_k = conv_out + num_k_heads * head_k;
            float* conv_v = conv_out + 2 * num_k_heads * head_k;

            // 4. Delta-net recurrence + 5. Gated RMSNorm (parallelized across v-heads)
            auto t_delta = Clock::now();
            auto& state = g_ssm_state[ssm_idx];
            float* ssm_proj_in = g_ssm_proj_in.data();
            constexpr float ssm_scale = 1.0f / 11.3137f;

            // Process a range of v-heads (lambda for parallel execution)
            auto process_vheads = [&](int vh_start, int vh_end) {
                for (int vh = vh_start; vh < vh_end; vh++) {
                    int kh = vh % num_k_heads;
                    alignas(64) float q[head_k], k_vec[head_k], v_vec[head_v];
                    memcpy(q, conv_q + kh * head_k, head_k * sizeof(float));
                    memcpy(k_vec, conv_k + kh * head_k, head_k * sizeof(float));
                    memcpy(v_vec, conv_v + vh * head_v, head_v * sizeof(float));

                    // RMSNorm Q and K using AVX-512
                    __m512 vqn = _mm512_setzero_ps(), vkn = _mm512_setzero_ps();
                    for (int d = 0; d < head_k; d += 16) {
                        __m512 vq = _mm512_load_ps(q + d);
                        __m512 vk = _mm512_load_ps(k_vec + d);
                        vqn = _mm512_fmadd_ps(vq, vq, vqn);
                        vkn = _mm512_fmadd_ps(vk, vk, vkn);
                    }
                    float qn_s = 1.0f / sqrtf(_mm512_reduce_add_ps(vqn) + MC::rms_norm_eps);
                    float kn_s = 1.0f / sqrtf(_mm512_reduce_add_ps(vkn) + MC::rms_norm_eps);
                    __m512 vqn_b = _mm512_set1_ps(qn_s);
                    __m512 vkn_b = _mm512_set1_ps(kn_s);
                    for (int d = 0; d < head_k; d += 16) {
                        _mm512_store_ps(q + d, _mm512_mul_ps(_mm512_load_ps(q + d), vqn_b));
                        _mm512_store_ps(k_vec + d, _mm512_mul_ps(_mm512_load_ps(k_vec + d), vkn_b));
                    }

                    float biased = alpha_raw[vh] + lw.ssm_dt_bias_host[vh];
                    float sp = (biased > 20.0f) ? biased : logf(1.0f + fast_expf(biased));
                    float gate_val = sp * lw.ssm_a_host[vh];
                    float decay = expf(gate_val);
                    float beta_val = fast_sigmoidf(beta_raw[vh]);
                    float* sh = state.data() + vh * head_v * head_k;

                    __m512 vdecay = _mm512_set1_ps(decay);
                    for (int i = 0; i < head_v; i++) {
                        float* row = sh + i * head_k;
                        __m512 vsk = _mm512_setzero_ps();
                        for (int j = 0; j < head_k; j += 16) {
                            __m512 vr = _mm512_mul_ps(_mm512_loadu_ps(row + j), vdecay);
                            _mm512_storeu_ps(row + j, vr);
                            vsk = _mm512_fmadd_ps(vr, _mm512_load_ps(k_vec + j), vsk);
                        }
                        float sk = _mm512_reduce_add_ps(vsk);
                        float dd = beta_val * (v_vec[i] - sk);
                        __m512 vdd = _mm512_set1_ps(dd);
                        __m512 vout_acc = _mm512_setzero_ps();
                        for (int j = 0; j < head_k; j += 16) {
                            __m512 vr = _mm512_loadu_ps(row + j);
                            __m512 vk = _mm512_load_ps(k_vec + j);
                            vr = _mm512_fmadd_ps(vk, vdd, vr);
                            _mm512_storeu_ps(row + j, vr);
                            vout_acc = _mm512_fmadd_ps(vr, _mm512_load_ps(q + j), vout_acc);
                        }
                        ssm_proj_in[vh * head_v + i] = _mm512_reduce_add_ps(vout_acc) * ssm_scale;
                    }

                    // Gated RMSNorm for this v-head (fully vectorized AVX-512)
                    float* vo = ssm_proj_in + vh * head_v;
                    __m512 vssq = _mm512_setzero_ps();
                    for (int d = 0; d < head_v; d += 16) {
                        __m512 vv = _mm512_loadu_ps(vo + d);
                        vssq = _mm512_fmadd_ps(vv, vv, vssq);
                    }
                    float rms = 1.0f / sqrtf(_mm512_reduce_add_ps(vssq) / head_v + MC::rms_norm_eps);
                    __m512 vrms = _mm512_set1_ps(rms);
                    for (int d = 0; d < head_v; d += 16) {
                        __m512 vv = _mm512_loadu_ps(vo + d);
                        __m512 vnorm = _mm512_mul_ps(_mm512_mul_ps(vv, vrms),
                                                      _mm512_loadu_ps(lw.ssm_norm_host.data() + d));
                        __m512 vz = _mm512_loadu_ps(z_raw + vh * head_v + d);
                        _mm512_storeu_ps(vo + d, _mm512_mul_ps(vnorm, fast_silu_avx512(vz)));
                    }
                }
            };

            // Parallelize across 4 threads using persistent pool (main + 3 workers)
            g_worker_pool.parallel_for(num_v, process_vheads);

            g_time_deltanet += std::chrono::duration<double, std::milli>(Clock::now() - t_delta).count();
            g_time_host += std::chrono::duration<double, std::milli>(Clock::now() - t_host0).count();

            // 6. Write residual to device, then outproj+FFN chain
            // (hidden already on device — written before rmsnorm+GEMV trace in TP mode,
            //  or left on device from previous FFN chain in non-TP mode)
            auto t_rw = Clock::now();
            write_f32_to_buf(g_residual_dev_buf, ssm_proj_in, MC::ssm_d_inner);
            if (g_mesh1) {
                g_chip1_writer.wait();  // wait for async hidden write to chip 1
            }
            g_time_reswrite += std::chrono::duration<double, std::milli>(Clock::now() - t_rw).count();

            auto t2 = Clock::now();
            if (g_mesh1) {
                // TP FFN: chip 0 + chip 1 traced independently
                if (g_ffn_chain_traces_valid[layer] && g_ffn_chain_traces_valid_1[layer]) {
                    g_mesh->replay_mesh_trace(0, g_ffn_chain_traces[layer], false);
                    write_f32_to_buf(g_residual_dev_buf_1, ssm_proj_in, MC::ssm_d_inner, g_mesh1.get());
                    g_mesh1->replay_mesh_trace(0, g_ffn_chain_traces_1[layer], false);
                } else if (g_all_caches_warm) {
                    // Capture chip 0 trace
                    outproj_ffn_chain_ops_tp0(g_wt.ssm_out_buf[ssm_idx], MC::n_embd, MC::ssm_d_inner,
                                              g_wt.post_norm_buf[layer],
                                              g_wt.ssm_ffn_gate_up_buf[ssm_idx],
                                              g_wt.ssm_ffn_down_buf[ssm_idx]);
                    Finish(cq0);
                    auto tid0 = g_mesh->begin_mesh_trace(0);
                    outproj_ffn_chain_ops_tp0(g_wt.ssm_out_buf[ssm_idx], MC::n_embd, MC::ssm_d_inner,
                                              g_wt.post_norm_buf[layer],
                                              g_wt.ssm_ffn_gate_up_buf[ssm_idx],
                                              g_wt.ssm_ffn_down_buf[ssm_idx]);
                    g_mesh->end_mesh_trace(0, tid0);
                    g_ffn_chain_traces[layer] = tid0;
                    g_ffn_chain_traces_valid[layer] = true;
                    // Capture chip 1 trace
                    write_f32_to_buf(g_residual_dev_buf_1, ssm_proj_in, MC::ssm_d_inner, g_mesh1.get());
                    outproj_ffn_chain_ops_tp1(g_wt.ssm_out_buf_1[ssm_idx], MC::n_embd, MC::ssm_d_inner,
                                              g_wt.post_norm_buf_1[layer],
                                              g_wt.ssm_ffn_gate_up_buf_1[ssm_idx],
                                              g_wt.ssm_ffn_down_buf_1[ssm_idx]);
                    auto& cq1 = g_mesh1->mesh_command_queue();
                    Finish(cq1);
                    auto tid1 = g_mesh1->begin_mesh_trace(0);
                    outproj_ffn_chain_ops_tp1(g_wt.ssm_out_buf_1[ssm_idx], MC::n_embd, MC::ssm_d_inner,
                                              g_wt.post_norm_buf_1[layer],
                                              g_wt.ssm_ffn_gate_up_buf_1[ssm_idx],
                                              g_wt.ssm_ffn_down_buf_1[ssm_idx]);
                    g_mesh1->end_mesh_trace(0, tid1);
                    g_ffn_chain_traces_1[layer] = tid1;
                    g_ffn_chain_traces_valid_1[layer] = true;
                    // Replay both
                    g_mesh->replay_mesh_trace(0, g_ffn_chain_traces[layer], false);
                    write_f32_to_buf(g_residual_dev_buf_1, ssm_proj_in, MC::ssm_d_inner, g_mesh1.get());
                    g_mesh1->replay_mesh_trace(0, g_ffn_chain_traces_1[layer], false);
                } else {
                    // Warmup only: execute without trace capture
                    outproj_ffn_chain_ops_tp0(g_wt.ssm_out_buf[ssm_idx], MC::n_embd, MC::ssm_d_inner,
                                              g_wt.post_norm_buf[layer],
                                              g_wt.ssm_ffn_gate_up_buf[ssm_idx],
                                              g_wt.ssm_ffn_down_buf[ssm_idx]);
                    write_f32_to_buf(g_residual_dev_buf_1, ssm_proj_in, MC::ssm_d_inner, g_mesh1.get());
                    outproj_ffn_chain_ops_tp1(g_wt.ssm_out_buf_1[ssm_idx], MC::n_embd, MC::ssm_d_inner,
                                              g_wt.post_norm_buf_1[layer],
                                              g_wt.ssm_ffn_gate_up_buf_1[ssm_idx],
                                              g_wt.ssm_ffn_down_buf_1[ssm_idx]);
                }
            } else {
                if (g_ffn_chain_traces_valid[layer]) {
                    g_mesh->replay_mesh_trace(0, g_ffn_chain_traces[layer], false);
                } else if (g_all_caches_warm) {
                    outproj_ffn_chain_ops(g_wt.ssm_out_buf[ssm_idx], MC::n_embd, MC::ssm_d_inner,
                                          g_wt.post_norm_buf[layer],
                                          g_wt.ssm_ffn_gate_up_buf[ssm_idx],
                                          g_wt.ssm_ffn_down_buf[ssm_idx]);
                    Finish(cq0);
                    auto tid = g_mesh->begin_mesh_trace(0);
                    outproj_ffn_chain_ops(g_wt.ssm_out_buf[ssm_idx], MC::n_embd, MC::ssm_d_inner,
                                          g_wt.post_norm_buf[layer],
                                          g_wt.ssm_ffn_gate_up_buf[ssm_idx],
                                          g_wt.ssm_ffn_down_buf[ssm_idx]);
                    g_mesh->end_mesh_trace(0, tid);
                    g_ffn_chain_traces[layer] = tid;
                    g_ffn_chain_traces_valid[layer] = true;
                } else {
                    outproj_ffn_chain_ops(g_wt.ssm_out_buf[ssm_idx], MC::n_embd, MC::ssm_d_inner,
                                          g_wt.post_norm_buf[layer],
                                          g_wt.ssm_ffn_gate_up_buf[ssm_idx],
                                          g_wt.ssm_ffn_down_buf[ssm_idx]);
                }
            }
            g_time_ffn += std::chrono::duration<double, std::milli>(Clock::now() - t2).count();

            ssm_idx++;
        } else {
            // ======== Full Attention Layer ========
            auto& lw = g_model.attn_layers[attn_idx];
            auto& aw = g_attn_small[attn_idx];

            // 1. On-device rmsnorm + QKV GEMV (traced)
            auto t0 = Clock::now();

            // TP reduction: parallel reads from both chips, combine, write back
            if (layer > 0 && g_mesh1) {
                g_chip1_writer.submit([&]{
                    auto& cq1 = g_mesh1->mesh_command_queue();
                    read_device_to_f32_chip1(g_partial_down_buf_1, g_partial_f32.data(), MC::n_embd, cq1);
                });
                read_device_to_f32(g_hidden_dev_buf, g_hidden_f32.data(), MC::n_embd, cq0);
                auto t_dev_done = Clock::now();
                g_time_ffn_device += std::chrono::duration<double, std::milli>(t_dev_done - t0).count();
                g_chip1_writer.wait();
                for (int i = 0; i < MC::n_embd; i++) g_hidden_f32[i] += g_partial_f32[i];
                write_f32_to_buf(g_hidden_dev_buf, g_hidden_f32.data(), MC::n_embd);
                g_time_tp_reduce += std::chrono::duration<double, std::milli>(Clock::now() - t_dev_done).count();
            } else if (layer > 0) {
                Finish(cq0);
            }

            auto t_after_read = Clock::now();
            g_time_ffn_wait += std::chrono::duration<double, std::milli>(t_after_read - t0).count();

            // On-device rmsnorm + QKV GEMV
            constexpr int q_dim = MC::n_head * MC::head_dim * 2;
            constexpr int kv_dim_one = MC::n_head_kv * MC::head_dim;
            float* qkv = g_qkv.data();

            // Start async write of hidden to chip 1 (overlaps with GEMV on chip 0)
            if (g_mesh1) {
                g_chip1_writer.submit([&]{
                    write_f32_to_chip1_bg(g_hidden_dev_buf_1, g_hidden_f32.data(), MC::n_embd);
                });
            }

            // Norm + QKV GEMV (traced after caches warm)
            if (g_norm_matmul_traces_valid[layer]) {
                g_mesh->replay_mesh_trace(0, g_norm_matmul_traces[layer], false);
            } else if (g_all_caches_warm) {
                norm_matmul_ops(g_wt.attn_norm_buf[layer], g_wt.attn_wqkv_buf[attn_idx],
                                g_qkv_rows, MC::n_embd);
                Finish(cq0);
                auto tid = g_mesh->begin_mesh_trace(0);
                norm_matmul_ops(g_wt.attn_norm_buf[layer], g_wt.attn_wqkv_buf[attn_idx],
                                g_qkv_rows, MC::n_embd);
                g_mesh->end_mesh_trace(0, tid);
                g_norm_matmul_traces[layer] = tid;
                g_norm_matmul_traces_valid[layer] = true;
            } else {
                norm_matmul_ops(g_wt.attn_norm_buf[layer], g_wt.attn_wqkv_buf[attn_idx],
                                g_qkv_rows, MC::n_embd);
            }

            auto& gb_qkv = get_gemv_buf(g_mesh.get(), g_qkv_rows, MC::n_embd);
            EnqueueReadMeshBuffer(cq0, gb_qkv.out_host_tiled, gb_qkv.out_buf, true);
            g_time_gemv_read += std::chrono::duration<double, std::milli>(Clock::now() - t_after_read).count();
            g_time_norm_mm += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

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
                alignas(64) float acc[MC::head_dim];
                memset(acc, 0, MC::head_dim * sizeof(float));
                float max_score = -FLT_MAX, sum_exp = 0;
                for (int kp = 0; kp < kv_len; kp++) {
                    float* kh = g_k_cache[attn_idx].data() + (size_t)kp * kv_dim + kv_h * MC::head_dim;
                    // AVX-512 dot product: Q · K
                    __m512 vdot = _mm512_setzero_ps();
                    for (int d = 0; d < MC::head_dim; d += 16)
                        vdot = _mm512_fmadd_ps(_mm512_loadu_ps(qh + d), _mm512_loadu_ps(kh + d), vdot);
                    float score = _mm512_reduce_add_ps(vdot) * MC::attn_scale;
                    float new_max = std::max(max_score, score);
                    float exp_s = expf(score - new_max);
                    float corr = expf(max_score - new_max);
                    sum_exp = sum_exp * corr + exp_s;
                    // AVX-512 weighted accumulation: acc = acc * corr + exp_s * V
                    float* vh = g_v_cache[attn_idx].data() + (size_t)kp * kv_dim + kv_h * MC::head_dim;
                    __m512 vcorr = _mm512_set1_ps(corr);
                    __m512 vexp = _mm512_set1_ps(exp_s);
                    for (int d = 0; d < MC::head_dim; d += 16) {
                        __m512 va = _mm512_load_ps(acc + d);
                        va = _mm512_fmadd_ps(vexp, _mm512_loadu_ps(vh + d), _mm512_mul_ps(va, vcorr));
                        _mm512_store_ps(acc + d, va);
                    }
                    max_score = new_max;
                }
                // Output + sigmoid gating fused
                float* gh = gate_heads + h * MC::head_dim;
                for (int d = 0; d < MC::head_dim; d++)
                    out[d] = (acc[d] / sum_exp) * fast_sigmoidf(gh[d]);
            }
            g_time_attn_host += std::chrono::duration<double, std::milli>(Clock::now() - t_host1).count();
            g_time_host += std::chrono::duration<double, std::milli>(Clock::now() - t_host1).count();

            // 8. Write residual to device, then outproj+FFN chain
            // (hidden already on device — written before rmsnorm+GEMV trace)
            auto t_rw2 = Clock::now();
            write_f32_to_buf(g_residual_dev_buf, attn_out, MC::n_head * MC::head_dim);
            if (g_mesh1) {
                g_chip1_writer.wait();  // wait for async hidden write to chip 1
            }
            g_time_reswrite += std::chrono::duration<double, std::milli>(Clock::now() - t_rw2).count();

            auto t2 = Clock::now();
            if (g_mesh1) {
                if (g_ffn_chain_traces_valid[layer] && g_ffn_chain_traces_valid_1[layer]) {
                    g_mesh->replay_mesh_trace(0, g_ffn_chain_traces[layer], false);
                    write_f32_to_buf(g_residual_dev_buf_1, attn_out, MC::n_head * MC::head_dim, g_mesh1.get());
                    g_mesh1->replay_mesh_trace(0, g_ffn_chain_traces_1[layer], false);
                } else if (g_all_caches_warm) {
                    outproj_ffn_chain_ops_tp0(g_wt.attn_wo_buf[attn_idx], MC::n_embd, MC::n_head * MC::head_dim,
                                              g_wt.post_norm_buf[layer],
                                              g_wt.attn_ffn_gate_up_buf[attn_idx],
                                              g_wt.attn_ffn_down_buf[attn_idx]);
                    Finish(cq0);
                    auto tid0 = g_mesh->begin_mesh_trace(0);
                    outproj_ffn_chain_ops_tp0(g_wt.attn_wo_buf[attn_idx], MC::n_embd, MC::n_head * MC::head_dim,
                                              g_wt.post_norm_buf[layer],
                                              g_wt.attn_ffn_gate_up_buf[attn_idx],
                                              g_wt.attn_ffn_down_buf[attn_idx]);
                    g_mesh->end_mesh_trace(0, tid0);
                    g_ffn_chain_traces[layer] = tid0;
                    g_ffn_chain_traces_valid[layer] = true;
                    write_f32_to_buf(g_residual_dev_buf_1, attn_out, MC::n_head * MC::head_dim, g_mesh1.get());
                    outproj_ffn_chain_ops_tp1(g_wt.attn_wo_buf_1[attn_idx], MC::n_embd, MC::n_head * MC::head_dim,
                                              g_wt.post_norm_buf_1[layer],
                                              g_wt.attn_ffn_gate_up_buf_1[attn_idx],
                                              g_wt.attn_ffn_down_buf_1[attn_idx]);
                    auto& cq1 = g_mesh1->mesh_command_queue();
                    Finish(cq1);
                    auto tid1 = g_mesh1->begin_mesh_trace(0);
                    outproj_ffn_chain_ops_tp1(g_wt.attn_wo_buf_1[attn_idx], MC::n_embd, MC::n_head * MC::head_dim,
                                              g_wt.post_norm_buf_1[layer],
                                              g_wt.attn_ffn_gate_up_buf_1[attn_idx],
                                              g_wt.attn_ffn_down_buf_1[attn_idx]);
                    g_mesh1->end_mesh_trace(0, tid1);
                    g_ffn_chain_traces_1[layer] = tid1;
                    g_ffn_chain_traces_valid_1[layer] = true;
                    g_mesh->replay_mesh_trace(0, g_ffn_chain_traces[layer], false);
                    write_f32_to_buf(g_residual_dev_buf_1, attn_out, MC::n_head * MC::head_dim, g_mesh1.get());
                    g_mesh1->replay_mesh_trace(0, g_ffn_chain_traces_1[layer], false);
                } else {
                    outproj_ffn_chain_ops_tp0(g_wt.attn_wo_buf[attn_idx], MC::n_embd, MC::n_head * MC::head_dim,
                                              g_wt.post_norm_buf[layer],
                                              g_wt.attn_ffn_gate_up_buf[attn_idx],
                                              g_wt.attn_ffn_down_buf[attn_idx]);
                    write_f32_to_buf(g_residual_dev_buf_1, attn_out, MC::n_head * MC::head_dim, g_mesh1.get());
                    outproj_ffn_chain_ops_tp1(g_wt.attn_wo_buf_1[attn_idx], MC::n_embd, MC::n_head * MC::head_dim,
                                              g_wt.post_norm_buf_1[layer],
                                              g_wt.attn_ffn_gate_up_buf_1[attn_idx],
                                              g_wt.attn_ffn_down_buf_1[attn_idx]);
                }
            } else {
                if (g_ffn_chain_traces_valid[layer]) {
                    g_mesh->replay_mesh_trace(0, g_ffn_chain_traces[layer], false);
                } else if (g_all_caches_warm) {
                    outproj_ffn_chain_ops(g_wt.attn_wo_buf[attn_idx], MC::n_embd, MC::n_head * MC::head_dim,
                                          g_wt.post_norm_buf[layer],
                                          g_wt.attn_ffn_gate_up_buf[attn_idx],
                                          g_wt.attn_ffn_down_buf[attn_idx]);
                    Finish(cq0);
                    auto tid = g_mesh->begin_mesh_trace(0);
                    outproj_ffn_chain_ops(g_wt.attn_wo_buf[attn_idx], MC::n_embd, MC::n_head * MC::head_dim,
                                          g_wt.post_norm_buf[layer],
                                          g_wt.attn_ffn_gate_up_buf[attn_idx],
                                          g_wt.attn_ffn_down_buf[attn_idx]);
                    g_mesh->end_mesh_trace(0, tid);
                    g_ffn_chain_traces[layer] = tid;
                    g_ffn_chain_traces_valid[layer] = true;
                } else {
                    outproj_ffn_chain_ops(g_wt.attn_wo_buf[attn_idx], MC::n_embd, MC::n_head * MC::head_dim,
                                          g_wt.post_norm_buf[layer],
                                          g_wt.attn_ffn_gate_up_buf[attn_idx],
                                          g_wt.attn_ffn_down_buf[attn_idx]);
                }
            }
            g_time_ffn += std::chrono::duration<double, std::milli>(Clock::now() - t2).count();

            attn_idx++;
        }
    }

    // Output norm (host) + LM head GEMV (device)
    auto t_lm = Clock::now();

    // Read hidden from device (waits for last FFN chain) + TP reduction
    if (g_mesh1) {
        g_chip1_writer.submit([&]{
            auto& cq1 = g_mesh1->mesh_command_queue();
            read_device_to_f32_chip1(g_partial_down_buf_1, g_partial_f32.data(), MC::n_embd, cq1);
        });
        read_device_to_f32(g_hidden_dev_buf, g_hidden_f32.data(), MC::n_embd, cq0);
        g_chip1_writer.wait();
        for (int i = 0; i < MC::n_embd; i++) g_hidden_f32[i] += g_partial_f32[i];
    } else {
        read_device_to_f32(g_hidden_dev_buf, g_hidden_f32.data(), MC::n_embd, cq0);
    }

    // Host rmsnorm with output norm weights
    rmsnorm(g_hidden_f32.data(), g_output_norm.data(), g_norm_out.data(), MC::n_embd);
    write_f32_to_buf(g_norm_dev_buf, g_norm_out.data(), MC::n_embd);

    float* logits = g_logits.data();

    if (g_mesh1) {
        // 2-chip LM head: split rows across chips for 2x bandwidth
        uint32_t M0 = g_wt.lm_head_Mt0 * TILE_HEIGHT;
        uint32_t M1 = g_wt.lm_head_Mt1 * TILE_HEIGHT;
        auto& gb_lm0 = get_gemv_buf(g_mesh.get(), M0, MC::n_embd);
        auto& gb_lm1 = get_gemv_buf(g_mesh1.get(), M1, MC::n_embd);
        auto& cq1 = g_mesh1->mesh_command_queue();

        // Write norm to chip 1 (chip 0 already has it from above)
        write_f32_to_buf(g_norm_dev_buf_1, g_norm_out.data(), MC::n_embd, g_mesh1.get());

        if (g_lmhead_trace_valid) {
            g_mesh->replay_mesh_trace(0, g_lmhead_trace, false);
            g_mesh1->replay_mesh_trace(0, g_lmhead_trace_1, false);
        } else if (g_all_caches_warm) {
            dispatch_gemv(g_mesh.get(), g_norm_dev_buf, g_wt.lm_head_buf, gb_lm0.out_buf, M0, MC::n_embd);
            dispatch_gemv(g_mesh1.get(), g_norm_dev_buf_1, g_wt.lm_head_buf_1, gb_lm1.out_buf, M1, MC::n_embd);
            Finish(cq0);
            Finish(cq1);
            auto tid0 = g_mesh->begin_mesh_trace(0);
            dispatch_gemv(g_mesh.get(), g_norm_dev_buf, g_wt.lm_head_buf, gb_lm0.out_buf, M0, MC::n_embd);
            g_mesh->end_mesh_trace(0, tid0);
            auto tid1 = g_mesh1->begin_mesh_trace(0);
            dispatch_gemv(g_mesh1.get(), g_norm_dev_buf_1, g_wt.lm_head_buf_1, gb_lm1.out_buf, M1, MC::n_embd);
            g_mesh1->end_mesh_trace(0, tid1);
            g_lmhead_trace = tid0;
            g_lmhead_trace_1 = tid1;
            g_lmhead_trace_valid = true;
        } else {
            dispatch_gemv(g_mesh.get(), g_norm_dev_buf, g_wt.lm_head_buf, gb_lm0.out_buf, M0, MC::n_embd);
            dispatch_gemv(g_mesh1.get(), g_norm_dev_buf_1, g_wt.lm_head_buf_1, gb_lm1.out_buf, M1, MC::n_embd);
        }

        // Read from both chips in parallel (each waits for its chip's GEMV)
        g_chip1_writer.submit([&]{
            EnqueueReadMeshBuffer(cq1, gb_lm1.out_host_tiled, gb_lm1.out_buf, true);
        });
        EnqueueReadMeshBuffer(cq0, gb_lm0.out_host_tiled, gb_lm0.out_buf, true);
        g_chip1_writer.wait();

        // Untilize: chip 0 has first M0 rows, chip 1 has next M1 rows
        read_gemv_to_f32(gb_lm0, logits, M0);
        read_gemv_to_f32(gb_lm1, logits + M0, M1);
    } else {
        // Single-chip LM head (traced after caches warm)
        auto& gb_lm = get_gemv_buf(g_mesh.get(), MC::n_vocab, MC::n_embd);
        if (g_lmhead_trace_valid) {
            g_mesh->replay_mesh_trace(0, g_lmhead_trace, false);
        } else if (g_all_caches_warm) {
            dispatch_gemv(g_mesh.get(), g_norm_dev_buf, g_lm_head_buf, gb_lm.out_buf,
                          MC::n_vocab, MC::n_embd);
            Finish(cq0);
            auto tid = g_mesh->begin_mesh_trace(0);
            dispatch_gemv(g_mesh.get(), g_norm_dev_buf, g_lm_head_buf, gb_lm.out_buf,
                          MC::n_vocab, MC::n_embd);
            g_mesh->end_mesh_trace(0, tid);
            g_lmhead_trace = tid;
            g_lmhead_trace_valid = true;
        } else {
            dispatch_gemv(g_mesh.get(), g_norm_dev_buf, g_lm_head_buf, gb_lm.out_buf,
                          MC::n_vocab, MC::n_embd);
        }
        EnqueueReadMeshBuffer(cq0, gb_lm.out_host_tiled, gb_lm.out_buf, true);
        read_gemv_to_f32(gb_lm, logits, MC::n_vocab);
    }
    g_time_lmhead += std::chrono::duration<double, std::milli>(Clock::now() - t_lm).count();

    if (!g_all_caches_warm) g_all_caches_warm = true;
    g_decode_count++;
    if (g_verbose && g_decode_count % 10 == 0) {
        int dc = g_decode_count;
        printf("  [profile @%d] norm_mm=%.0f outproj=%.0f ffn=%.0f host=%.0f reswr=%.0f lmhead=%.0f ms/tok\n",
               dc, g_time_norm_mm / dc, g_time_outproj / dc, g_time_ffn / dc,
               g_time_host / dc, g_time_reswrite / dc, g_time_lmhead / dc);
        printf("    host_detail: conv1d=%.1f deltanet=%.1f untilize=%.1f attn=%.1f ms/tok\n",
               g_time_conv1d / dc, g_time_deltanet / dc, g_time_untilize / dc, g_time_attn_host / dc);
        printf("    norm_mm_detail: ffn_wait=%.1f rmsnorm_write=%.1f gemv_read=%.1f ms/tok\n",
               g_time_ffn_wait / dc, g_time_rmsnorm_write / dc, g_time_gemv_read / dc);
        printf("    tp_detail: ffn_device=%.1f tp_reduce=%.1f ms/tok\n",
               g_time_ffn_device / dc, g_time_tp_reduce / dc);
    }

    return logits;
}

// ============================================================================
// Public API
// ============================================================================

bool load_model_and_tokenizer(const char* model_path, int max_ctx) {
    if (getenv("QUIET")) g_verbose = false;
    printf("Loading model from %s (max_ctx=%d)...\n", model_path, max_ctx);

    // Open both N300 chips as a 1×2 MeshDevice, then create submeshes
    bool dual_chip = false;
    try {
        g_mesh2 = MeshDevice::create(
            MeshDeviceConfig(MeshShape(1, 2)),
            DEFAULT_L1_SMALL_SIZE,
            DEFAULT_TRACE_REGION_SIZE,
            1,  // num_command_queues
            DispatchCoreConfig{DispatchCoreType::WORKER});
        // Create chip 0 and chip 1 submeshes
        g_mesh = g_mesh2->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
        g_mesh1 = g_mesh2->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 1));
        dual_chip = true;
        printf("Opened 2-chip MeshDevice, chip 0 submesh + chip 1 submesh\n");
    } catch (const std::exception& e) {
        printf("Failed to open 1x2 mesh: %s\nFalling back to single chip.\n", e.what());
        g_mesh = MeshDevice::create_unit_mesh(0);
        g_mesh1 = nullptr;
        g_mesh2 = nullptr;
    }

    auto grid = g_mesh->compute_with_storage_grid_size();
    printf("Chip 0: compute grid %zux%zu (%zu cores)\n", grid.x, grid.y, grid.x * grid.y);
    if (g_mesh1) {
        auto grid1 = g_mesh1->compute_with_storage_grid_size();
        printf("Chip 1: compute grid %zux%zu (%zu cores)\n", grid1.x, grid1.y, grid1.x * grid1.y);
        g_chip1_writer.start();
    }
    g_worker_pool.start();

    // Query DRAM bank topology for sharded GEMV (chip 0)
    g_num_dram_banks = g_mesh->num_dram_channels();
    g_dram_workers = g_mesh->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_1);
    printf("Chip 0 DRAM banks: %u, optimal workers:\n", g_num_dram_banks);
    for (uint32_t b = 0; b < g_num_dram_banks; b++) {
        printf("  bank %u -> core (%zu, %zu)\n", b, g_dram_workers[b].x, g_dram_workers[b].y);
    }
    if (g_mesh1) {
        g_num_dram_banks_1 = g_mesh1->num_dram_channels();
        g_dram_workers_1 = g_mesh1->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_1);
        printf("Chip 1 DRAM banks: %u\n", g_num_dram_banks_1);
    }

    // Enable program cache for faster repeated matmul dispatch
    g_mesh->enable_program_cache();
    if (g_mesh1) g_mesh1->enable_program_cache();

    MeshCommandQueue& cq = g_mesh->mesh_command_queue();
    g_max_ctx = max_ctx;

    if (!g_tokenizer.load(model_path)) {
        fprintf(stderr, "Failed to load tokenizer\n");
        return false;
    }

    // Check for BFP8_B weight cache
    std::string cache_path = std::string(model_path) + ".bfp8cache";
    bool use_cache = check_bfp8_cache(model_path, cache_path);
    if (use_cache) {
        printf("Found valid BFP8_B weight cache: %s\n", cache_path.c_str());
    }

    if (!load_gguf_weights(model_path, g_model, g_mesh.get(), cq, use_cache)) {
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

    // Pack and upload weight matrices to device
    if (use_cache) {
        printf("Loading pre-packed weights from cache...\n");
        create_weight_tensors_from_cache(cache_path);
    } else {
        printf("Creating weight tensor wrappers...\n");
        create_weight_tensors(cache_path);
    }

    // Shared setup: norms, device buffers, GEMV pre-allocation
    setup_post_weight_buffers();

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

    // Truncate prompt to fit context window (keep last tokens, leave room for generation)
    std::vector<int> tokens = prompt_tokens;
    int max_prompt = g_max_ctx - std::min(max_tokens, g_max_ctx / 2);
    if (max_prompt < 1) max_prompt = 1;
    if ((int)tokens.size() > max_prompt) {
        printf("  [truncating prompt from %zu to %d tokens to fit ctx=%d]\n",
               tokens.size(), max_prompt, g_max_ctx);
        tokens.erase(tokens.begin(), tokens.begin() + ((int)tokens.size() - max_prompt));
    }

    // --- Prefix caching: reuse KV/SSM state for matching prompt prefix ---
    int prefix_len = 0;
    {
        int cache_len = (int)g_cached_tokens.size();
        int prompt_len = (int)tokens.size();
        int check_len = std::min(cache_len, prompt_len);
        while (prefix_len < check_len && tokens[prefix_len] == g_cached_tokens[prefix_len])
            prefix_len++;

        if (prefix_len < cache_len) {
            // Cached state diverges from new prompt — full reset required
            reset_state();
            prefix_len = 0;
        }
        if (prefix_len >= prompt_len) {
            // Entire prompt already cached (re-generation of same prompt) — reset
            reset_state();
            prefix_len = 0;
        }
    }
    int new_tokens = (int)tokens.size() - prefix_len;
    if (prefix_len > 0)
        printf("  [prefix cache: reusing %d tokens, prefilling %d new]\n", prefix_len, new_tokens);

    auto t_start = std::chrono::high_resolution_clock::now();

    // Process prompt tokens in batches of PREFILL_BATCH (32) using batched GEMV.
    // Reads weights once per batch instead of once per token.
    {
        int n_prompt = (int)tokens.size();
        int offset = prefix_len;  // skip cached prefix
        const float* logits = nullptr;

        while (offset < n_prompt) {
            int batch = std::min(PREFILL_BATCH, n_prompt - offset);
            if (g_verbose) printf("  [prefill batch %d..%d of %d]\n", offset, offset + batch - 1, n_prompt);

            if (batch >= 2) {
                // Batched prefill: process batch tokens in parallel GEMV
                logits = forward_prefill(tokens.data() + offset, batch);
            } else {
                // Single token: use regular decode path
                int token = tokens[offset];
                const uint16_t* emb = g_model.tok_embd_host.data() + (size_t)token * MC::n_embd;
                uint32_t* hbits = reinterpret_cast<uint32_t*>(g_hidden_f32.data());
                for (int j = 0; j < MC::n_embd; j++)
                    hbits[j] = static_cast<uint32_t>(emb[j]) << 16;
                logits = forward_decode();
                g_pos++;
            }
            offset += batch;
        }

        // Sample first output token from last prefill logits
        if (logits) {
            float max_l = -FLT_MAX;
            for (int v = 0; v < MC::n_vocab; v++) {
                if (logits[v] > max_l) { max_l = logits[v]; next_token = v; }
            }
        }
    }

    // Update cached tokens with the full prompt
    g_cached_tokens.assign(tokens.begin(), tokens.end());

    auto t_prefill = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill - t_start).count();
    printf("  [prefill: %.1f ms for %d new tokens (%.1f ms/tok)]\n",
           prefill_ms, new_tokens, new_tokens > 0 ? prefill_ms / new_tokens : 0.0);

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

        // Forward pass with generated token (raw bf16→f32 bit ops)
        const uint16_t* emb = g_model.tok_embd_host.data() + (size_t)next_token * MC::n_embd;
        uint32_t* hbits = reinterpret_cast<uint32_t*>(g_hidden_f32.data());
        for (int j = 0; j < MC::n_embd; j++)
            hbits[j] = static_cast<uint32_t>(emb[j]) << 16;

        const float* logits = forward_decode();
        g_pos++;
        g_cached_tokens.push_back(next_token);  // track for prefix caching

        auto t1 = std::chrono::high_resolution_clock::now();
        double tok_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (g_verbose) printf("  [decode: %.0f ms]\n", tok_ms);

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
    g_cached_tokens.clear();
}

const Tokenizer& get_tokenizer() {
    return g_tokenizer;
}

void shutdown() {
    if (!g_loaded) return;
    g_loaded = false;

    // Stop background threads
    g_chip1_writer.stop();
    g_worker_pool.stop();

    // Release traces
    for (int i = 0; i < 32; i++) {
        if (g_norm_matmul_traces_valid[i]) {
            g_mesh->release_mesh_trace(g_norm_matmul_traces[i]);
            g_norm_matmul_traces_valid[i] = false;
        }
        if (g_ffn_chain_traces_valid[i]) {
            g_mesh->release_mesh_trace(g_ffn_chain_traces[i]);
            g_ffn_chain_traces_valid[i] = false;
        }
        if (g_ffn_chain_traces_valid_1[i] && g_mesh1) {
            g_mesh1->release_mesh_trace(g_ffn_chain_traces_1[i]);
            g_ffn_chain_traces_valid_1[i] = false;
        }
    }
    if (g_lmhead_trace_valid) {
        g_mesh->release_mesh_trace(g_lmhead_trace);
        if (g_mesh1) g_mesh1->release_mesh_trace(g_lmhead_trace_1);
        g_lmhead_trace_valid = false;
    }

    // Clear cached custom kernel workloads (must happen before device close)
    g_eltwise_cache.clear();
    g_gemv_cache.clear();
    g_gemv_resadd_cache.clear();
    g_gemv_split_cache.clear();
    g_fused_norm_gemv_cache.clear();
    g_rmsnorm_cache.clear();
    g_mc_rmsnorm_cache.clear();
    g_swiglu_cache.clear();
    g_fpu_rmsnorm_cache.clear();

    // Clear pre-allocated GEMV and FFN buffers
    g_ffn_bufs.clear();
    g_gemv_bufs.clear();
    // Clear TP FFN buffers
    g_tp_ffn_0 = TpFfnBuf{};
    g_tp_ffn_1 = TpFfnBuf{};
    g_partial_down_buf_1.reset();
    g_hidden_dev_buf_1.reset();
    g_residual_dev_buf_1.reset();

    // Clear weight MeshBuffers
    for (auto& b : g_wt.attn_wqkv_buf) b.reset();
    for (auto& b : g_wt.attn_ffn_gate_buf) b.reset();
    for (auto& b : g_wt.attn_ffn_up_buf) b.reset();
    for (auto& b : g_wt.attn_ffn_down_buf) b.reset();
    for (auto& b : g_wt.attn_wo_buf) b.reset();
    for (auto& b : g_wt.ssm_w_combined_buf) b.reset();
    for (auto& b : g_wt.ssm_ffn_gate_buf) b.reset();
    for (auto& b : g_wt.ssm_ffn_up_buf) b.reset();
    for (auto& b : g_wt.ssm_ffn_down_buf) b.reset();
    for (auto& b : g_wt.ssm_out_buf) b.reset();
    // Chip 1 weight buffers
    for (auto& b : g_wt.ssm_ffn_gate_buf_1) b.reset();
    for (auto& b : g_wt.ssm_ffn_up_buf_1) b.reset();
    for (auto& b : g_wt.ssm_ffn_down_buf_1) b.reset();
    for (auto& b : g_wt.ssm_out_buf_1) b.reset();
    for (auto& b : g_wt.attn_ffn_gate_buf_1) b.reset();
    for (auto& b : g_wt.attn_ffn_up_buf_1) b.reset();
    for (auto& b : g_wt.attn_ffn_down_buf_1) b.reset();
    for (auto& b : g_wt.attn_wo_buf_1) b.reset();
    for (auto& b : g_wt.post_norm_buf_1) b.reset();
    g_wt.lm_head_buf.reset();
    g_wt.output_norm_buf.reset();
    g_lm_head_buf.reset();
    g_output_norm_buf.reset();
    for (auto& b : g_wt.attn_norm_buf) b.reset();
    for (auto& b : g_wt.post_norm_buf) b.reset();

    g_model.output_norm.reset();
    for (auto& l : g_model.attn_layers) {
        l.attn_norm.reset(); l.attn_q_norm.reset();
        l.attn_k_norm.reset(); l.post_attn_norm.reset();
    }
    for (auto& l : g_model.ssm_layers) {
        l.attn_norm.reset(); l.post_attn_norm.reset();
    }

    // Close submeshes first, then parent mesh
    g_mesh.reset();
    g_mesh1.reset();
    if (g_mesh2) {
        g_mesh2->close();
        g_mesh2.reset();
    }
}
