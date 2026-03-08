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
#include <immintrin.h>

// blockfloat_common not installed in build; use source tree path
#include "/home/ubuntu/tt-metal/tt_metal/impl/data_format/blockfloat_common.hpp"

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using MC = ModelConfig;

// ============================================================================
// Global state
// ============================================================================
static std::shared_ptr<MeshDevice> g_mesh;   // chip 0: all weights (layers + LM head)
static ModelBuffers g_model;
static Tokenizer g_tokenizer;
static bool g_loaded = false;
static int g_max_ctx = 0;
static int g_pos = 0;

// DRAM bank topology for sharded GEMV
static uint32_t g_num_dram_banks = 0;
static std::vector<CoreCoord> g_dram_workers;  // optimal Tensix worker per DRAM bank

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

    // LM head
    std::shared_ptr<MeshBuffer> lm_head_buf;

    // Norm weight buffers (BF16 on device, for custom dispatch_rmsnorm)
    std::shared_ptr<MeshBuffer> attn_norm_buf[32];
    std::shared_ptr<MeshBuffer> post_norm_buf[32];
    std::shared_ptr<MeshBuffer> output_norm_buf;

};
static WeightBuffers g_wt;

// Persistent hidden state on device (avoids PCIe round-trips between layers)
static std::shared_ptr<MeshBuffer> g_hidden_dev_buf;
static std::shared_ptr<MeshBuffer> g_residual_dev_buf;
// Temp buffer for matmul input after rms_norm (on device)
static std::shared_ptr<MeshBuffer> g_norm_dev_buf;

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
    uint32_t nbanks = (g_num_dram_banks > 0) ? g_num_dram_banks : 12;
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

// Cached eltwise binary workload: created once, reused across calls (trace-compatible).
// Key = (op_type, src0_addr, src1_addr, dst_addr, n_tiles)
struct CachedEltwiseWorkload {
    MeshWorkload workload;
    bool valid = false;
};
static std::map<std::tuple<uint32_t,uint32_t,uint32_t,uint32_t,uint32_t>, CachedEltwiseWorkload> g_eltwise_cache;

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
    auto key = std::make_tuple(op_type, (uint32_t)src0_buf->address(),
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
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_ct_args});

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
static std::map<std::tuple<uint32_t,uint32_t,uint32_t>, CachedGemvWorkload> g_gemv_cache;

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
    auto key = std::make_tuple((uint32_t)act_buf->address(),
                               (uint32_t)weight_buf->address(), (uint32_t)out_buf->address());
    auto& cached = g_gemv_cache[key];

    if (!cached.valid) {
        MeshCoordinateRange dev_range(device->shape());

        uint32_t Kt = K / TILE_WIDTH;
        uint32_t Mt = M / TILE_HEIGHT;
        uint32_t num_banks = g_num_dram_banks;
        uint32_t Mt_per_bank = (Mt + num_banks - 1) / num_banks;

        // Build CoreRangeSet from optimal DRAM workers (non-rectangular)
        std::vector<CoreRange> core_ranges;
        for (uint32_t b = 0; b < num_banks; b++) {
            auto& c = g_dram_workers[b];
            core_ranges.push_back(CoreRange(c, c));
        }
        CoreRangeSet all_cores(core_ranges);

        Program program = CreateProgram();

        uint32_t bf16_tile_bytes = TILE_HEIGHT * TILE_WIDTH * sizeof(bfloat16);
        uint32_t weight_tile_bytes = tile_size(weight_format);

        // BLOCK=16 empirically optimal (tested 4, 16, 64, 128)
        uint32_t effective_block = 16;
        uint32_t weight_cb_tiles = effective_block * 2;  // double-buffered for TRID pipelining

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
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_ct_args});

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
            CoreCoord core = g_dram_workers[b];
            uint32_t start_row = b * Mt_per_bank;
            uint32_t mt_this_core = (start_row >= Mt) ? 0 :
                                    std::min(Mt_per_bank, Mt - start_row);

            // Reader: [act_addr, weight_bank_addr, Mt_per_core, bank_id]
            SetRuntimeArgs(program, reader_kid, core,
                           {(uint32_t)act_buf->address(), (uint32_t)weight_buf->address(),
                            mt_this_core, b});

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
static std::map<std::tuple<uint32_t,uint32_t,uint32_t,uint32_t>, CachedGemvResaddWorkload> g_gemv_resadd_cache;

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
    auto key = std::make_tuple((uint32_t)act_buf->address(),
                               (uint32_t)weight_buf->address(),
                               (uint32_t)residual_buf->address(), M);
    auto& cached = g_gemv_resadd_cache[key];

    if (!cached.valid) {
        MeshCoordinateRange dev_range(device->shape());

        uint32_t Kt = K / TILE_WIDTH;
        uint32_t Mt = M / TILE_HEIGHT;
        uint32_t num_banks = g_num_dram_banks;
        uint32_t Mt_per_bank = (Mt + num_banks - 1) / num_banks;

        std::vector<CoreRange> core_ranges;
        for (uint32_t b = 0; b < num_banks; b++) {
            auto& c = g_dram_workers[b];
            core_ranges.push_back(CoreRange(c, c));
        }
        CoreRangeSet all_cores(core_ranges);

        Program program = CreateProgram();

        uint32_t bf16_tile_bytes = TILE_HEIGHT * TILE_WIDTH * sizeof(bfloat16);
        uint32_t weight_tile_bytes = tile_size(weight_format);

        uint32_t effective_block = 16;
        uint32_t weight_cb_tiles = effective_block * 2;

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

        // Reader: same as regular GEMV
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
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_ct_args});

        // Writer: fused resadd writer (reads residual, adds output, writes back)
        std::vector<uint32_t> writer_ct_args = {(uint32_t)CBIndex::c_16};
        TensorAccessorArgs(*residual_buf).append_to(writer_ct_args);

        auto writer_kid = CreateKernel(program,
            kernel_path("dataflow/writer_gemv_resadd.cpp"), all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                               .noc = NOC::RISCV_0_default,
                               .compile_args = writer_ct_args});

        for (uint32_t b = 0; b < num_banks; b++) {
            CoreCoord core = g_dram_workers[b];
            uint32_t start_row = b * Mt_per_bank;
            uint32_t mt_this_core = (start_row >= Mt) ? 0 :
                                    std::min(Mt_per_bank, Mt - start_row);

            SetRuntimeArgs(program, reader_kid, core,
                           {(uint32_t)act_buf->address(), (uint32_t)weight_buf->address(),
                            mt_this_core, b});
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

// Cached GEMV with fused RMSNorm: each reader core independently computes full rmsnorm
// before doing GEMV weight reads. Eliminates host PCIe round-trip for rmsnorm.
struct CachedFusedNormGemvWorkload {
    MeshWorkload workload;
    bool valid = false;
};
static std::map<std::tuple<uint32_t,uint32_t,uint32_t,uint32_t,uint32_t>, CachedFusedNormGemvWorkload> g_fused_norm_gemv_cache;

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
    auto key = std::make_tuple((uint32_t)hidden_buf->address(),
                               (uint32_t)norm_weight_buf->address(),
                               (uint32_t)weight_buf->address(),
                               (uint32_t)out_buf->address(), M);
    auto& cached = g_fused_norm_gemv_cache[key];

    if (!cached.valid) {
        MeshCoordinateRange dev_range(device->shape());

        uint32_t Kt = K / TILE_WIDTH;
        uint32_t Mt = M / TILE_HEIGHT;
        uint32_t num_banks = g_num_dram_banks;
        uint32_t Mt_per_bank = (Mt + num_banks - 1) / num_banks;

        std::vector<CoreRange> core_ranges;
        for (uint32_t b = 0; b < num_banks; b++) {
            auto& c = g_dram_workers[b];
            core_ranges.push_back(CoreRange(c, c));
        }
        CoreRangeSet all_cores(core_ranges);

        Program program = CreateProgram();

        uint32_t bf16_tile_bytes = TILE_HEIGHT * TILE_WIDTH * sizeof(bfloat16);
        uint32_t weight_tile_bytes = tile_size(weight_format);

        uint32_t effective_block = 16;
        uint32_t weight_cb_tiles = effective_block * 2;

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

        auto reader_kid = CreateKernel(program,
            kernel_path("dataflow/reader_gemv_fused_norm.cpp"), all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc = NOC::RISCV_1_default,
                               .compile_args = reader_ct_args});

        // Compute: same as regular GEMV
        std::vector<uint32_t> compute_ct_args = {Kt, effective_block};
        auto compute_kid = CreateKernel(program,
            kernel_path("compute/gemv.cpp"), all_cores,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_ct_args});

        // Writer: standard output writer
        std::vector<uint32_t> writer_ct_args = {(uint32_t)CBIndex::c_16};
        TensorAccessorArgs(*out_buf).append_to(writer_ct_args);

        auto writer_kid = CreateKernel(program,
            kernel_path("dataflow/writer_gemv_multicore.cpp"), all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                               .noc = NOC::RISCV_0_default,
                               .compile_args = writer_ct_args});

        for (uint32_t b = 0; b < num_banks; b++) {
            CoreCoord core = g_dram_workers[b];
            uint32_t start_row = b * Mt_per_bank;
            uint32_t mt_this_core = (start_row >= Mt) ? 0 :
                                    std::min(Mt_per_bank, Mt - start_row);

            // Reader: [hidden_addr, norm_weight_addr, weight_bank_addr, Mt_per_core, bank_id, n_elements]
            SetRuntimeArgs(program, reader_kid, core,
                           {(uint32_t)hidden_buf->address(), (uint32_t)norm_weight_buf->address(),
                            (uint32_t)weight_buf->address(), mt_this_core, b, n_elements});

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

// Cached SwiGLU workload: SiLU(gate) * up
struct CachedSwigluWorkload {
    MeshWorkload workload;
    bool valid = false;
};
static std::map<std::tuple<uint32_t,uint32_t,uint32_t>, CachedSwigluWorkload> g_swiglu_cache;

// Dispatch SwiGLU (multi-core): out = SiLU(gate) * up
// Uses 12 DRAM worker cores, each handling a slice of tiles.
static void dispatch_swiglu(MeshDevice* device,
                             std::shared_ptr<MeshBuffer> gate_buf,
                             std::shared_ptr<MeshBuffer> up_buf,
                             std::shared_ptr<MeshBuffer> out_buf,
                             uint32_t n_tiles) {
    auto& cq = device->mesh_command_queue();
    auto key = std::make_tuple((uint32_t)gate_buf->address(),
                               (uint32_t)up_buf->address(), (uint32_t)out_buf->address());
    auto& cached = g_swiglu_cache[key];

    if (!cached.valid) {
        MeshCoordinateRange dev_range(device->shape());
        Program program = CreateProgram();

        uint32_t num_banks = g_num_dram_banks;
        uint32_t tiles_per_bank = (n_tiles + num_banks - 1) / num_banks;

        std::vector<CoreRange> core_ranges;
        for (uint32_t b = 0; b < num_banks; b++) {
            auto& c = g_dram_workers[b];
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
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4});

        // Writer: writes output tiles with start offset
        std::vector<uint32_t> writer_ct_args = {(uint32_t)CBIndex::c_16};
        TensorAccessorArgs(*out_buf).append_to(writer_ct_args);

        auto writer_kid = CreateKernel(program,
            kernel_path("dataflow/writer_gemv_multicore.cpp"), all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                               .noc = NOC::RISCV_0_default,
                               .compile_args = writer_ct_args});

        for (uint32_t b = 0; b < num_banks; b++) {
            CoreCoord core = g_dram_workers[b];
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
static std::map<std::tuple<uint32_t,uint32_t,uint32_t>, CachedRmsnormWorkload> g_rmsnorm_cache;

// Dispatch custom RMSNorm: out = rms_norm(input) * weight
// All buffers are [1, n_embd_padded] BF16 on device.
// MeshWorkload is cached and reused — trace-compatible after first warmup call.
static void dispatch_rmsnorm(MeshDevice* device,
                              std::shared_ptr<MeshBuffer> in_buf,
                              std::shared_ptr<MeshBuffer> weight_buf,
                              std::shared_ptr<MeshBuffer> out_buf,
                              uint32_t n_elements, uint32_t n_tiles) {
    auto& cq = device->mesh_command_queue();
    auto key = std::make_tuple((uint32_t)in_buf->address(),
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
static std::map<std::tuple<uint32_t,uint32_t,uint32_t>, CachedMulticoreRmsnormWorkload> g_mc_rmsnorm_cache;

static void dispatch_rmsnorm_multicore(MeshDevice* device,
                                        std::shared_ptr<MeshBuffer> in_buf,
                                        std::shared_ptr<MeshBuffer> weight_buf,
                                        std::shared_ptr<MeshBuffer> out_buf,
                                        uint32_t n_elements, uint32_t n_tiles) {
    auto& cq = device->mesh_command_queue();
    auto key = std::make_tuple((uint32_t)in_buf->address(),
                               (uint32_t)weight_buf->address(), (uint32_t)out_buf->address());
    auto& cached = g_mc_rmsnorm_cache[key];

    if (!cached.valid) {
        MeshCoordinateRange dev_range(device->shape());
        Program program = CreateProgram();
        uint32_t num_banks = g_num_dram_banks;

        // Build CoreRangeSet from DRAM workers
        std::vector<CoreRange> core_ranges;
        for (uint32_t b = 0; b < num_banks; b++) {
            auto& c = g_dram_workers[b];
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
        auto* dev0 = device->get_device(MeshCoordinate(0, 0));
        std::vector<CoreCoord> noc_coords(num_banks);
        for (uint32_t b = 0; b < num_banks; b++) {
            noc_coords[b] = dev0->worker_core_from_logical_core(g_dram_workers[b]);
        }

        // Use a small L1 region for scratch (sem addresses + data)
        // We'll use the memory right after sem1 for scratch data
        // Actually, use a fixed offset in the semaphore region
        // The scratch addr needs to be the same on all cores and known at dispatch time.
        // We'll use sem1_addr + 16 (L1 aligned) as scratch base.
        // On Wormhole, semaphore base is at a fixed L1 address.
        // We need at least num_banks * 4 bytes for partials on core 0.
        // Use a CB (c_2) for scratch space instead.

        // CB c_2: scratch (1 tile = 1024 bytes, plenty for 12 floats + norm_factor)
        CircularBufferConfig cb_scratch_cfg =
            CircularBufferConfig(tile_bytes, {{CBIndex::c_2, tt::DataFormat::Float16_b}})
                .set_page_size(CBIndex::c_2, tile_bytes);
        auto cb_scratch = CreateCircularBuffer(program, all_cores, cb_scratch_cfg);

        // Set per-core runtime args
        for (uint32_t b = 0; b < num_banks; b++) {
            CoreCoord core = g_dram_workers[b];

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

// Run norm on host + matmul on device.
// Reads hidden from device, does fast AVX-512 rmsnorm on host, writes normalized
// result to device, dispatches GEMV.
static void norm_matmul_ops(std::shared_ptr<MeshBuffer> norm_weight_buf,
                            std::shared_ptr<MeshBuffer> weight_buf,
                            uint32_t M, uint32_t K) {
    auto& gb = get_gemv_buf(g_mesh.get(), M, K);
    dispatch_gemv(g_mesh.get(), g_norm_dev_buf, weight_buf, gb.out_buf, M, K);
}

// Run output norm + LM head matmul on device (custom kernels, no host round-trip)
static std::shared_ptr<MeshBuffer> g_output_norm_buf;
static std::shared_ptr<MeshBuffer> g_lm_head_buf;
static void norm_lmhead_ops() {
    constexpr uint32_t embd_tiles = MC::n_embd / TILE_WIDTH;
    dispatch_rmsnorm(g_mesh.get(), g_hidden_dev_buf, g_output_norm_buf,
                     g_norm_dev_buf, MC::n_embd, embd_tiles);
    auto& gb = get_gemv_buf(g_mesh.get(), MC::n_vocab, MC::n_embd);
    dispatch_gemv(g_mesh.get(), g_norm_dev_buf, g_lm_head_buf, gb.out_buf, MC::n_vocab, MC::n_embd);
}
static MeshTraceId g_lmhead_trace;
static bool g_lmhead_trace_valid = false;

// Run outproj matmul + residual add + norm + FFN chain + residual add on device.
// All using custom kernels (dispatch_gemv, dispatch_rmsnorm, dispatch_swiglu, dispatch_eltwise).
static void outproj_ffn_chain_ops(std::shared_ptr<MeshBuffer> outproj_weight_buf,
                                   uint32_t outproj_M, uint32_t outproj_K,
                                   std::shared_ptr<MeshBuffer> norm_weight_buf,
                                   std::shared_ptr<MeshBuffer> gate_weight_buf,
                                   std::shared_ptr<MeshBuffer> up_weight_buf,
                                   std::shared_ptr<MeshBuffer> down_weight_buf) {
    // 1. Output projection + residual add (fused): hidden += residual @ outproj_weight^T
    dispatch_gemv_resadd(g_mesh.get(), g_residual_dev_buf, outproj_weight_buf,
                         g_hidden_dev_buf, outproj_M, outproj_K);

    // 2. RMSNorm (multi-core: 12 cores each handle local bank's tiles)
    constexpr uint32_t embd_tiles = MC::n_embd / TILE_WIDTH;
    dispatch_rmsnorm_multicore(g_mesh.get(), g_hidden_dev_buf, norm_weight_buf,
                                g_norm_dev_buf, MC::n_embd, embd_tiles);

    // 3. FFN: gate + up matmuls → SwiGLU (SiLU(gate)*up) → down matmul + residual add
    auto& fb = get_ffn_buf(g_mesh.get());

    // Gate projection
    dispatch_gemv(g_mesh.get(), g_norm_dev_buf, gate_weight_buf, fb.gate_buf,
                  MC::n_ff, MC::n_embd);

    // Up projection
    dispatch_gemv(g_mesh.get(), g_norm_dev_buf, up_weight_buf, fb.up_buf,
                  MC::n_ff, MC::n_embd);

    // SwiGLU: SiLU(gate) * up
    constexpr uint32_t ff_tiles = MC::n_ff / TILE_WIDTH;
    dispatch_swiglu(g_mesh.get(), fb.gate_buf, fb.up_buf, fb.act_buf, ff_tiles);

    // Down projection + residual add (fused): hidden += act @ down_weight^T
    dispatch_gemv_resadd(g_mesh.get(), fb.act_buf, down_weight_buf,
                         g_hidden_dev_buf, MC::n_embd, MC::n_ff);
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

// Pack bf16 data as BFP8_B tiles (CPU-side, can be called from worker threads)
static std::vector<uint32_t> pack_bf16_as_bfp8b(const uint16_t* bf16_data, uint32_t M, uint32_t K) {
    size_t n = (size_t)M * K;
    const bfloat16* src = reinterpret_cast<const bfloat16*>(bf16_data);
    ttsl::Span<const bfloat16> span(src, n);
    return pack_as_bfp_tiles<tt::DataFormat::Bfp8_b>(span, /*row_major_input=*/true, /*is_exp_a=*/false);
}

// Upload packed BFP8_B data to device, return MeshBuffer directly
static std::shared_ptr<MeshBuffer> upload_packed_bfp8b_buf(MeshDevice* device,
                                                            const std::vector<uint32_t>& packed,
                                                            uint32_t M, uint32_t K) {
    constexpr uint32_t TH = TILE_HEIGHT, TW = TILE_WIDTH;
    uint32_t Mt = M / TH, Kt = K / TW;
    constexpr uint32_t bfp8_tile_bytes = BFLOAT8_B_TILE_HW;  // 1088

    // Create DRAM-sharded buffer: each bank stores Mt_per_bank * Kt contiguous tiles
    uint32_t num_banks = g_num_dram_banks;
    uint32_t Mt_per_bank = (Mt + num_banks - 1) / num_banks;
    uint32_t Mt_padded = Mt_per_bank * num_banks;
    uint32_t total_pages = Mt_padded * Kt;
    uint32_t pages_per_bank = Mt_per_bank * Kt;
    uint32_t total_bytes = total_pages * bfp8_tile_bytes;

    // Pad packed data with zero tiles if Mt not divisible by num_banks
    std::vector<uint32_t> padded_packed;
    uint32_t words_per_tile = bfp8_tile_bytes / sizeof(uint32_t);  // 272
    uint32_t needed_words = total_pages * words_per_tile;
    if (packed.size() < needed_words) {
        padded_packed = packed;
        padded_packed.resize(needed_words, 0);
    }
    const auto& write_data = (packed.size() >= needed_words) ? packed : padded_packed;

    // DRAM bank core coordinates: 1D range (x=bank_id, y=0)
    CoreRange dram_bank_range({0, 0}, {num_banks - 1, 0});
    auto dram_cores = corerange_to_cores(CoreRangeSet(dram_bank_range));

    BufferDistributionSpec shard_spec(
        tt::tt_metal::Shape({1, total_pages}),
        tt::tt_metal::Shape({1, pages_per_bank}),
        dram_cores);

    DeviceLocalBufferConfig dram_cfg{
        .page_size = bfp8_tile_bytes,
        .buffer_type = BufferType::DRAM,
        .sharding_args = BufferShardingArgs(shard_spec)};

    // Use {1, total_pages} shape so bytes_per_datum = 1088 exactly (avoids BFP8_B rounding)
    Shape2D global_shape = {1, total_pages};
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

// Create weight buffers — pack weights as BFP8_B (multi-threaded) then upload.
// Each weight's host bf16 is freed immediately after packing to minimize peak memory.
static void create_weight_tensors() {
    printf("Packing and uploading weights as BFLOAT8_B (multi-threaded)...\n");
    auto t0 = std::chrono::steady_clock::now();

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

            // Upload packed data sequentially (device I/O)
            g_wt.ssm_w_combined_buf[ssm_idx] = upload_packed_bfp8b_buf(g_mesh.get(), p_combined, combined_rows, MC::n_embd);
            g_wt.ssm_ffn_gate_buf[ssm_idx] = upload_packed_bfp8b_buf(g_mesh.get(), p_gate, MC::n_ff, MC::n_embd);
            g_wt.ssm_ffn_up_buf[ssm_idx] = upload_packed_bfp8b_buf(g_mesh.get(), p_up, MC::n_ff, MC::n_embd);
            g_wt.ssm_ffn_down_buf[ssm_idx] = upload_packed_bfp8b_buf(g_mesh.get(), p_down, MC::n_embd, MC::n_ff);
            g_wt.ssm_out_buf[ssm_idx] = upload_packed_bfp8b_buf(g_mesh.get(), p_out, MC::n_embd, MC::ssm_d_inner);

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

            // Upload packed data sequentially
            g_wt.attn_wqkv_buf[attn_idx] = upload_packed_bfp8b_buf(g_mesh.get(), p_qkv, qkv_rows, MC::n_embd);
            g_wt.attn_ffn_gate_buf[attn_idx] = upload_packed_bfp8b_buf(g_mesh.get(), p_gate, MC::n_ff, MC::n_embd);
            g_wt.attn_ffn_up_buf[attn_idx] = upload_packed_bfp8b_buf(g_mesh.get(), p_up, MC::n_ff, MC::n_embd);
            g_wt.attn_ffn_down_buf[attn_idx] = upload_packed_bfp8b_buf(g_mesh.get(), p_down, MC::n_embd, MC::n_ff);
            g_wt.attn_wo_buf[attn_idx] = upload_packed_bfp8b_buf(g_mesh.get(), p_wo, MC::n_embd, MC::n_head * MC::head_dim);

            printf("  Attn layer %d uploaded\n", attn_idx);
            attn_idx++;
        }
    }

    auto t1_time = std::chrono::steady_clock::now();
    double layer_sec = std::chrono::duration<double>(t1_time - t0).count();
    printf("Uploaded %d attention + %d SSM weight tensors as BFLOAT8_B (%.1fs).\n",
           attn_idx, ssm_idx, layer_sec);

    // LM head on chip 0 (fused with output norm in traced chain)
    auto p_lm = pack_bf16_as_bfp8b(g_model.output_host.data(), MC::n_vocab, MC::n_embd);
    { std::vector<uint16_t>().swap(g_model.output_host); }
    g_wt.lm_head_buf = upload_packed_bfp8b_buf(g_mesh.get(), p_lm, MC::n_vocab, MC::n_embd);
    g_lm_head_buf = g_wt.lm_head_buf;
    printf("  lm_head uploaded\n");

    // Assign norm weight buffers for on-device RMSNorm
    printf("Setting up norm weight buffers...\n");
    attn_idx = 0; ssm_idx = 0;
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
    get_gemv_buf(g_mesh.get(), MC::n_ff, MC::n_embd);        // Gate/up projections
    get_gemv_buf(g_mesh.get(), MC::n_embd, MC::n_ff);        // Down projection
    get_gemv_buf(g_mesh.get(), MC::n_vocab, MC::n_embd);     // LM head
    get_ffn_buf(g_mesh.get());                                // FFN intermediates
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
// Large matmuls on device via custom GEMV kernels, small ops on host CPU.
// Returns pointer to static g_logits buffer (valid until next call).
// ============================================================================
static int g_decode_count = 0;
static double g_time_gemv = 0, g_time_ffn = 0, g_time_ssm = 0, g_time_attn = 0, g_time_lmhead = 0;
static double g_time_outproj = 0, g_time_reswrite = 0, g_time_host = 0, g_time_norm_mm = 0;
static double g_time_conv1d = 0, g_time_deltanet = 0, g_time_untilize = 0, g_time_attn_host = 0;

static float* forward_decode() {
    using Clock = std::chrono::high_resolution_clock;
    int pos = g_pos;
    auto& cq0 = g_mesh->mesh_command_queue();

    // Write hidden state to device (chip 0) — stays on device through all layers
    write_hidden_to_device(g_hidden_f32.data());

    // One-time micro-benchmark: measure per-program overhead at decode 5
    if (g_decode_count == 5) {
        Finish(cq0);  // drain everything
        auto& gb_test = get_gemv_buf(g_mesh.get(), g_combined_rows, MC::n_embd);

        // Warmup: ensure cached
        dispatch_rmsnorm(g_mesh.get(), g_hidden_dev_buf, g_wt.attn_norm_buf[0],
                         g_norm_dev_buf, MC::n_embd, MC::n_embd / TILE_WIDTH);
        dispatch_gemv(g_mesh.get(), g_norm_dev_buf, g_wt.ssm_w_combined_buf[0],
                      gb_test.out_buf, g_combined_rows, MC::n_embd);
        Finish(cq0);

        // Test 1: Time 10x single GEMV dispatch+Finish
        auto ta = Clock::now();
        for (int r = 0; r < 10; r++) {
            dispatch_gemv(g_mesh.get(), g_norm_dev_buf, g_wt.ssm_w_combined_buf[0],
                          gb_test.out_buf, g_combined_rows, MC::n_embd);
            Finish(cq0);
        }
        auto tb = Clock::now();
        double t_single = std::chrono::duration<double, std::milli>(tb - ta).count() / 10.0;

        // Test 2: Time 10x Finish-only (already synced)
        auto tc = Clock::now();
        for (int r = 0; r < 10; r++) {
            Finish(cq0);
        }
        auto td = Clock::now();
        double t_finish = std::chrono::duration<double, std::milli>(td - tc).count() / 10.0;

        // Test 3: Time 10x (fused norm+GEMV + Finish)
        auto te = Clock::now();
        for (int r = 0; r < 10; r++) {
            dispatch_gemv_fused_norm(g_mesh.get(), g_hidden_dev_buf, g_wt.attn_norm_buf[0],
                                     g_wt.ssm_w_combined_buf[0], gb_test.out_buf,
                                     g_combined_rows, MC::n_embd, MC::n_embd);
            Finish(cq0);
        }
        auto tf = Clock::now();
        double t_norm_gemv = std::chrono::duration<double, std::milli>(tf - te).count() / 10.0;

        // Test 4: Time 10x full FFN chain dispatch+Finish
        // First ensure FFN chain trace is warm for layer 0
        write_f32_to_buf(g_residual_dev_buf, g_hidden_f32.data(), MC::ssm_d_inner);
        if (!g_ffn_chain_traces_valid[0]) {
            outproj_ffn_chain_ops(g_wt.ssm_out_buf[0], MC::n_embd, MC::ssm_d_inner,
                                  g_wt.post_norm_buf[0],
                                  g_wt.ssm_ffn_gate_buf[0], g_wt.ssm_ffn_up_buf[0],
                                  g_wt.ssm_ffn_down_buf[0]);
            Finish(cq0);
            auto tid = g_mesh->begin_mesh_trace(0);
            outproj_ffn_chain_ops(g_wt.ssm_out_buf[0], MC::n_embd, MC::ssm_d_inner,
                                  g_wt.post_norm_buf[0],
                                  g_wt.ssm_ffn_gate_buf[0], g_wt.ssm_ffn_up_buf[0],
                                  g_wt.ssm_ffn_down_buf[0]);
            g_mesh->end_mesh_trace(0, tid);
            g_ffn_chain_traces[0] = tid;
            g_ffn_chain_traces_valid[0] = true;
        }
        Finish(cq0);

        auto tg = Clock::now();
        for (int r = 0; r < 10; r++) {
            g_mesh->replay_mesh_trace(0, g_ffn_chain_traces[0], false);
            Finish(cq0);
        }
        auto th = Clock::now();
        double t_ffn_trace = std::chrono::duration<double, std::milli>(th - tg).count() / 10.0;

        // Test 5: Time 10x pipelined GEMV (no Finish between)
        auto ti = Clock::now();
        for (int r = 0; r < 10; r++) {
            dispatch_gemv(g_mesh.get(), g_norm_dev_buf, g_wt.ssm_w_combined_buf[0],
                          gb_test.out_buf, g_combined_rows, MC::n_embd);
        }
        Finish(cq0);
        auto tj = Clock::now();
        double t_pipelined = std::chrono::duration<double, std::milli>(tj - ti).count() / 10.0;

        // Test 6: rmsnorm alone + Finish
        Finish(cq0);
        auto t6a = Clock::now();
        for (int r = 0; r < 10; r++) {
            dispatch_rmsnorm(g_mesh.get(), g_hidden_dev_buf, g_wt.attn_norm_buf[0],
                             g_norm_dev_buf, MC::n_embd, MC::n_embd / TILE_WIDTH);
            Finish(cq0);
        }
        auto t6b = Clock::now();
        double t_norm_only = std::chrono::duration<double, std::milli>(t6b - t6a).count() / 10.0;

        // Test 7: Two different GEMVs (different weight buffers) + Finish
        Finish(cq0);
        auto t7a = Clock::now();
        for (int r = 0; r < 10; r++) {
            dispatch_gemv(g_mesh.get(), g_norm_dev_buf, g_wt.ssm_w_combined_buf[0],
                          gb_test.out_buf, g_combined_rows, MC::n_embd);
            dispatch_gemv(g_mesh.get(), g_norm_dev_buf, g_wt.ssm_w_combined_buf[1],
                          gb_test.out_buf, g_combined_rows, MC::n_embd);
            Finish(cq0);
        }
        auto t7b = Clock::now();
        double t_two_diff_gemv = std::chrono::duration<double, std::milli>(t7b - t7a).count() / 10.0;

        // Test 8: Two same GEMVs + Finish (for comparison)
        Finish(cq0);
        auto t8a = Clock::now();
        for (int r = 0; r < 10; r++) {
            dispatch_gemv(g_mesh.get(), g_norm_dev_buf, g_wt.ssm_w_combined_buf[0],
                          gb_test.out_buf, g_combined_rows, MC::n_embd);
            dispatch_gemv(g_mesh.get(), g_norm_dev_buf, g_wt.ssm_w_combined_buf[0],
                          gb_test.out_buf, g_combined_rows, MC::n_embd);
            Finish(cq0);
        }
        auto t8b = Clock::now();
        double t_two_same_gemv = std::chrono::duration<double, std::milli>(t8b - t8a).count() / 10.0;

        // Test 9: eltwise add alone + Finish
        Finish(cq0);
        constexpr uint32_t embd_tiles_test = MC::n_embd / TILE_WIDTH;
        auto t9a = Clock::now();
        for (int r = 0; r < 10; r++) {
            dispatch_eltwise_binary(g_mesh.get(), 0, g_hidden_dev_buf, gb_test.out_buf,
                                    g_hidden_dev_buf, embd_tiles_test);
            Finish(cq0);
        }
        auto t9b = Clock::now();
        double t_eltwise = std::chrono::duration<double, std::milli>(t9b - t9a).count() / 10.0;

        // Test 10: GEMV + eltwise + Finish
        Finish(cq0);
        auto t10a = Clock::now();
        for (int r = 0; r < 10; r++) {
            dispatch_gemv(g_mesh.get(), g_norm_dev_buf, g_wt.ssm_w_combined_buf[0],
                          gb_test.out_buf, g_combined_rows, MC::n_embd);
            dispatch_eltwise_binary(g_mesh.get(), 0, g_hidden_dev_buf, gb_test.out_buf,
                                    g_hidden_dev_buf, embd_tiles_test);
            Finish(cq0);
        }
        auto t10b = Clock::now();
        double t_gemv_eltwise = std::chrono::duration<double, std::milli>(t10b - t10a).count() / 10.0;

        printf("=== Per-program overhead benchmark ===\n");
        printf("  Single GEMV + Finish:     %.3f ms\n", t_single);
        printf("  Finish-only (no-op):      %.3f ms\n", t_finish);
        printf("  norm+GEMV + Finish:       %.3f ms\n", t_norm_gemv);
        printf("  FFN chain trace + Finish: %.3f ms\n", t_ffn_trace);
        printf("  Pipelined GEMV (10x):     %.3f ms/each\n", t_pipelined);
        printf("  rmsnorm alone + Finish:   %.3f ms\n", t_norm_only);
        printf("  2x diff GEMV + Finish:    %.3f ms\n", t_two_diff_gemv);
        printf("  2x same GEMV + Finish:    %.3f ms\n", t_two_same_gemv);
        printf("  eltwise add + Finish:     %.3f ms\n", t_eltwise);
        printf("  GEMV + eltwise + Finish:  %.3f ms\n", t_gemv_eltwise);
        printf("  GEMV data: %.1f MB\n",
               (double)(g_combined_rows * MC::n_embd) * 1.0625 / 1e6);
        fflush(stdout);
    }

    int attn_idx = 0, ssm_idx = 0;
    for (int layer = 0; layer < MC::n_layers; layer++) {
        if (MC::is_recurrent(layer)) {
            // ======== SSM (Delta-Net) Layer ========
            auto& lw = g_model.ssm_layers[ssm_idx];

            // 1. Host rmsnorm + combined GEMV on device
            auto& gb_comb = get_gemv_buf(g_mesh.get(), g_combined_rows, MC::n_embd);
            auto t0 = Clock::now();

            // Read hidden from device (blocking read waits for prev FFN chain)
            if (layer > 0)
                read_device_to_f32(g_hidden_dev_buf, g_hidden_f32.data(), MC::n_embd, cq0);

            // Host rmsnorm (AVX-512, ~0.03ms — faster than device software float)
            rmsnorm(g_hidden_f32.data(), g_layer_norms[layer].attn_norm.data(),
                    g_norm_out.data(), MC::n_embd);
            write_f32_to_buf(g_norm_dev_buf, g_norm_out.data(), MC::n_embd);

            // GEMV only
            dispatch_gemv(g_mesh.get(), g_norm_dev_buf, g_wt.ssm_w_combined_buf[ssm_idx],
                          gb_comb.out_buf, g_combined_rows, MC::n_embd);
            EnqueueReadMeshBuffer(cq0, gb_comb.out_host_tiled, gb_comb.out_buf, true);

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
            // Layout: cs = [3 rows × 8192 channels], weights = [8192 × 4] row-major
            const float* w = lw.ssm_conv1d_host.data();
            const float* s0 = cs.data();
            const float* s1 = cs.data() + MC::ssm_conv_channels;
            const float* s2 = cs.data() + 2 * MC::ssm_conv_channels;
            for (int ch = 0; ch < MC::ssm_conv_channels; ch++) {
                float sum = s0[ch] * w[ch * 4 + 0]
                          + s1[ch] * w[ch * 4 + 1]
                          + s2[ch] * w[ch * 4 + 2]
                          + qkv_raw[ch] * w[ch * 4 + 3];
                conv_out[ch] = fast_siluf(sum);
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

                    // Gated RMSNorm for this v-head (AVX-512)
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
                        float tmp[16];
                        _mm512_storeu_ps(tmp, vnorm);
                        for (int t = 0; t < 16 && d + t < head_v; t++) {
                            float z = z_raw[vh * head_v + d + t];
                            tmp[t] *= fast_siluf(z);
                        }
                        _mm512_storeu_ps(vo + d, _mm512_loadu_ps(tmp));
                    }
                }
            };

            // Parallelize across 2 threads (main + 1 worker)
            constexpr int mid_vh = num_v / 2;
            std::thread worker(process_vheads, 0, mid_vh);
            process_vheads(mid_vh, num_v);
            worker.join();

            g_time_deltanet += std::chrono::duration<double, std::milli>(Clock::now() - t_delta).count();
            g_time_host += std::chrono::duration<double, std::milli>(Clock::now() - t_host0).count();

            // 6. Write ssm_proj_in to g_residual_dev, then outproj+FFN chain ON DEVICE
            auto t_rw = Clock::now();
            write_f32_to_buf(g_residual_dev_buf, ssm_proj_in, MC::ssm_d_inner);
            g_time_reswrite += std::chrono::duration<double, std::milli>(Clock::now() - t_rw).count();

            auto t2 = Clock::now();
            if (!g_ffn_chain_traces_valid[layer]) {
                outproj_ffn_chain_ops(g_wt.ssm_out_buf[ssm_idx], MC::n_embd, MC::ssm_d_inner,
                                      g_wt.post_norm_buf[layer],
                                      g_wt.ssm_ffn_gate_buf[ssm_idx], g_wt.ssm_ffn_up_buf[ssm_idx],
                                      g_wt.ssm_ffn_down_buf[ssm_idx]);
                Finish(cq0);
                auto tid = g_mesh->begin_mesh_trace(0);
                outproj_ffn_chain_ops(g_wt.ssm_out_buf[ssm_idx], MC::n_embd, MC::ssm_d_inner,
                                      g_wt.post_norm_buf[layer],
                                      g_wt.ssm_ffn_gate_buf[ssm_idx], g_wt.ssm_ffn_up_buf[ssm_idx],
                                      g_wt.ssm_ffn_down_buf[ssm_idx]);
                g_mesh->end_mesh_trace(0, tid);
                g_ffn_chain_traces[layer] = tid;
                g_ffn_chain_traces_valid[layer] = true;
            } else {
                g_mesh->replay_mesh_trace(0, g_ffn_chain_traces[layer], false);
            }
            g_time_ffn += std::chrono::duration<double, std::milli>(Clock::now() - t2).count();

            ssm_idx++;
        } else {
            // ======== Full Attention Layer ========
            auto& lw = g_model.attn_layers[attn_idx];
            auto& aw = g_attn_small[attn_idx];

            // 1. Host rmsnorm + QKV GEMV on device
            auto& gb_qkv = get_gemv_buf(g_mesh.get(), g_qkv_rows, MC::n_embd);
            auto t0 = Clock::now();

            // Read hidden from device (blocking read waits for prev FFN chain)
            if (layer > 0)
                read_device_to_f32(g_hidden_dev_buf, g_hidden_f32.data(), MC::n_embd, cq0);

            // Host rmsnorm (AVX-512, faster than device software float)
            rmsnorm(g_hidden_f32.data(), g_layer_norms[layer].attn_norm.data(),
                    g_norm_out.data(), MC::n_embd);
            write_f32_to_buf(g_norm_dev_buf, g_norm_out.data(), MC::n_embd);

            // QKV GEMV
            dispatch_gemv(g_mesh.get(), g_norm_dev_buf, g_wt.attn_wqkv_buf[attn_idx],
                          gb_qkv.out_buf, g_qkv_rows, MC::n_embd);
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
                    out[d] = (acc[d] / sum_exp) * fast_sigmoidf(gh[d]);
            }
            g_time_attn_host += std::chrono::duration<double, std::milli>(Clock::now() - t_host1).count();
            g_time_host += std::chrono::duration<double, std::milli>(Clock::now() - t_host1).count();

            // 8. Write attn_out to g_residual_dev, then outproj+FFN chain ON DEVICE
            auto t_rw2 = Clock::now();
            write_f32_to_buf(g_residual_dev_buf, attn_out, MC::n_head * MC::head_dim);
            g_time_reswrite += std::chrono::duration<double, std::milli>(Clock::now() - t_rw2).count();

            auto t2 = Clock::now();
            if (!g_ffn_chain_traces_valid[layer]) {
                outproj_ffn_chain_ops(g_wt.attn_wo_buf[attn_idx], MC::n_embd, MC::n_head * MC::head_dim,
                                      g_wt.post_norm_buf[layer],
                                      g_wt.attn_ffn_gate_buf[attn_idx], g_wt.attn_ffn_up_buf[attn_idx],
                                      g_wt.attn_ffn_down_buf[attn_idx]);
                Finish(cq0);
                auto tid = g_mesh->begin_mesh_trace(0);
                outproj_ffn_chain_ops(g_wt.attn_wo_buf[attn_idx], MC::n_embd, MC::n_head * MC::head_dim,
                                      g_wt.post_norm_buf[layer],
                                      g_wt.attn_ffn_gate_buf[attn_idx], g_wt.attn_ffn_up_buf[attn_idx],
                                      g_wt.attn_ffn_down_buf[attn_idx]);
                g_mesh->end_mesh_trace(0, tid);
                g_ffn_chain_traces[layer] = tid;
                g_ffn_chain_traces_valid[layer] = true;
            } else {
                g_mesh->replay_mesh_trace(0, g_ffn_chain_traces[layer], false);
            }
            g_time_ffn += std::chrono::duration<double, std::milli>(Clock::now() - t2).count();

            attn_idx++;
        }
    }

    // Output norm (host) + LM head GEMV (device, traced)
    auto t_lm = Clock::now();
    auto& gb_lm = get_gemv_buf(g_mesh.get(), MC::n_vocab, MC::n_embd);

    // Read hidden from device (waits for last FFN chain)
    read_device_to_f32(g_hidden_dev_buf, g_hidden_f32.data(), MC::n_embd, cq0);

    // Host rmsnorm with output norm weights
    rmsnorm(g_hidden_f32.data(), g_output_norm.data(), g_norm_out.data(), MC::n_embd);
    write_f32_to_buf(g_norm_dev_buf, g_norm_out.data(), MC::n_embd);

    // LM head GEMV (traced for fast dispatch)
    if (!g_lmhead_trace_valid) {
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
        g_mesh->replay_mesh_trace(0, g_lmhead_trace, false);
    }
    EnqueueReadMeshBuffer(cq0, gb_lm.out_host_tiled, gb_lm.out_buf, true);

    // Untilize logits
    float* logits = g_logits.data();
    read_gemv_to_f32(gb_lm, logits, MC::n_vocab);
    g_time_lmhead += std::chrono::duration<double, std::milli>(Clock::now() - t_lm).count();

    g_decode_count++;
    if (g_decode_count % 10 == 0) {
        int dc = g_decode_count;
        printf("  [profile @%d] norm_mm=%.0f outproj=%.0f ffn=%.0f host=%.0f reswr=%.0f lmhead=%.0f ms/tok\n",
               dc, g_time_norm_mm / dc, g_time_outproj / dc, g_time_ffn / dc,
               g_time_host / dc, g_time_reswrite / dc, g_time_lmhead / dc);
        printf("    host_detail: conv1d=%.1f deltanet=%.1f untilize=%.1f attn=%.1f ms/tok\n",
               g_time_conv1d / dc, g_time_deltanet / dc, g_time_untilize / dc, g_time_attn_host / dc);
    }

    return logits;
}

// ============================================================================
// Public API
// ============================================================================

bool load_model_and_tokenizer(const char* model_path, int max_ctx) {
    printf("Loading model from %s (max_ctx=%d)...\n", model_path, max_ctx);

    // Open chip 0 only (opening 2 devices adds ~10ms/tok driver overhead)
    auto meshes = MeshDevice::create_unit_meshes({0});
    g_mesh = meshes[0];

    auto grid = g_mesh->compute_with_storage_grid_size();
    printf("Chip 0 opened: compute grid %zux%zu (%zu cores)\n",
           grid.x, grid.y, grid.x * grid.y);

    // Query DRAM bank topology for sharded GEMV
    g_num_dram_banks = g_mesh->num_dram_channels();
    g_dram_workers = g_mesh->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_0);
    printf("DRAM banks: %u, optimal workers:\n", g_num_dram_banks);
    for (uint32_t b = 0; b < g_num_dram_banks; b++) {
        printf("  bank %u -> core (%zu, %zu)\n", b, g_dram_workers[b].x, g_dram_workers[b].y);
    }

    // Enable program cache for faster repeated matmul dispatch
    g_mesh->enable_program_cache();

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

    // Pack and upload weight matrices to device
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

        // Forward pass with generated token (raw bf16→f32 bit ops)
        const uint16_t* emb = g_model.tok_embd_host.data() + (size_t)next_token * MC::n_embd;
        uint32_t* hbits = reinterpret_cast<uint32_t*>(g_hidden_f32.data());
        for (int j = 0; j < MC::n_embd; j++)
            hbits[j] = static_cast<uint32_t>(emb[j]) << 16;

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
    }
    if (g_lmhead_trace_valid) {
        g_mesh->release_mesh_trace(g_lmhead_trace);
        g_lmhead_trace_valid = false;
    }

    // Clear cached custom kernel workloads (must happen before device close)
    g_eltwise_cache.clear();
    g_gemv_cache.clear();
    g_gemv_resadd_cache.clear();
    g_fused_norm_gemv_cache.clear();
    g_rmsnorm_cache.clear();
    g_mc_rmsnorm_cache.clear();
    g_swiglu_cache.clear();

    // Clear pre-allocated GEMV and FFN buffers
    g_ffn_bufs.clear();
    g_gemv_bufs.clear();

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

    if (g_mesh) {
        g_mesh->close();
        g_mesh.reset();
    }
}
