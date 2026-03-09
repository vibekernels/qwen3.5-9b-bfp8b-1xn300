// SPDX-License-Identifier: Apache-2.0
// DRAM-sharded GEMV reader with FPU-based RMSNorm.
// Reader pushes hidden tiles, norm weights, scaler, and epsilon to CBs.
// Compute kernel does the actual RMSNorm using FPU, then GEMV.
//
// Compile-time args: [cb_hidden, cb_weight, cb_norm_w, cb_scaler, cb_eps,
//                     Kt, BLOCK, acc_hidden_config, acc_norm_w_config]
// Runtime args: [hidden_addr, norm_w_addr, weight_bank_addr,
//                Mt_per_core, bank_id, n_elements]

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, 4);
    return static_cast<uint16_t>(bits >> 16);
}

void kernel_main() {
    uint32_t hidden_addr       = get_arg_val<uint32_t>(0);
    uint32_t norm_w_addr       = get_arg_val<uint32_t>(1);
    uint32_t weight_bank_addr  = get_arg_val<uint32_t>(2);
    uint32_t Mt_per_core       = get_arg_val<uint32_t>(3);
    uint32_t bank_id           = get_arg_val<uint32_t>(4);
    uint32_t n_elements        = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_hidden  = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight  = get_compile_time_arg_val(1);
    constexpr uint32_t cb_norm_w  = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scaler  = get_compile_time_arg_val(3);
    constexpr uint32_t cb_eps     = get_compile_time_arg_val(4);
    constexpr uint32_t Kt         = get_compile_time_arg_val(5);
    constexpr uint32_t BLOCK      = get_compile_time_arg_val(6);

    uint32_t hidden_tile_size = get_tile_size(cb_hidden);
    uint32_t weight_tile_size = get_tile_size(cb_weight);

    // Tensor accessors for hidden and norm weight buffers (interleaved)
    constexpr auto acc_hidden_args = TensorAccessorArgs<7>();
    const auto acc_hidden = TensorAccessor(acc_hidden_args, hidden_addr, hidden_tile_size);
    constexpr auto acc_norm_w_args = TensorAccessorArgs<acc_hidden_args.next_compile_time_args_offset()>();
    const auto acc_norm_w = TensorAccessor(acc_norm_w_args, norm_w_addr, hidden_tile_size);

    // ======== Phase 0: Generate scaler (1/N) and epsilon tiles ========
    {
        // Scaler tile: fill with 1/N for REDUCE_SCALAR
        // For REDUCE_SCALAR with SUM, scaler is multiplied during accumulation.
        // The scaler tile format: first row of each face has the value.
        float inv_n = 1.0f / (float)n_elements;
        uint16_t inv_n_bf16 = f32_to_bf16_bits(inv_n);
        uint32_t packed = ((uint32_t)inv_n_bf16 << 16) | inv_n_bf16;

        cb_reserve_back(cb_scaler, 1);
        uint32_t scaler_addr = get_write_ptr(cb_scaler);
        volatile tt_l1_ptr uint32_t* scaler_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scaler_addr);

        // Zero the tile
        for (uint32_t i = 0; i < hidden_tile_size / sizeof(uint32_t); i++) {
            scaler_ptr[i] = 0;
        }

        // Fill first row of each face with the scaler value (16 elements = 8 uint32s)
        // Face 0: offset 0 (row 0, 16 elements)
        for (uint32_t i = 0; i < 8; i++) scaler_ptr[i] = packed;
        // Face 1: offset 256 bytes = 128 uint16 = 64 uint32
        for (uint32_t i = 0; i < 8; i++) scaler_ptr[128 + i] = packed;
        // Face 2: offset 512 bytes = 256 uint16 = 128 uint32
        for (uint32_t i = 0; i < 8; i++) scaler_ptr[256 + i] = packed;
        // Face 3: offset 768 bytes = 384 uint16 = 192 uint32
        for (uint32_t i = 0; i < 8; i++) scaler_ptr[384 + i] = packed;

        cb_push_back(cb_scaler, 1);
    }
    {
        // Epsilon tile: single value at [0,0] for add_tiles
        // After REDUCE_SCALAR, result is a scalar in [0,0] of the tile.
        // We add epsilon to this. For add_tiles, the eps tile should have
        // epsilon at [0,0] and zeros elsewhere.
        float eps = 1e-6f;
        uint16_t eps_bf16 = f32_to_bf16_bits(eps);

        cb_reserve_back(cb_eps, 1);
        uint32_t eps_addr = get_write_ptr(cb_eps);
        volatile tt_l1_ptr uint32_t* eps_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(eps_addr);

        // Zero the tile
        for (uint32_t i = 0; i < hidden_tile_size / sizeof(uint32_t); i++) {
            eps_ptr[i] = 0;
        }

        // Set [face 0, row 0, col 0] = epsilon
        volatile tt_l1_ptr uint16_t* eps_u16 =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(eps_addr);
        eps_u16[0] = eps_bf16;

        cb_push_back(cb_eps, 1);
    }

    // ======== Phase 1: Read ALL hidden tiles ========
    cb_reserve_back(cb_hidden, Kt);
    uint32_t hidden_l1_base = get_write_ptr(cb_hidden);
    for (uint32_t kt = 0; kt < Kt; kt++) {
        noc_async_read_tile(kt, acc_hidden, hidden_l1_base + kt * hidden_tile_size);
    }
    noc_async_read_barrier();
    cb_push_back(cb_hidden, Kt);

    // ======== Phase 2: Read ALL norm weight tiles ========
    cb_reserve_back(cb_norm_w, Kt);
    uint32_t norm_l1_base = get_write_ptr(cb_norm_w);
    for (uint32_t kt = 0; kt < Kt; kt++) {
        noc_async_read_tile(kt, acc_norm_w, norm_l1_base + kt * hidden_tile_size);
    }
    noc_async_read_barrier();
    cb_push_back(cb_norm_w, Kt);

    // ======== Phase 3: Read GEMV weight tiles (same as reader_gemv_dram_sharded) ========
    // Compute kernel handles rmsnorm using FPU while we wait for weight reads.
    // Weight reads pipeline with compute's GEMV.
    uint64_t bank_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, weight_bank_addr);
    uint32_t weight_offset = 0;

    for (uint32_t mt = 0; mt < Mt_per_core; mt++) {
        uint32_t num_blocks = (Kt + BLOCK - 1) / BLOCK;
        uint32_t prev_trid = 0;
        uint32_t prev_batch = 0;

        for (uint32_t blk = 0; blk < num_blocks; blk++) {
            uint32_t batch = ((blk + 1) * BLOCK <= Kt) ? BLOCK : (Kt - blk * BLOCK);
            uint32_t curr_trid = (blk & 1) ? 2 : 1;

            cb_reserve_back(cb_weight, batch);
            uint32_t l1_base = get_write_ptr(cb_weight);

            noc_async_read_set_trid(curr_trid);
            noc_async_read(bank_noc_addr + weight_offset, l1_base, batch * weight_tile_size);
            weight_offset += batch * weight_tile_size;

            if (prev_trid != 0) {
                noc_async_read_barrier_with_trid(prev_trid);
                cb_push_back(cb_weight, prev_batch);
            }

            prev_trid = curr_trid;
            prev_batch = batch;
        }

        if (prev_trid != 0) {
            noc_async_read_barrier_with_trid(prev_trid);
            cb_push_back(cb_weight, prev_batch);
        }
    }
}
