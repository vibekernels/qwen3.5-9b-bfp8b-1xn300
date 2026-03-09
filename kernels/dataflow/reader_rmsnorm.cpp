// SPDX-License-Identifier: Apache-2.0
// Single-kernel RMSNorm: reads input + weight, computes RMSNorm in scalar, writes output.
// OPTIMIZED: batched NOC reads — all tiles read at once, single barrier.
// Uses two CBs: cb_in for input tiles (retained for pass 2), cb_weight for weight tiles.
//
// Compile-time args: [n_tiles, acc_in_config, acc_weight_config, acc_out_config]
// Runtime args: [in_addr, weight_addr, out_addr, n_elements]

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

inline float bf16_to_f32(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}

inline uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, 4);
    return static_cast<uint16_t>(bits >> 16);
}

void kernel_main() {
    uint32_t in_addr     = get_arg_val<uint32_t>(0);
    uint32_t weight_addr = get_arg_val<uint32_t>(1);
    uint32_t out_addr    = get_arg_val<uint32_t>(2);
    uint32_t n_elements  = get_arg_val<uint32_t>(3);

    constexpr uint32_t n_tiles = get_compile_time_arg_val(0);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_weight = tt::CBIndex::c_1;
    uint32_t tile_size = get_tile_size(cb_in);

    constexpr auto acc_in_args = TensorAccessorArgs<1>();
    const auto acc_in = TensorAccessor(acc_in_args, in_addr, tile_size);
    constexpr auto acc_w_args = TensorAccessorArgs<acc_in_args.next_compile_time_args_offset()>();
    const auto acc_w = TensorAccessor(acc_w_args, weight_addr, tile_size);
    constexpr auto acc_out_args = TensorAccessorArgs<acc_w_args.next_compile_time_args_offset()>();
    const auto acc_out = TensorAccessor(acc_out_args, out_addr, tile_size);

    // ---- Read ALL input tiles at once (batched NOC read) ----
    cb_reserve_back(cb_in, n_tiles);
    uint32_t in_l1_base = get_write_ptr(cb_in);
    for (uint32_t t = 0; t < n_tiles; t++) {
        noc_async_read_tile(t, acc_in, in_l1_base + t * tile_size);
    }
    noc_async_read_barrier();  // single barrier for all reads
    cb_push_back(cb_in, n_tiles);

    // ---- Pass 1: compute sum of squares from all tiles ----
    cb_wait_front(cb_in, n_tiles);
    float sum_sq = 0.0f;
    for (uint32_t t = 0; t < n_tiles; t++) {
        volatile tt_l1_ptr uint16_t* d = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_l1_base + t * tile_size);
        uint32_t base = t * 32;
        for (uint32_t j = 0; j < 16 && (base + j) < n_elements; j++) {
            float v = bf16_to_f32(d[j]);
            sum_sq += v * v;
        }
        for (uint32_t j = 0; j < 16 && (base + 16 + j) < n_elements; j++) {
            float v = bf16_to_f32(d[256 + j]);
            sum_sq += v * v;
        }
    }

    // Compute 1/sqrt(mean(x^2) + eps) using fast inverse sqrt (2 Newton iterations)
    float mean_sq = sum_sq / (float)n_elements;
    float val = mean_sq + 1e-6f;
    float x2 = val * 0.5f;
    uint32_t i;
    __builtin_memcpy(&i, &val, 4);
    i = 0x5f3759df - (i >> 1);
    float norm_factor;
    __builtin_memcpy(&norm_factor, &i, 4);
    norm_factor = norm_factor * (1.5f - x2 * norm_factor * norm_factor);
    norm_factor = norm_factor * (1.5f - x2 * norm_factor * norm_factor);

    // ---- Read ALL weight tiles at once (batched NOC read) ----
    cb_reserve_back(cb_weight, n_tiles);
    uint32_t w_l1_base = get_write_ptr(cb_weight);
    for (uint32_t t = 0; t < n_tiles; t++) {
        noc_async_read_tile(t, acc_w, w_l1_base + t * tile_size);
    }
    noc_async_read_barrier();  // single barrier for all weight reads
    cb_push_back(cb_weight, n_tiles);
    cb_wait_front(cb_weight, n_tiles);

    // ---- Pass 2: normalize, multiply by weight, write output ----
    // Input tiles still in cb_in L1, weight tiles in cb_weight L1.
    // Write output in-place over weight tiles (reuse L1 space).
    for (uint32_t t = 0; t < n_tiles; t++) {
        volatile tt_l1_ptr uint16_t* in_d = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_l1_base + t * tile_size);
        volatile tt_l1_ptr uint16_t* w_d = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(w_l1_base + t * tile_size);
        uint32_t base = t * 32;

        // Face 0: elements [0..15]
        for (uint32_t j = 0; j < 16 && (base + j) < n_elements; j++) {
            float result = bf16_to_f32(in_d[j]) * norm_factor * bf16_to_f32(w_d[j]);
            w_d[j] = f32_to_bf16(result);
        }
        // Face 2: elements [16..31]
        for (uint32_t j = 0; j < 16 && (base + 16 + j) < n_elements; j++) {
            float result = bf16_to_f32(in_d[256 + j]) * norm_factor * bf16_to_f32(w_d[256 + j]);
            w_d[256 + j] = f32_to_bf16(result);
        }

        // Write output tile (from weight L1 slot which now holds result)
        noc_async_write_tile(t, acc_out, w_l1_base + t * tile_size);
    }
    noc_async_write_barrier();  // single barrier for all writes

    // Release CBs
    cb_pop_front(cb_in, n_tiles);
    cb_pop_front(cb_weight, n_tiles);
}
