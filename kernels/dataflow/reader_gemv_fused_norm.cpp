// SPDX-License-Identifier: Apache-2.0
// Multi-core GEMV reader with FUSED RMSNorm.
// Each core independently computes RMSNorm on the full hidden vector,
// then uses the normalized result as GEMV activations.
// Weight tiles read via TensorAccessor (interleaved buffer).
//
// Compile-time args: [cb_act, cb_weight, cb_norm, Kt, BLOCK,
//                     acc_hidden_config, acc_norm_weight_config, acc_weight_config]
// Runtime args: [hidden_addr, norm_weight_addr, weight_addr,
//                Mt_per_core, n_elements, weight_start_tile]

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
    uint32_t hidden_addr           = get_arg_val<uint32_t>(0);
    uint32_t norm_weight_addr      = get_arg_val<uint32_t>(1);
    uint32_t weight_addr           = get_arg_val<uint32_t>(2);
    uint32_t Mt_per_core           = get_arg_val<uint32_t>(3);
    uint32_t n_elements            = get_arg_val<uint32_t>(4);
    uint32_t weight_start_tile     = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_act    = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(1);
    constexpr uint32_t cb_norm   = get_compile_time_arg_val(2);
    constexpr uint32_t Kt        = get_compile_time_arg_val(3);
    constexpr uint32_t BLOCK     = get_compile_time_arg_val(4);

    uint32_t act_tile_size    = get_tile_size(cb_act);
    uint32_t weight_tile_size = get_tile_size(cb_weight);

    // Hidden and norm_weight accessors (interleaved buffers)
    constexpr auto acc_hidden_args = TensorAccessorArgs<5>();
    const auto acc_hidden = TensorAccessor(acc_hidden_args, hidden_addr, act_tile_size);
    constexpr auto acc_norm_w_args = TensorAccessorArgs<acc_hidden_args.next_compile_time_args_offset()>();
    const auto acc_norm_w = TensorAccessor(acc_norm_w_args, norm_weight_addr, act_tile_size);
    // Weight accessor (interleaved)
    constexpr auto acc_weight_args = TensorAccessorArgs<acc_norm_w_args.next_compile_time_args_offset()>();
    const auto acc_weight = TensorAccessor(acc_weight_args, weight_addr, weight_tile_size);

    // ======== Phase 1: Fused RMSNorm ========

    // Batch read ALL hidden tiles into cb_act
    cb_reserve_back(cb_act, Kt);
    uint32_t act_l1_base = get_write_ptr(cb_act);
    for (uint32_t kt = 0; kt < Kt; kt++) {
        noc_async_read_tile(kt, acc_hidden, act_l1_base + kt * act_tile_size);
    }
    noc_async_read_barrier();

    // Compute sum of squares from all tiles
    float sum_sq = 0.0f;
    for (uint32_t kt = 0; kt < Kt; kt++) {
        volatile tt_l1_ptr uint16_t* d =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(act_l1_base + kt * act_tile_size);
        uint32_t base = kt * 32;
        for (uint32_t j = 0; j < 16 && (base + j) < n_elements; j++) {
            float v = bf16_to_f32(d[j]);
            sum_sq += v * v;
        }
        for (uint32_t j = 0; j < 16 && (base + 16 + j) < n_elements; j++) {
            float v = bf16_to_f32(d[256 + j]);
            sum_sq += v * v;
        }
    }

    // Fast inverse sqrt: 1/sqrt(sum_sq/n + eps)
    float mean_sq = sum_sq / (float)n_elements;
    float val = mean_sq + 1e-6f;
    float x2 = val * 0.5f;
    uint32_t ii;
    __builtin_memcpy(&ii, &val, 4);
    ii = 0x5f3759df - (ii >> 1);
    float norm_factor;
    __builtin_memcpy(&norm_factor, &ii, 4);
    norm_factor = norm_factor * (1.5f - x2 * norm_factor * norm_factor);
    norm_factor = norm_factor * (1.5f - x2 * norm_factor * norm_factor);

    // Batch read ALL norm weight tiles into cb_norm
    cb_reserve_back(cb_norm, Kt);
    uint32_t norm_l1_base = get_write_ptr(cb_norm);
    for (uint32_t kt = 0; kt < Kt; kt++) {
        noc_async_read_tile(kt, acc_norm_w, norm_l1_base + kt * act_tile_size);
    }
    noc_async_read_barrier();

    // Normalize in-place: act[i] = hidden[i] * norm_factor * norm_weight[i]
    for (uint32_t kt = 0; kt < Kt; kt++) {
        volatile tt_l1_ptr uint16_t* hd =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(act_l1_base + kt * act_tile_size);
        volatile tt_l1_ptr uint16_t* wd =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(norm_l1_base + kt * act_tile_size);
        uint32_t base = kt * 32;
        // Face 0: elements [0..15]
        for (uint32_t j = 0; j < 16 && (base + j) < n_elements; j++) {
            float result = bf16_to_f32(hd[j]) * norm_factor * bf16_to_f32(wd[j]);
            hd[j] = f32_to_bf16(result);
        }
        // Face 1: elements [16..31]
        for (uint32_t j = 0; j < 16 && (base + 16 + j) < n_elements; j++) {
            float result = bf16_to_f32(hd[256 + j]) * norm_factor * bf16_to_f32(wd[256 + j]);
            hd[256 + j] = f32_to_bf16(result);
        }
    }

    // Release norm weight CB (no longer needed)
    cb_push_back(cb_norm, Kt);
    cb_wait_front(cb_norm, Kt);
    cb_pop_front(cb_norm, Kt);

    // Push normalized activations for compute kernel
    cb_push_back(cb_act, Kt);

    // ======== Phase 2: GEMV weight reads via TensorAccessor ========
    uint32_t weight_tile = weight_start_tile;

    for (uint32_t mt = 0; mt < Mt_per_core; mt++) {
        uint32_t num_blocks = (Kt + BLOCK - 1) / BLOCK;

        for (uint32_t blk = 0; blk < num_blocks; blk++) {
            uint32_t batch = ((blk + 1) * BLOCK <= Kt) ? BLOCK : (Kt - blk * BLOCK);

            cb_reserve_back(cb_weight, batch);
            uint32_t l1_base = get_write_ptr(cb_weight);

            for (uint32_t b = 0; b < batch; b++) {
                noc_async_read_tile(weight_tile, acc_weight, l1_base + b * weight_tile_size);
                weight_tile++;
            }
            noc_async_read_barrier();
            cb_push_back(cb_weight, batch);
        }
    }
}
