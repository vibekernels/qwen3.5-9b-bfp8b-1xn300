// SPDX-License-Identifier: Apache-2.0
// Reader for single-core FPU-based RMSNorm.
// Reads input + weight from DRAM, generates scaler/epsilon tiles.

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, 4);
    return static_cast<uint16_t>(bits >> 16);
}

// Fill tile with a pre-computed bf16 value (passed as uint16_t bits)
inline void fill_tile_bf16(uint32_t addr, uint32_t tile_size, uint16_t bf16_val) {
    volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr);
    uint32_t n_elems = tile_size / sizeof(uint16_t);
    for (uint32_t i = 0; i < n_elems; i++)
        p[i] = bf16_val;
}

void kernel_main() {
    uint32_t hidden_addr   = get_arg_val<uint32_t>(0);
    uint32_t norm_w_addr   = get_arg_val<uint32_t>(1);
    uint32_t n_elements    = get_arg_val<uint32_t>(2);
    uint32_t scaler_bf16   = get_arg_val<uint32_t>(3);  // pre-computed 1/N as bf16 bits

    constexpr uint32_t Kt = get_compile_time_arg_val(0);

    constexpr uint32_t cb_hidden  = tt::CBIndex::c_0;
    constexpr uint32_t cb_norm_w  = tt::CBIndex::c_2;
    constexpr uint32_t cb_scaler  = tt::CBIndex::c_5;
    constexpr uint32_t cb_eps     = tt::CBIndex::c_6;

    uint32_t tile_size = get_tile_size(cb_hidden);

    constexpr auto acc_hidden_args = TensorAccessorArgs<1>();
    const auto acc_hidden = TensorAccessor(acc_hidden_args, hidden_addr, tile_size);
    constexpr auto acc_norm_w_args = TensorAccessorArgs<acc_hidden_args.next_compile_time_args_offset()>();
    const auto acc_norm_w = TensorAccessor(acc_norm_w_args, norm_w_addr, tile_size);

    // Generate scaler tile: all elements = 1/N (pre-computed on host)
    {
        cb_reserve_back(cb_scaler, 1);
        fill_tile_bf16(get_write_ptr(cb_scaler), tile_size, (uint16_t)scaler_bf16);
        cb_push_back(cb_scaler, 1);
    }

    // Generate epsilon tile: all elements = 1e-6
    {
        cb_reserve_back(cb_eps, 1);
        uint16_t eps_bf16 = f32_to_bf16_bits(1e-6f);
        fill_tile_bf16(get_write_ptr(cb_eps), tile_size, eps_bf16);
        cb_push_back(cb_eps, 1);
    }

    // Read all hidden tiles into cb_hidden
    cb_reserve_back(cb_hidden, Kt);
    uint32_t base = get_write_ptr(cb_hidden);
    for (uint32_t kt = 0; kt < Kt; kt++)
        noc_async_read_tile(kt, acc_hidden, base + kt * tile_size);
    noc_async_read_barrier();
    cb_push_back(cb_hidden, Kt);

    // Read norm weights into cb_norm_w
    cb_reserve_back(cb_norm_w, Kt);
    uint32_t w_base = get_write_ptr(cb_norm_w);
    for (uint32_t kt = 0; kt < Kt; kt++)
        noc_async_read_tile(kt, acc_norm_w, w_base + kt * tile_size);
    noc_async_read_barrier();
    cb_push_back(cb_norm_w, Kt);
}
