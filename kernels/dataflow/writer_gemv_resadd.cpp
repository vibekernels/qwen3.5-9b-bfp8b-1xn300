// SPDX-License-Identifier: Apache-2.0
// Fused GEMV writer + residual add: reads existing residual tile, adds GEMV output, writes back.
// Eliminates separate eltwise_binary dispatch for residual connections.
//
// Compile-time args: [cb_out, acc_residual_config]
// Runtime args: [residual_addr, Mt_per_core, out_start_tile]

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
    uint32_t residual_addr   = get_arg_val<uint32_t>(0);
    uint32_t Mt_per_core     = get_arg_val<uint32_t>(1);
    uint32_t out_start_tile  = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);

    uint32_t tile_size = get_tile_size(cb_out);

    constexpr auto acc_args = TensorAccessorArgs<1>();
    const auto acc = TensorAccessor(acc_args, residual_addr, tile_size);

    // Use cb_scratch (c_2) as temp space for reading residual tiles
    constexpr uint32_t cb_scratch = tt::CBIndex::c_2;
    cb_reserve_back(cb_scratch, 1);
    uint32_t scratch_l1 = get_write_ptr(cb_scratch);

    for (uint32_t mt = 0; mt < Mt_per_core; mt++) {
        // Wait for GEMV output tile
        cb_wait_front(cb_out, 1);
        uint32_t out_l1 = get_read_ptr(cb_out);

        // Read existing residual tile from DRAM
        noc_async_read_tile(out_start_tile + mt, acc, scratch_l1);
        noc_async_read_barrier();

        // Add residual to GEMV output in-place (scalar bf16 arithmetic)
        // Tile layout: face 0 at offset 0 (16 elements), face 2 at offset 256 (16 elements)
        volatile tt_l1_ptr uint16_t* gemv =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out_l1);
        volatile tt_l1_ptr uint16_t* res =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch_l1);

        // Face 0: elements [0..15]
        for (uint32_t j = 0; j < 16; j++) {
            float sum = bf16_to_f32(gemv[j]) + bf16_to_f32(res[j]);
            gemv[j] = f32_to_bf16(sum);
        }
        // Face 2: elements [16..31]
        for (uint32_t j = 0; j < 16; j++) {
            float sum = bf16_to_f32(gemv[256 + j]) + bf16_to_f32(res[256 + j]);
            gemv[256 + j] = f32_to_bf16(sum);
        }

        // Write result back to residual buffer
        noc_async_write_tile(out_start_tile + mt, acc, out_l1);
        noc_async_write_barrier();

        cb_pop_front(cb_out, 1);
    }
}
