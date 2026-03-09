// SPDX-License-Identifier: Apache-2.0
// Split writer for fused gate+up GEMV: writes tiles below split_tile to dst0,
// tiles at or above split_tile to dst1 (with adjusted index).
// This allows a single GEMV to populate two separate output buffers.
//
// Compile-time args: [cb_out, acc0_config, acc1_config]
// Runtime args: [dst0_addr, dst1_addr, Mt_per_core, out_start_tile, split_tile]

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    uint32_t dst0_addr      = get_arg_val<uint32_t>(0);
    uint32_t dst1_addr      = get_arg_val<uint32_t>(1);
    uint32_t Mt_per_core    = get_arg_val<uint32_t>(2);
    uint32_t out_start_tile = get_arg_val<uint32_t>(3);
    uint32_t split_tile     = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);

    uint32_t tile_size = get_tile_size(cb_out);

    constexpr auto acc0_args = TensorAccessorArgs<1>();
    const auto acc0 = TensorAccessor(acc0_args, dst0_addr, tile_size);
    constexpr auto acc1_args = TensorAccessorArgs<acc0_args.next_compile_time_args_offset()>();
    const auto acc1 = TensorAccessor(acc1_args, dst1_addr, tile_size);

    for (uint32_t mt = 0; mt < Mt_per_core; mt++) {
        cb_wait_front(cb_out, 1);
        uint32_t l1_addr = get_read_ptr(cb_out);
        uint32_t tile_idx = out_start_tile + mt;

        if (tile_idx < split_tile) {
            noc_async_write_tile(tile_idx, acc0, l1_addr);
        } else {
            noc_async_write_tile(tile_idx - split_tile, acc1, l1_addr);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
