// SPDX-License-Identifier: Apache-2.0
// Multi-core binary reader: reads n_tiles from two DRAM interleaved buffers, starting at start_tile.
// Compile-time args: [cb0, cb1, acc0_config, acc1_config]
// Runtime args: [src0_addr, src1_addr, n_tiles, start_tile]

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    uint32_t src0_addr   = get_arg_val<uint32_t>(0);
    uint32_t src1_addr   = get_arg_val<uint32_t>(1);
    uint32_t n_tiles     = get_arg_val<uint32_t>(2);
    uint32_t start_tile  = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb1 = get_compile_time_arg_val(1);

    uint32_t tile_size0 = get_tile_size(cb0);
    uint32_t tile_size1 = get_tile_size(cb1);

    constexpr auto acc0_args = TensorAccessorArgs<2>();
    const auto acc0 = TensorAccessor(acc0_args, src0_addr, tile_size0);

    constexpr auto acc1_args = TensorAccessorArgs<acc0_args.next_compile_time_args_offset()>();
    const auto acc1 = TensorAccessor(acc1_args, src1_addr, tile_size1);

    for (uint32_t i = 0; i < n_tiles; i++) {
        uint32_t tile_idx = start_tile + i;

        cb_reserve_back(cb0, 1);
        uint32_t l1_addr0 = get_write_ptr(cb0);
        noc_async_read_tile(tile_idx, acc0, l1_addr0);
        noc_async_read_barrier();
        cb_push_back(cb0, 1);

        cb_reserve_back(cb1, 1);
        uint32_t l1_addr1 = get_write_ptr(cb1);
        noc_async_read_tile(tile_idx, acc1, l1_addr1);
        noc_async_read_barrier();
        cb_push_back(cb1, 1);
    }
}
