// SPDX-License-Identifier: Apache-2.0
// Generic writer for operations that produce a single output tensor.
// Writes tiles from output circular buffer to DRAM.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr    = get_arg_val<uint32_t>(0);
    uint32_t num_tiles   = get_arg_val<uint32_t>(1);
    uint32_t start_tile  = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr auto dst_args = TensorAccessorArgs<0>();
    const auto dst_accessor = TensorAccessor(dst_args, dst_addr, get_tile_size(cb_out));

    for (uint32_t t = 0; t < num_tiles; t++) {
        uint32_t tile_idx = start_tile + t;
        cb_wait_front(cb_out, 1);
        noc_async_write_tile(tile_idx, dst_accessor, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
