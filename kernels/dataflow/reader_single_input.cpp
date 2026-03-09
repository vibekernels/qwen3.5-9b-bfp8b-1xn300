// SPDX-License-Identifier: Apache-2.0
// Generic reader for unary operations.
// Reads tiles from a single input buffer.
// Used for: embedding lookup, unary activations, etc.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src_addr    = get_arg_val<uint32_t>(0);
    uint32_t num_tiles   = get_arg_val<uint32_t>(1);
    uint32_t start_tile  = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;

    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto src_accessor = TensorAccessor(src_args, src_addr, get_tile_size(cb_in));

    for (uint32_t t = 0; t < num_tiles; t++) {
        uint32_t tile_idx = start_tile + t;
        cb_reserve_back(cb_in, 1);
        noc_async_read_tile(tile_idx, src_accessor, get_write_ptr(cb_in));
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
}
