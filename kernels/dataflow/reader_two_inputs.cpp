// SPDX-License-Identifier: Apache-2.0
// Generic reader for binary elementwise operations.
// Reads tiles from two input buffers (A and B) in lockstep.
// Used for: SwiGLU (gate + up), sigmoid_gate (attn + gate),
//           RMSNorm pass 2 (input + weight), etc.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t a_addr     = get_arg_val<uint32_t>(0);
    uint32_t b_addr     = get_arg_val<uint32_t>(1);
    uint32_t num_tiles  = get_arg_val<uint32_t>(2);
    uint32_t start_tile = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;

    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a_accessor = TensorAccessor(a_args, a_addr, get_tile_size(cb_a));
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b_accessor = TensorAccessor(b_args, b_addr, get_tile_size(cb_b));

    for (uint32_t t = 0; t < num_tiles; t++) {
        uint32_t tile_idx = start_tile + t;

        cb_reserve_back(cb_a, 1);
        noc_async_read_tile(tile_idx, a_accessor, get_write_ptr(cb_a));
        noc_async_read_barrier();
        cb_push_back(cb_a, 1);

        cb_reserve_back(cb_b, 1);
        noc_async_read_tile(tile_idx, b_accessor, get_write_ptr(cb_b));
        noc_async_read_barrier();
        cb_push_back(cb_b, 1);
    }
}
