// SPDX-License-Identifier: Apache-2.0
// Multi-core GEMV writer: writes Mt_per_core output tiles to the correct offset in output buffer.
// Compile-time args: [cb_out, acc_config]
// Runtime args: [dst_addr, Mt_per_core, out_start_tile]

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    uint32_t dst_addr       = get_arg_val<uint32_t>(0);
    uint32_t Mt_per_core    = get_arg_val<uint32_t>(1);
    uint32_t out_start_tile = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);

    uint32_t tile_size = get_tile_size(cb_out);

    constexpr auto acc_args = TensorAccessorArgs<1>();
    const auto acc = TensorAccessor(acc_args, dst_addr, tile_size);

    for (uint32_t mt = 0; mt < Mt_per_core; mt++) {
        cb_wait_front(cb_out, 1);
        uint32_t l1_addr = get_read_ptr(cb_out);
        noc_async_write_tile(out_start_tile + mt, acc, l1_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
