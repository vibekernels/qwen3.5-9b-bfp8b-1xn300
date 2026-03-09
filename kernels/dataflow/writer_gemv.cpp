// SPDX-License-Identifier: Apache-2.0
// Writer for GEMV: writes Mt output tiles from CB to DRAM buffer.
// Compile-time args: [cb_out, acc_config]
// Runtime args: [dst_addr, Mt]

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt       = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);

    uint32_t tile_size = get_tile_size(cb_out);

    constexpr auto acc_args = TensorAccessorArgs<1>();
    const auto acc = TensorAccessor(acc_args, dst_addr, tile_size);

    for (uint32_t mt = 0; mt < Mt; mt++) {
        cb_wait_front(cb_out, 1);
        uint32_t l1_addr = get_read_ptr(cb_out);
        noc_async_write_tile(mt, acc, l1_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
