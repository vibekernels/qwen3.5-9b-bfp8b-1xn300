// SPDX-License-Identifier: Apache-2.0
// Writer for single-core FPU-based RMSNorm.
// Writes normalized output tiles to DRAM.
//
// Compile-time args: [Kt, acc_out_config]
// Runtime args: [out_addr]

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t Kt = get_compile_time_arg_val(0);

    constexpr uint32_t cb_out = tt::CBIndex::c_1;  // separate output CB
    uint32_t tile_size = get_tile_size(cb_out);

    constexpr auto acc_out_args = TensorAccessorArgs<1>();
    const auto acc_out = TensorAccessor(acc_out_args, out_addr, tile_size);

    cb_wait_front(cb_out, Kt);
    uint32_t base = get_read_ptr(cb_out);
    for (uint32_t kt = 0; kt < Kt; kt++)
        noc_async_write_tile(kt, acc_out, base + kt * tile_size);
    noc_async_write_barrier();
    cb_pop_front(cb_out, Kt);
}
