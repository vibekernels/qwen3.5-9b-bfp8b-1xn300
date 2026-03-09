// SPDX-License-Identifier: Apache-2.0
// Reader for GEMV: y[1,M] = x[1,K] @ W[M,K]^T
// Streams activation x (reused) and weight tile-rows for each output tile.
// Compile-time args: [cb_act, cb_weight, Kt, acc_act_config, acc_weight_config]
// Runtime args: [act_addr, weight_addr, Mt]

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    uint32_t act_addr    = get_arg_val<uint32_t>(0);
    uint32_t weight_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt          = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_act    = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(1);
    constexpr uint32_t Kt        = get_compile_time_arg_val(2);

    uint32_t act_tile_size    = get_tile_size(cb_act);
    uint32_t weight_tile_size = get_tile_size(cb_weight);

    constexpr auto acc_act_args = TensorAccessorArgs<3>();
    const auto acc_act = TensorAccessor(acc_act_args, act_addr, act_tile_size);

    constexpr auto acc_weight_args = TensorAccessorArgs<acc_act_args.next_compile_time_args_offset()>();
    const auto acc_weight = TensorAccessor(acc_weight_args, weight_addr, weight_tile_size);

    // For each output tile mt: stream Kt pairs of (act, weight) tiles
    // Weight is stored [M, K] row-major tiled, transposed for matmul
    for (uint32_t mt = 0; mt < Mt; mt++) {
        for (uint32_t kt = 0; kt < Kt; kt++) {
            // Activation tile [0, kt] — same for every mt
            cb_reserve_back(cb_act, 1);
            uint32_t l1_act = get_write_ptr(cb_act);
            noc_async_read_tile(kt, acc_act, l1_act);
            noc_async_read_barrier();
            cb_push_back(cb_act, 1);

            // Weight tile [mt, kt]
            cb_reserve_back(cb_weight, 1);
            uint32_t l1_weight = get_write_ptr(cb_weight);
            noc_async_read_tile(mt * Kt + kt, acc_weight, l1_weight);
            noc_async_read_barrier();
            cb_push_back(cb_weight, 1);
        }
    }
}
