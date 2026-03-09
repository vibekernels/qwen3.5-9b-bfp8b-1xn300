// SPDX-License-Identifier: Apache-2.0
// Multi-core GEMV reader: each core reads its assigned weight rows from interleaved DRAM.
// Activation tiles loaded ONCE into cb_act (stays in L1 for all output rows).
// Weight tiles read via TensorAccessor (interleaved buffer, round-robin across banks).
//
// Compile-time args: [cb_act, cb_weight, Kt, BLOCK, acc_act_config, acc_weight_config]
// Runtime args: [act_addr, weight_addr, Mt_per_core, weight_start_tile]

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    uint32_t act_addr           = get_arg_val<uint32_t>(0);
    uint32_t weight_addr        = get_arg_val<uint32_t>(1);
    uint32_t Mt_per_core        = get_arg_val<uint32_t>(2);
    uint32_t weight_start_tile  = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_act    = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(1);
    constexpr uint32_t Kt        = get_compile_time_arg_val(2);
    constexpr uint32_t BLOCK     = get_compile_time_arg_val(3);

    uint32_t act_tile_size    = get_tile_size(cb_act);
    uint32_t weight_tile_size = get_tile_size(cb_weight);

    // Activation accessor (interleaved)
    constexpr auto acc_act_args = TensorAccessorArgs<4>();
    const auto acc_act = TensorAccessor(acc_act_args, act_addr, act_tile_size);

    // Weight accessor (interleaved)
    constexpr auto acc_weight_args = TensorAccessorArgs<acc_act_args.next_compile_time_args_offset()>();
    const auto acc_weight = TensorAccessor(acc_weight_args, weight_addr, weight_tile_size);

    // Phase 1: Load ALL activation tiles into cb_act (stays in L1)
    constexpr uint32_t ACT_BATCH = 8;
    for (uint32_t kt = 0; kt < Kt; kt += ACT_BATCH) {
        uint32_t batch = (kt + ACT_BATCH <= Kt) ? ACT_BATCH : (Kt - kt);
        cb_reserve_back(cb_act, batch);
        uint32_t l1_base = get_write_ptr(cb_act);
        for (uint32_t b = 0; b < batch; b++) {
            noc_async_read_tile(kt + b, acc_act, l1_base + b * act_tile_size);
        }
        noc_async_read_barrier();
        cb_push_back(cb_act, batch);
    }

    // Phase 2: Read weight tiles from interleaved buffer via TensorAccessor.
    // Each output row needs Kt weight tiles, read in BLOCK-sized batches.
    uint32_t weight_tile = weight_start_tile;

    for (uint32_t mt = 0; mt < Mt_per_core; mt++) {
        uint32_t num_blocks = (Kt + BLOCK - 1) / BLOCK;

        for (uint32_t blk = 0; blk < num_blocks; blk++) {
            uint32_t batch = ((blk + 1) * BLOCK <= Kt) ? BLOCK : (Kt - blk * BLOCK);

            cb_reserve_back(cb_weight, batch);
            uint32_t l1_base = get_write_ptr(cb_weight);

            for (uint32_t b = 0; b < batch; b++) {
                noc_async_read_tile(weight_tile, acc_weight, l1_base + b * weight_tile_size);
                weight_tile++;
            }
            noc_async_read_barrier();
            cb_push_back(cb_weight, batch);
        }
    }
}
