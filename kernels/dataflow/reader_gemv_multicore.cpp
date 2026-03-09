// SPDX-License-Identifier: Apache-2.0
// Multi-core GEMV reader: each core reads its slice of weight rows.
// Activation tiles loaded ONCE into cb_act (stays in L1 for all output rows).
// Weight tiles read in batches with TRID pipelining for better NOC throughput.
//
// Compile-time args: [cb_act, cb_weight, Kt, acc_act_config, acc_weight_config]
// Runtime args: [act_addr, weight_addr, Mt_per_core, weight_start_tile]

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    uint32_t act_addr         = get_arg_val<uint32_t>(0);
    uint32_t weight_addr      = get_arg_val<uint32_t>(1);
    uint32_t Mt_per_core      = get_arg_val<uint32_t>(2);
    uint32_t weight_start_tile = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_act    = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(1);
    constexpr uint32_t Kt        = get_compile_time_arg_val(2);

    uint32_t act_tile_size    = get_tile_size(cb_act);
    uint32_t weight_tile_size = get_tile_size(cb_weight);

    constexpr auto acc_act_args = TensorAccessorArgs<3>();
    const auto acc_act = TensorAccessor(acc_act_args, act_addr, act_tile_size);

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

    // Phase 2: For each output row, stream weight tiles
    // Use TRID pipelining: alternate between TRID 1 and 2 for consecutive blocks.
    // Wait for previous block's TRID while issuing current block's reads.
    constexpr uint32_t BLOCK = 4;
    for (uint32_t mt = 0; mt < Mt_per_core; mt++) {
        uint32_t weight_row = weight_start_tile + mt;
        uint32_t num_blocks = (Kt + BLOCK - 1) / BLOCK;

        uint32_t prev_trid = 0;  // 0 = no previous block to wait for
        uint32_t prev_batch = 0;

        for (uint32_t blk = 0; blk < num_blocks; blk++) {
            uint32_t kt_start = blk * BLOCK;
            uint32_t batch = (kt_start + BLOCK <= Kt) ? BLOCK : (Kt - kt_start);
            uint32_t curr_trid = (blk & 1) ? 2 : 1;

            // Reserve CB space for this block
            cb_reserve_back(cb_weight, batch);
            uint32_t l1_base = get_write_ptr(cb_weight);

            // Issue reads for this block with curr_trid
            noc_async_read_set_trid(curr_trid);
            for (uint32_t b = 0; b < batch; b++) {
                noc_async_read_tile(weight_row * Kt + kt_start + b, acc_weight,
                                    l1_base + b * weight_tile_size);
            }

            // If there was a previous block, wait for it and push to compute
            if (prev_trid != 0) {
                noc_async_read_barrier_with_trid(prev_trid);
                cb_push_back(cb_weight, prev_batch);
            }

            prev_trid = curr_trid;
            prev_batch = batch;
        }

        // Wait for last block and push
        if (prev_trid != 0) {
            noc_async_read_barrier_with_trid(prev_trid);
            cb_push_back(cb_weight, prev_batch);
        }
    }
}
