// SPDX-License-Identifier: Apache-2.0
// DRAM-sharded GEMV reader: each core reads from its assigned DRAM bank.
// Activation tiles loaded ONCE into cb_act (stays in L1 for all output rows).
// Weight tiles read as large contiguous blocks from a single DRAM bank.
//
// Compile-time args: [cb_act, cb_weight, Kt, BLOCK, acc_act_config]
// Runtime args: [act_addr, weight_bank_addr, Mt_per_core, bank_id, weight_start_offset]

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    uint32_t act_addr              = get_arg_val<uint32_t>(0);
    uint32_t weight_bank_addr      = get_arg_val<uint32_t>(1);
    uint32_t Mt_per_core           = get_arg_val<uint32_t>(2);
    uint32_t bank_id               = get_arg_val<uint32_t>(3);
    uint32_t weight_start_offset   = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_act    = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(1);
    constexpr uint32_t Kt        = get_compile_time_arg_val(2);
    constexpr uint32_t BLOCK     = get_compile_time_arg_val(3);

    uint32_t act_tile_size    = get_tile_size(cb_act);
    uint32_t weight_tile_size = get_tile_size(cb_weight);

    // Activation accessor (interleaved, standard TensorAccessor)
    constexpr auto acc_act_args = TensorAccessorArgs<4>();
    const auto acc_act = TensorAccessor(acc_act_args, act_addr, act_tile_size);

    // Phase 1: Load ALL activation tiles into cb_act (stays in L1)
    cb_reserve_back(cb_act, Kt);
    uint32_t act_l1_base = get_write_ptr(cb_act);
    for (uint32_t kt = 0; kt < Kt; kt++) {
        noc_async_read_tile(kt, acc_act, act_l1_base + kt * act_tile_size);
    }
    noc_async_read_barrier();
    cb_push_back(cb_act, Kt);

    // Phase 2: Read weight tiles from assigned DRAM bank.
    // Large contiguous reads (BLOCK tiles per noc_async_read), no TRID.
    uint64_t bank_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, weight_bank_addr);
    uint32_t weight_offset = weight_start_offset;

    for (uint32_t mt = 0; mt < Mt_per_core; mt++) {
        uint32_t num_blocks = (Kt + BLOCK - 1) / BLOCK;

        for (uint32_t blk = 0; blk < num_blocks; blk++) {
            uint32_t batch = ((blk + 1) * BLOCK <= Kt) ? BLOCK : (Kt - blk * BLOCK);

            cb_reserve_back(cb_weight, batch);
            uint32_t l1_base = get_write_ptr(cb_weight);

            // Single contiguous read for the entire block
            noc_async_read(bank_noc_addr + weight_offset, l1_base, batch * weight_tile_size);
            weight_offset += batch * weight_tile_size;

            noc_async_read_barrier();
            cb_push_back(cb_weight, batch);
        }
    }
}
