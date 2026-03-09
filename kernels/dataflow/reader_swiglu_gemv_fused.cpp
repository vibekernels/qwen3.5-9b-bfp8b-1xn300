// SPDX-License-Identifier: Apache-2.0
// Fused SwiGLU + GEMV reader: reads gate+up tiles for SwiGLU compute, then reads
// weight tiles for GEMV. The SwiGLU output stays in L1 as GEMV activations (no DRAM round-trip).
//
// Compile-time args: [cb_gate(c_0), cb_up(c_1), cb_weight(c_3), Kt, BLOCK,
//                     acc_gate_config, acc_up_config]
// Runtime args: [gate_addr, up_addr, weight_bank_addr, n_act_tiles, Mt_per_core, bank_id]

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    uint32_t gate_addr         = get_arg_val<uint32_t>(0);
    uint32_t up_addr           = get_arg_val<uint32_t>(1);
    uint32_t weight_bank_addr  = get_arg_val<uint32_t>(2);
    uint32_t n_act_tiles       = get_arg_val<uint32_t>(3);
    uint32_t Mt_per_core       = get_arg_val<uint32_t>(4);
    uint32_t bank_id           = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_gate   = get_compile_time_arg_val(0);  // c_0: gate input
    constexpr uint32_t cb_up     = get_compile_time_arg_val(1);  // c_1: up input
    constexpr uint32_t cb_weight = get_compile_time_arg_val(2);  // c_3: GEMV weights
    constexpr uint32_t Kt        = get_compile_time_arg_val(3);
    constexpr uint32_t BLOCK     = get_compile_time_arg_val(4);

    uint32_t gate_tile_size   = get_tile_size(cb_gate);
    uint32_t weight_tile_size = get_tile_size(cb_weight);

    constexpr auto acc_gate_args = TensorAccessorArgs<5>();
    const auto acc_gate = TensorAccessor(acc_gate_args, gate_addr, gate_tile_size);
    constexpr auto acc_up_args = TensorAccessorArgs<acc_gate_args.next_compile_time_args_offset()>();
    const auto acc_up = TensorAccessor(acc_up_args, up_addr, gate_tile_size);

    // Phase 1: Read gate+up tiles for SwiGLU compute
    // Compute kernel will produce SwiGLU output in cb_act (c_16 repacked to c_4)
    uint32_t start_tile = get_arg_val<uint32_t>(6);
    for (uint32_t i = 0; i < n_act_tiles; i++) {
        uint32_t tile_idx = start_tile + i;

        cb_reserve_back(cb_gate, 1);
        uint32_t l1_addr0 = get_write_ptr(cb_gate);
        noc_async_read_tile(tile_idx, acc_gate, l1_addr0);
        noc_async_read_barrier();
        cb_push_back(cb_gate, 1);

        cb_reserve_back(cb_up, 1);
        uint32_t l1_addr1 = get_write_ptr(cb_up);
        noc_async_read_tile(tile_idx, acc_up, l1_addr1);
        noc_async_read_barrier();
        cb_push_back(cb_up, 1);
    }

    // Phase 2: Read weight tiles from assigned DRAM bank for GEMV
    // (SwiGLU output is already in cb_act from compute kernel)
    uint64_t bank_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, weight_bank_addr);
    uint32_t weight_offset = 0;

    for (uint32_t mt = 0; mt < Mt_per_core; mt++) {
        uint32_t num_blocks = (Kt + BLOCK - 1) / BLOCK;
        uint32_t prev_trid = 0;
        uint32_t prev_batch = 0;

        for (uint32_t blk = 0; blk < num_blocks; blk++) {
            uint32_t batch = ((blk + 1) * BLOCK <= Kt) ? BLOCK : (Kt - blk * BLOCK);
            uint32_t curr_trid = (blk & 1) ? 2 : 1;

            cb_reserve_back(cb_weight, batch);
            uint32_t l1_base = get_write_ptr(cb_weight);

            noc_async_read_set_trid(curr_trid);
            noc_async_read(bank_noc_addr + weight_offset, l1_base, batch * weight_tile_size);
            weight_offset += batch * weight_tile_size;

            if (prev_trid != 0) {
                noc_async_read_barrier_with_trid(prev_trid);
                cb_push_back(cb_weight, prev_batch);
            }

            prev_trid = curr_trid;
            prev_batch = batch;
        }

        if (prev_trid != 0) {
            noc_async_read_barrier_with_trid(prev_trid);
            cb_push_back(cb_weight, prev_batch);
        }
    }
}
