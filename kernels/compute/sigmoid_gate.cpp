// SPDX-License-Identifier: Apache-2.0
// Sigmoid gating compute kernel: output = sigmoid(gate) * attn_out
//
// Used in the attention layer to apply the gate to the attention output.
// Input: attn_out tiles in c_0, gate tiles in c_1
// Output: sigmoid(gate) * attn_out in c_16

#include "api/compute/compute_api.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_compile_time_arg_val(0);

    constexpr uint32_t cb_attn = tt::CBIndex::c_0;   // attention output
    constexpr uint32_t cb_gate = tt::CBIndex::c_1;    // gate values
    constexpr uint32_t cb_out  = tt::CBIndex::c_16;
    constexpr uint32_t cb_sig  = tt::CBIndex::c_2;    // intermediate: sigmoid(gate)

    // Compute sigmoid(gate) into intermediate
    unary_op_init_common(cb_gate, cb_sig);

    for (uint32_t t = 0; t < num_tiles; t++) {
        tile_regs_acquire();
        cb_wait_front(cb_gate, 1);

        copy_tile(cb_gate, 0, 0);
        sigmoid_tile(0);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_sig, 1);
        pack_tile(0, cb_sig);
        cb_push_back(cb_sig, 1);

        cb_pop_front(cb_gate, 1);
        tile_regs_release();
    }

    // Multiply sigmoid(gate) * attn_out
    binary_op_init_common(cb_attn, cb_sig, cb_out);

    for (uint32_t t = 0; t < num_tiles; t++) {
        tile_regs_acquire();
        cb_wait_front(cb_attn, 1);
        cb_wait_front(cb_sig, 1);

        mul_tiles(cb_attn, cb_sig, 0, 0, 0);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        cb_pop_front(cb_attn, 1);
        cb_pop_front(cb_sig, 1);
        tile_regs_release();
    }
}
}  // namespace NAMESPACE
