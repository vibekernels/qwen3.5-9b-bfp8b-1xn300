// SPDX-License-Identifier: Apache-2.0
// Gated RMSNorm compute kernel: output = rmsnorm(input, weight) * silu(gate)
//
// Used in SSM layers after delta-net attention.
// input: [num_heads, head_dim] — delta-net output
// weight: [head_dim] — shared norm weights
// gate: [num_heads, head_dim] — z gate values
//
// CB layout:
//   c_0: input tiles (delta-net output)
//   c_1: weight tiles (norm weights, broadcast)
//   c_2: gate tiles (z)
//   c_16: output tiles

#include "api/compute/compute_api.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_compile_time_arg_val(0);

    constexpr uint32_t cb_in   = tt::CBIndex::c_0;
    constexpr uint32_t cb_w    = tt::CBIndex::c_1;
    constexpr uint32_t cb_gate = tt::CBIndex::c_2;
    constexpr uint32_t cb_out  = tt::CBIndex::c_16;
    constexpr uint32_t cb_tmp  = tt::CBIndex::c_3;  // intermediate

    // Step 1: Compute SiLU(gate) into tmp
    unary_op_init_common(cb_gate, cb_tmp);
    for (uint32_t t = 0; t < num_tiles; t++) {
        tile_regs_acquire();
        cb_wait_front(cb_gate, 1);
        copy_tile(cb_gate, 0, 0);
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_tmp, 1);
        pack_tile(0, cb_tmp);
        cb_push_back(cb_tmp, 1);
        cb_pop_front(cb_gate, 1);
        tile_regs_release();
    }

    // Step 2: Multiply input * weight (simplified rmsnorm - scale is baked in)
    // Then multiply by silu(gate)
    // For full rmsnorm we'd need reduction, but the scale factor
    // can be precomputed on the host for single-token decode.
    binary_op_init_common(cb_in, cb_w, cb_out);
    for (uint32_t t = 0; t < num_tiles; t++) {
        tile_regs_acquire();
        cb_wait_front(cb_in, 1);
        cb_wait_front(cb_w, 1);

        // input * weight (rmsnorm component)
        mul_tiles(cb_in, cb_w, 0, 0, 0);

        tile_regs_commit();
        tile_regs_wait();

        // Now multiply by silu(gate)
        // We need to read from cb_tmp and multiply
        // For simplicity, pack intermediate result first
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        cb_pop_front(cb_in, 1);
        cb_pop_front(cb_w, 1);
        tile_regs_release();
    }

    // Step 3: Final multiply: rmsnorm_result * silu_gate
    // Read back from cb_out and cb_tmp, multiply, write to cb_out
    // This requires a second output CB or in-place operation
    // For now this is a placeholder - full implementation needs
    // careful CB management with additional intermediate buffers.
}
}  // namespace NAMESPACE
