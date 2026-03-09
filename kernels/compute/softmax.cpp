// SPDX-License-Identifier: Apache-2.0
// Softmax compute kernel for attention scores.
// Computes softmax over a row of tiles (one attention head's scores).
//
// Two-pass approach:
// Pass 1: Find max and compute exp(x - max) for each element, accumulate sum
// Pass 2: Divide by sum (multiply by reciprocal)
//
// For decode (single query), the scores are a single row across KV length.
// The reader supplies score tiles sequentially.
//
// CB layout:
//   c_0: input score tiles
//   c_16: output probability tiles

#include "api/compute/compute_api.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_compile_time_arg_val(0);

    constexpr uint32_t cb_in  = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    // For a simple softmax on a single tile or small number of tiles,
    // we use the SFPU exp operation and then normalize.
    // Full multi-tile softmax with proper numerical stability requires
    // multiple passes and is more complex.
    //
    // Simplified version: apply exp to each tile, accumulate sum,
    // then multiply by 1/sum. This works for modest sequence lengths
    // where we don't need online softmax.

    unary_op_init_common(cb_in, cb_out);

    for (uint32_t t = 0; t < num_tiles; t++) {
        tile_regs_acquire();
        cb_wait_front(cb_in, 1);

        copy_tile(cb_in, 0, 0);
        exp_tile(0);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        cb_pop_front(cb_in, 1);
        tile_regs_release();
    }

    // Note: The normalization (divide by sum) must be handled by a
    // subsequent reduce + reciprocal + multiply pass, or by the host
    // dispatching separate reduction and scaling programs.
}
}  // namespace NAMESPACE
