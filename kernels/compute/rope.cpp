// SPDX-License-Identifier: Apache-2.0
// RoPE (Rotary Position Embedding) compute kernel.
//
// Applies rotation to the first rope_dim=64 dimensions of each head.
// For each pair (i, i+32) in the head:
//   x0' = x0 * cos(theta) - x1 * sin(theta)
//   x1' = x1 * cos(theta) + x0 * sin(theta)
// where theta = pos * freq, freq = 1 / (base ^ (2*i / rope_dim))
//
// On Tensix, we use SFPU sin/cos operations per tile element.
// The position and frequency must be passed as tile data.
//
// CB layout:
//   c_0: QK tiles [n_heads, head_dim] organized as tiles
//   c_1: cos(theta) tiles (precomputed by host)
//   c_2: sin(theta) tiles (precomputed by host)
//   c_16: rotated QK output tiles
//
// Strategy: precompute cos/sin tables on host, upload as tile data,
// then use FPU multiply and add operations:
//   out0 = x0 * cos - x1 * sin
//   out1 = x1 * cos + x0 * sin

#include "api/compute/compute_api.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_compile_time_arg_val(0);

    constexpr uint32_t cb_qk  = tt::CBIndex::c_0;   // input Q or K tiles
    constexpr uint32_t cb_cos = tt::CBIndex::c_1;    // cos(theta) tiles
    constexpr uint32_t cb_sin = tt::CBIndex::c_2;    // sin(theta) tiles
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    // For initial bring-up: pass through unchanged
    // RoPE will be applied once we have the cos/sin precomputation working
    unary_op_init_common(cb_qk, cb_out);

    for (uint32_t t = 0; t < num_tiles; t++) {
        tile_regs_acquire();
        cb_wait_front(cb_qk, 1);

        copy_tile(cb_qk, 0, 0);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        cb_pop_front(cb_qk, 1);
        tile_regs_release();
    }
}
}  // namespace NAMESPACE
