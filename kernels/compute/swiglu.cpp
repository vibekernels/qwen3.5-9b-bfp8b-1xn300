// SPDX-License-Identifier: Apache-2.0
// SwiGLU compute kernel: output = SiLU(gate) * up
// Single fused pass: for each tile, SiLU(gate) → c_2, then c_2 * up → c_16
// This avoids deadlock with the interleaved reader that pushes gate/up tiles in lockstep.
// Compile-time args: [num_tiles]

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);

    constexpr uint32_t cb_gate = tt::CBIndex::c_0;
    constexpr uint32_t cb_up   = tt::CBIndex::c_1;
    constexpr uint32_t cb_out  = tt::CBIndex::c_16;
    constexpr uint32_t cb_silu = tt::CBIndex::c_2;

    for (uint32_t t = 0; t < num_tiles; t++) {
        // Step 1: SiLU(gate) → cb_silu
        copy_tile_init(cb_gate);
        silu_tile_init();

        acquire_dst();
        cb_wait_front(cb_gate, 1);

        copy_tile(cb_gate, 0, 0);
        silu_tile(0);

        cb_reserve_back(cb_silu, 1);
        pack_tile(0, cb_silu);
        cb_push_back(cb_silu, 1);

        cb_pop_front(cb_gate, 1);
        release_dst();

        // Step 2: cb_silu * cb_up → cb_out
        binary_op_init_common(cb_silu, cb_up, cb_out);
        mul_tiles_init(cb_silu, cb_up);

        acquire_dst();
        cb_wait_front(cb_silu, 1);
        cb_wait_front(cb_up, 1);

        mul_tiles(cb_silu, cb_up, 0, 0, 0);

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        cb_pop_front(cb_silu, 1);
        cb_pop_front(cb_up, 1);
        release_dst();
    }
}
