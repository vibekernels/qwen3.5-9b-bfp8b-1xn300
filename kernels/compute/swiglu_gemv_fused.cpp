// SPDX-License-Identifier: Apache-2.0
// Fused SwiGLU + GEMV compute: first does SwiGLU(gate, up) -> activation tiles,
// then does GEMV matmul using those activations.
// Runtime args: [n_act_tiles, Mt_per_core]

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    uint32_t n_act_tiles = get_arg_val<uint32_t>(0);
    uint32_t Mt          = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_gate   = tt::CBIndex::c_0;   // gate input
    constexpr uint32_t cb_up     = tt::CBIndex::c_1;   // up input
    constexpr uint32_t cb_silu   = tt::CBIndex::c_2;   // intermediate SiLU
    constexpr uint32_t cb_weight = tt::CBIndex::c_3;   // GEMV weights
    constexpr uint32_t cb_act    = tt::CBIndex::c_4;   // SwiGLU output = GEMV activations
    constexpr uint32_t cb_out    = tt::CBIndex::c_16;  // GEMV output

    // Phase 1: SwiGLU → produces n_act_tiles into cb_act
    for (uint32_t t = 0; t < n_act_tiles; t++) {
        // SiLU(gate) → cb_silu
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

        // cb_silu * cb_up → cb_act
        binary_op_init_common(cb_silu, cb_up, cb_act);
        mul_tiles_init(cb_silu, cb_up);

        acquire_dst();
        cb_wait_front(cb_silu, 1);
        cb_wait_front(cb_up, 1);
        mul_tiles(cb_silu, cb_up, 0, 0, 0);
        cb_reserve_back(cb_act, 1);
        pack_tile(0, cb_act);
        cb_push_back(cb_act, 1);
        cb_pop_front(cb_silu, 1);
        cb_pop_front(cb_up, 1);
        release_dst();
    }

    // Phase 2: GEMV matmul using cb_act as activations, cb_weight as weights
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t BLOCK = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = (Kt + BLOCK - 1) / BLOCK;

    mm_init(cb_act, cb_weight, cb_out);
    cb_wait_front(cb_act, Kt);

    for (uint32_t mt = 0; mt < Mt; mt++) {
        acquire_dst();

        for (uint32_t blk = 0; blk < num_blocks; blk++) {
            constexpr uint32_t full_blocks = Kt / BLOCK;
            uint32_t batch = (blk < full_blocks) ? BLOCK : (Kt - blk * BLOCK);

            cb_wait_front(cb_weight, batch);

            for (uint32_t b = 0; b < batch; b++) {
                matmul_tiles(cb_act, cb_weight, blk * BLOCK + b, b, 0);
            }

            cb_pop_front(cb_weight, batch);
        }

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        release_dst();
    }

    cb_pop_front(cb_act, Kt);
}
