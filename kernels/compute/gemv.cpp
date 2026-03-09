// SPDX-License-Identifier: Apache-2.0
// Compute kernel for GEMV: accumulates matmul_tiles over K dimension.
// Activation tiles are pre-loaded in cb_act (Kt tiles, not consumed until end).
// Weight tiles consumed in BLOCK-sized batches matching reader's TRID pipelining.
// Compile-time args: [Kt, BLOCK]
// Runtime args: [Mt_per_core]

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"

void kernel_main() {
    uint32_t Mt = get_arg_val<uint32_t>(0);
    constexpr uint32_t Kt    = get_compile_time_arg_val(0);
    constexpr uint32_t BLOCK = get_compile_time_arg_val(1);

    constexpr uint32_t cb_act    = tt::CBIndex::c_0;
    constexpr uint32_t cb_weight = tt::CBIndex::c_1;
    constexpr uint32_t cb_out    = tt::CBIndex::c_16;

    // transpose=1: compute C = A × B^T so GEMV y = W @ x works correctly
    // (weight tiles stored as W[M,K], need B^T[k][j] = W[j][k])
    mm_init(cb_act, cb_weight, cb_out, /*transpose=*/1);

    // Wait for all activation tiles to be loaded by reader
    cb_wait_front(cb_act, Kt);

    constexpr uint32_t num_blocks = (Kt + BLOCK - 1) / BLOCK;

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

        // Pack accumulated result to output CB
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        release_dst();
    }

    // Release activation tiles now that all output rows are done
    cb_pop_front(cb_act, Kt);
}
