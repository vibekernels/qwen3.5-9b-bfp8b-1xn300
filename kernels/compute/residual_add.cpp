// SPDX-License-Identifier: Apache-2.0
// Residual add compute kernel: output = a + b (tile-wise addition)
// Used for adding residual connections in the transformer.
//
// Input: tiles from c_0 and c_1
// Output: sum tiles in c_16

#include "api/compute/compute_api.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_compile_time_arg_val(0);

    constexpr uint32_t cb_a   = tt::CBIndex::c_0;
    constexpr uint32_t cb_b   = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    binary_op_init_common(cb_a, cb_b, cb_out);

    for (uint32_t t = 0; t < num_tiles; t++) {
        tile_regs_acquire();
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);

        add_tiles(cb_a, cb_b, 0, 0, 0);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
        tile_regs_release();
    }
}
}  // namespace NAMESPACE
