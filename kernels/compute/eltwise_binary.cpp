// SPDX-License-Identifier: Apache-2.0
// Compute kernel: element-wise binary op (add or multiply) on tiles.
// Compile-time arg 0: n_tiles
// Compile-time arg 1: op_type (0 = add, 1 = multiply)

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t n_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t op_type = get_compile_time_arg_val(1);

    constexpr uint32_t cb0    = tt::CBIndex::c_0;
    constexpr uint32_t cb1    = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    binary_op_init_common(cb0, cb1, cb_out);
    if constexpr (op_type == 0) {
        add_tiles_init(cb0, cb1);
    } else {
        mul_tiles_init(cb0, cb1);
    }

    for (uint32_t i = 0; i < n_tiles; i++) {
        acquire_dst();

        cb_wait_front(cb0, 1);
        cb_wait_front(cb1, 1);

        if constexpr (op_type == 0) {
            add_tiles(cb0, cb1, 0, 0, 0);
        } else {
            mul_tiles(cb0, cb1, 0, 0, 0);
        }

        cb_pop_front(cb0, 1);
        cb_pop_front(cb1, 1);

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        release_dst();
    }
}
