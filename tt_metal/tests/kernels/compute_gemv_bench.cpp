// GEMV compute benchmark: accumulate matmul_tiles over K dimension.
// Measures how fast compute can consume weight tiles when doing real matmul.
// Compile-time args: [Kt, Mt_per_core]

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Mt = get_compile_time_arg_val(1);

    constexpr uint32_t cb_act    = tt::CBIndex::c_0;  // Kt activation tiles (pre-loaded)
    constexpr uint32_t cb_weight = tt::CBIndex::c_1;
    constexpr uint32_t cb_out    = tt::CBIndex::c_16;

    mm_init(cb_act, cb_weight, cb_out);

    // Wait for all activation tiles
    cb_wait_front(cb_act, Kt);

    for (uint32_t mt = 0; mt < Mt; mt++) {
        acquire_dst();

        for (uint32_t kt = 0; kt < Kt; kt++) {
            cb_wait_front(cb_weight, 1);
            matmul_tiles(cb_act, cb_weight, kt, 0, 0);
            cb_pop_front(cb_weight, 1);
        }

        // Pack accumulated result to output CB
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        release_dst();
    }

    cb_pop_front(cb_act, Kt);
}
