// Drain compute kernel: just consume weight tiles from CB as fast as possible
// to measure reader+compute CB pipeline overhead without matmul.
// Compile-time args: [total_tiles]
// Runtime args: none

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"

void kernel_main() {
    constexpr uint32_t total_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight = tt::CBIndex::c_0;

    for (uint32_t t = 0; t < total_tiles; t++) {
        cb_wait_front(cb_weight, 1);
        cb_pop_front(cb_weight, 1);
    }
}
