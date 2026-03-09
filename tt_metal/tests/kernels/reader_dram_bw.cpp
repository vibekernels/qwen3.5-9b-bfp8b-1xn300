// Pure DRAM read bandwidth test: one core per bank, read as fast as possible
// No compute, just measure raw read throughput.
// Compile-time args: [total_bytes_per_core, block_bytes]
// Runtime args: [bank_id, weight_bank_addr]

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    uint32_t bank_id          = get_arg_val<uint32_t>(0);
    uint32_t weight_bank_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t total_bytes  = get_compile_time_arg_val(0);
    constexpr uint32_t block_bytes  = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id        = tt::CBIndex::c_0;

    uint32_t tile_size = get_tile_size(cb_id);

    uint64_t bank_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, weight_bank_addr);

    uint32_t tiles_per_block = block_bytes / tile_size;
    uint32_t num_blocks = total_bytes / block_bytes;
    uint32_t offset = 0;

    // Use TRID pipelining for maximum throughput
    uint32_t prev_trid = 0;
    uint32_t prev_tiles = 0;

    for (uint32_t blk = 0; blk < num_blocks; blk++) {
        uint32_t curr_trid = (blk & 1) ? 2 : 1;

        cb_reserve_back(cb_id, tiles_per_block);
        uint32_t l1_base = get_write_ptr(cb_id);

        noc_async_read_set_trid(curr_trid);
        noc_async_read(bank_noc_addr + offset, l1_base, block_bytes);
        offset += block_bytes;

        if (prev_trid != 0) {
            noc_async_read_barrier_with_trid(prev_trid);
            cb_push_back(cb_id, prev_tiles);
            cb_wait_front(cb_id, prev_tiles);
            cb_pop_front(cb_id, prev_tiles);
        }

        prev_trid = curr_trid;
        prev_tiles = tiles_per_block;
    }

    // Flush last block
    if (prev_trid != 0) {
        noc_async_read_barrier_with_trid(prev_trid);
        cb_push_back(cb_id, prev_tiles);
        cb_wait_front(cb_id, prev_tiles);
        cb_pop_front(cb_id, prev_tiles);
    }
}
