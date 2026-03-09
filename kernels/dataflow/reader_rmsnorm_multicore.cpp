// SPDX-License-Identifier: Apache-2.0
// Multi-core RMSNorm: 12 cores each handle their local DRAM bank's tiles.
// Cross-core reduction via NOC writes + semaphores for the norm_factor.
//
// Each core reads tiles at indices: core_id, core_id+N, core_id+2N, ...
// Computes partial sum_sq, participates in reduction, then normalizes.
//
// Compile-time args: [n_tiles, num_cores, acc_in_config, acc_weight_config, acc_out_config]
// Runtime args: [in_addr, weight_addr, out_addr, n_elements, core_id, sem0_addr, sem1_addr,
//                scratch_addr, noc_x_0, noc_y_0, noc_x_1, noc_y_1, ..., noc_x_N-1, noc_y_N-1]
//
// sem0: on core 0, counts partial arrivals from other cores
// sem1: on all non-zero cores, signals norm_factor ready
// scratch_addr: L1 address for inter-core data (must be same on all cores)
//   Core 0: scratch[0..N-1] = partial_sum_sq from each core (uint32_t each)
//   Non-zero cores: scratch[0] = norm_factor (written by core 0)

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

inline float bf16_to_f32(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}

inline uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, 4);
    return static_cast<uint16_t>(bits >> 16);
}

void kernel_main() {
    uint32_t in_addr     = get_arg_val<uint32_t>(0);
    uint32_t weight_addr = get_arg_val<uint32_t>(1);
    uint32_t out_addr    = get_arg_val<uint32_t>(2);
    uint32_t n_elements  = get_arg_val<uint32_t>(3);
    uint32_t core_id     = get_arg_val<uint32_t>(4);
    uint32_t sem0_id     = get_arg_val<uint32_t>(5);
    uint32_t sem1_id     = get_arg_val<uint32_t>(6);
    // Arg 7 reserved

    // Get semaphore L1 addresses from IDs
    uint32_t sem0_addr = get_semaphore(sem0_id);
    uint32_t sem1_addr = get_semaphore(sem1_id);

    // Use CB c_2 for scratch space (inter-core communication)
    // Reserve + push immediately so the CB is balanced for trace replays
    constexpr uint32_t cb_scratch = tt::CBIndex::c_2;
    cb_reserve_back(cb_scratch, 1);
    uint32_t scratch_addr = get_write_ptr(cb_scratch);
    cb_push_back(cb_scratch, 1);

    constexpr uint32_t n_tiles   = get_compile_time_arg_val(0);
    constexpr uint32_t num_cores = get_compile_time_arg_val(1);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_weight = tt::CBIndex::c_1;
    uint32_t tile_size = get_tile_size(cb_in);

    constexpr auto acc_in_args = TensorAccessorArgs<2>();
    const auto acc_in = TensorAccessor(acc_in_args, in_addr, tile_size);
    constexpr auto acc_w_args = TensorAccessorArgs<acc_in_args.next_compile_time_args_offset()>();
    const auto acc_w = TensorAccessor(acc_w_args, weight_addr, tile_size);
    constexpr auto acc_out_args = TensorAccessorArgs<acc_w_args.next_compile_time_args_offset()>();
    const auto acc_out = TensorAccessor(acc_out_args, out_addr, tile_size);

    // Read NOC coordinates for all cores (needed for cross-core communication)
    // Each core gets all N cores' coordinates starting at runtime arg 8
    uint32_t noc_coords[num_cores * 2];
    for (uint32_t i = 0; i < num_cores; i++) {
        noc_coords[i * 2]     = get_arg_val<uint32_t>(8 + i * 2);
        noc_coords[i * 2 + 1] = get_arg_val<uint32_t>(8 + i * 2 + 1);
    }

    // Count how many tiles this core handles
    uint32_t my_tiles = 0;
    for (uint32_t t = core_id; t < n_tiles; t += num_cores)
        my_tiles++;

    // ---- Phase 1: Read local tiles ----
    cb_reserve_back(cb_in, my_tiles);
    uint32_t in_l1_base = get_write_ptr(cb_in);
    uint32_t idx = 0;
    for (uint32_t t = core_id; t < n_tiles; t += num_cores, idx++) {
        noc_async_read_tile(t, acc_in, in_l1_base + idx * tile_size);
    }
    noc_async_read_barrier();

    // ---- Phase 2: Compute partial sum_sq ----
    float partial_sum = 0.0f;
    idx = 0;
    for (uint32_t t = core_id; t < n_tiles; t += num_cores, idx++) {
        volatile tt_l1_ptr uint16_t* d =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_l1_base + idx * tile_size);
        uint32_t base = t * 32;
        for (uint32_t j = 0; j < 16 && (base + j) < n_elements; j++) {
            float v = bf16_to_f32(d[j]);
            partial_sum += v * v;
        }
        for (uint32_t j = 0; j < 16 && (base + 16 + j) < n_elements; j++) {
            float v = bf16_to_f32(d[256 + j]);
            partial_sum += v * v;
        }
    }

    // ---- Phase 3: Cross-core reduction ----
    // Store partial as uint32_t at scratch_addr
    volatile tt_l1_ptr uint32_t* my_scratch =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_addr);
    uint32_t partial_bits;
    __builtin_memcpy(&partial_bits, &partial_sum, 4);

    if (core_id == 0) {
        // Core 0: store own partial at scratch[0]
        my_scratch[0] = partial_bits;

        // Wait for all other cores to send their partials
        volatile tt_l1_ptr uint32_t* sem0 =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem0_addr);
        noc_semaphore_wait(sem0, num_cores - 1);
        noc_semaphore_set(sem0, 0);  // reset for next use

        // Sum all partials
        float total_sum = partial_sum;
        for (uint32_t i = 1; i < num_cores; i++) {
            uint32_t bits = my_scratch[i];
            float val;
            __builtin_memcpy(&val, &bits, 4);
            total_sum += val;
        }

        // Compute norm_factor = 1/sqrt(mean_sq + eps)
        float mean_sq = total_sum / (float)n_elements;
        float val = mean_sq + 1e-6f;
        float x2 = val * 0.5f;
        uint32_t ii;
        __builtin_memcpy(&ii, &val, 4);
        ii = 0x5f3759df - (ii >> 1);
        float norm_factor;
        __builtin_memcpy(&norm_factor, &ii, 4);
        norm_factor = norm_factor * (1.5f - x2 * norm_factor * norm_factor);
        norm_factor = norm_factor * (1.5f - x2 * norm_factor * norm_factor);

        uint32_t norm_bits;
        __builtin_memcpy(&norm_bits, &norm_factor, 4);

        // Store norm_factor locally for own use
        my_scratch[0] = norm_bits;

        // Broadcast norm_factor to all other cores' scratch[0]
        for (uint32_t i = 1; i < num_cores; i++) {
            uint64_t dst = get_noc_addr(noc_coords[i * 2], noc_coords[i * 2 + 1], scratch_addr);
            noc_async_write(scratch_addr, dst, 4);
        }
        noc_async_write_barrier();

        // Signal all other cores that norm_factor is ready
        for (uint32_t i = 1; i < num_cores; i++) {
            uint64_t sem_noc = get_noc_addr(noc_coords[i * 2], noc_coords[i * 2 + 1], sem1_addr);
            noc_semaphore_inc(sem_noc, 1);
        }
    } else {
        // Non-zero core: write partial to core 0's scratch[core_id]
        // First store it at local scratch[0], then NOC write to core 0
        my_scratch[0] = partial_bits;
        uint64_t dst = get_noc_addr(noc_coords[0], noc_coords[1],
                                     scratch_addr + core_id * sizeof(uint32_t));
        noc_async_write(scratch_addr, dst, 4);
        noc_async_write_barrier();

        // Signal core 0
        uint64_t sem0_noc = get_noc_addr(noc_coords[0], noc_coords[1], sem0_addr);
        noc_semaphore_inc(sem0_noc, 1);

        // Wait for norm_factor from core 0
        volatile tt_l1_ptr uint32_t* sem1 =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem1_addr);
        noc_semaphore_wait(sem1, 1);
        noc_semaphore_set(sem1, 0);  // reset
    }

    // ---- Phase 4: Read norm_factor ----
    uint32_t norm_bits = my_scratch[0];
    float norm_factor;
    __builtin_memcpy(&norm_factor, &norm_bits, 4);

    // ---- Phase 5: Read local weight tiles ----
    cb_reserve_back(cb_weight, my_tiles);
    uint32_t w_l1_base = get_write_ptr(cb_weight);
    idx = 0;
    for (uint32_t t = core_id; t < n_tiles; t += num_cores, idx++) {
        noc_async_read_tile(t, acc_w, w_l1_base + idx * tile_size);
    }
    noc_async_read_barrier();

    // ---- Phase 6: Normalize and write output ----
    idx = 0;
    for (uint32_t t = core_id; t < n_tiles; t += num_cores, idx++) {
        volatile tt_l1_ptr uint16_t* hd =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_l1_base + idx * tile_size);
        volatile tt_l1_ptr uint16_t* wd =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(w_l1_base + idx * tile_size);
        uint32_t base = t * 32;

        // Face 0: elements [0..15]
        for (uint32_t j = 0; j < 16 && (base + j) < n_elements; j++) {
            float result = bf16_to_f32(hd[j]) * norm_factor * bf16_to_f32(wd[j]);
            wd[j] = f32_to_bf16(result);
        }
        // Face 2: elements [16..31]
        for (uint32_t j = 0; j < 16 && (base + 16 + j) < n_elements; j++) {
            float result = bf16_to_f32(hd[256 + j]) * norm_factor * bf16_to_f32(wd[256 + j]);
            wd[256 + j] = f32_to_bf16(result);
        }

        // Write output tile (reuse weight L1 slot)
        noc_async_write_tile(t, acc_out, w_l1_base + idx * tile_size);
    }
    noc_async_write_barrier();

    // Release CBs (balanced for trace replay)
    cb_push_back(cb_in, my_tiles);
    cb_wait_front(cb_in, my_tiles);
    cb_pop_front(cb_in, my_tiles);
    cb_push_back(cb_weight, my_tiles);
    cb_wait_front(cb_weight, my_tiles);
    cb_pop_front(cb_weight, my_tiles);
    cb_wait_front(cb_scratch, 1);
    cb_pop_front(cb_scratch, 1);
}
