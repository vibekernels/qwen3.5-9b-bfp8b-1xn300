// SPDX-License-Identifier: Apache-2.0
// Single-core FPU-based RMSNorm compute kernel.
// Computes: output = (x / sqrt(mean(x^2) + eps)) * weight
//
// CB layout:
//   c_0  (cb_hidden): input x [Kt tiles]
//   c_16 (cb_out):    output [Kt tiles]
//   c_2  (cb_norm_w): gamma weights [Kt tiles]
//   c_5  (cb_scaler): 1/N scaler [1 tile]
//   c_6  (cb_eps):    epsilon [1 tile]
//   c_24 (cb_x2):     x^2 intermediate [Kt tiles]
//   c_4  (cb_var):    E[x^2] intermediate [1 tile]
//   c_7  (cb_rsqrt):  1/sqrt(E[x^2]+eps) [1 tile]

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/compute_kernel_api.h"

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_SCALAR
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);

    constexpr uint32_t cb_hidden  = tt::CBIndex::c_0;
    constexpr uint32_t cb_out     = tt::CBIndex::c_16;
    constexpr uint32_t cb_norm_w  = tt::CBIndex::c_2;
    constexpr uint32_t cb_var     = tt::CBIndex::c_4;
    constexpr uint32_t cb_scaler  = tt::CBIndex::c_5;
    constexpr uint32_t cb_eps     = tt::CBIndex::c_6;
    constexpr uint32_t cb_rsqrt   = tt::CBIndex::c_7;
    constexpr uint32_t cb_x2      = tt::CBIndex::c_24;

    constexpr uint32_t BLK = 8;
    constexpr uint32_t dst0 = 0;

    // Initialize packer for output CB
    binary_op_init_common(cb_hidden, cb_hidden, cb_out);

    // Wait for reader to provide all data
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);
    cb_wait_front(cb_hidden, Kt);
    cb_wait_front(cb_norm_w, Kt);

    // ================================================================
    // Step 1: Compute x^2 → cb_x2
    // ================================================================
    mul_tiles_init(cb_hidden, cb_hidden);
    for (uint32_t kt = 0; kt < Kt; kt += BLK) {
        uint32_t b = (kt + BLK <= Kt) ? BLK : (Kt - kt);
        cb_reserve_back(cb_x2, b);
        acquire_dst();
        for (uint32_t i = 0; i < b; i++)
            mul_tiles(cb_hidden, cb_hidden, kt + i, kt + i, i);
        for (uint32_t i = 0; i < b; i++)
            pack_tile(i, cb_x2);
        cb_push_back(cb_x2, b);
        release_dst();
    }

    // ================================================================
    // Step 2: Reduce sum(x^2) * scaler → E[x^2]
    // scaler tile has 1/N in all positions
    // ================================================================
    cb_reserve_back(cb_var, 1);
    reduce_init(cb_x2, cb_scaler, cb_var);
    acquire_dst();
    cb_wait_front(cb_x2, Kt);
    for (uint32_t kt = 0; kt < Kt; kt++) {
        reduce_tile(cb_x2, cb_scaler, kt, 0, dst0);
    }
    cb_pop_front(cb_x2, Kt);
    reduce_uninit();
    pack_tile(dst0, cb_var);
    release_dst();
    cb_push_back(cb_var, 1);

    // ================================================================
    // Step 3: E[x^2] + eps → rsqrt → 1/sqrt(E[x^2] + eps)
    // ================================================================
    cb_wait_front(cb_var, 1);
    cb_reserve_back(cb_rsqrt, 1);
    acquire_dst();
    add_tiles_init(cb_var, cb_eps);
    add_tiles(cb_var, cb_eps, 0, 0, dst0);
    rsqrt_tile_init();
    rsqrt_tile(dst0);
    pack_tile(dst0, cb_rsqrt);
    release_dst();
    cb_push_back(cb_rsqrt, 1);
    cb_pop_front(cb_var, 1);

    // ================================================================
    // Step 4: x * (1/sqrt(E[x^2]+eps)) via broadcast scalar multiply
    // Result stored in cb_x2 (reused)
    // ================================================================
    cb_wait_front(cb_rsqrt, 1);
    mul_tiles_bcast_scalar_init_short(cb_hidden, cb_rsqrt);
    for (uint32_t kt = 0; kt < Kt; kt += BLK) {
        uint32_t b = (kt + BLK <= Kt) ? BLK : (Kt - kt);
        cb_reserve_back(cb_x2, b);
        acquire_dst();
        for (uint32_t i = 0; i < b; i++)
            mul_tiles_bcast_scalar(cb_hidden, cb_rsqrt, kt + i, 0, i);
        for (uint32_t i = 0; i < b; i++)
            pack_tile(i, cb_x2);
        cb_push_back(cb_x2, b);
        release_dst();
    }
    cb_pop_front(cb_hidden, Kt);
    cb_pop_front(cb_rsqrt, 1);

    // ================================================================
    // Step 5: normalized_x * weight → output
    // ================================================================
    cb_wait_front(cb_x2, Kt);
    mul_tiles_init(cb_x2, cb_norm_w);
    for (uint32_t kt = 0; kt < Kt; kt += BLK) {
        uint32_t b = (kt + BLK <= Kt) ? BLK : (Kt - kt);
        cb_reserve_back(cb_out, b);
        acquire_dst();
        for (uint32_t i = 0; i < b; i++)
            mul_tiles(cb_x2, cb_norm_w, kt + i, kt + i, i);
        for (uint32_t i = 0; i < b; i++)
            pack_tile(i, cb_out);
        cb_push_back(cb_out, b);
        release_dst();
    }
    cb_pop_front(cb_x2, Kt);
    cb_pop_front(cb_norm_w, Kt);

    cb_pop_front(cb_scaler, 1);
    cb_pop_front(cb_eps, 1);
}
