// SPDX-License-Identifier: Apache-2.0
// Single-core FPU-based RMSNorm compute kernel.
//
// Phase 1: Square hidden tiles → cb_x2
// Phase 2: REDUCE_SCALAR across all cb_x2 → mean(x^2)
// Phase 3: Add epsilon, rsqrt → norm_factor
// Phase 4: Broadcast-scalar multiply hidden × norm_factor → cb_act
// Phase 5: Element-wise multiply act × norm_weights → cb_out (reuses cb_hidden)
//
// Compile-time args: [Kt]
// Runtime args: none
//
// CBs:
//   c_0  (cb_hidden): Kt tiles of hidden input, reused for normalized output
//   c_2  (cb_norm_w): Kt tiles of norm weights
//   c_24 (cb_x2/cb_act): Kt tiles shared
//   c_4  (cb_var): 1 tile for reduced variance
//   c_5  (cb_scaler): 1 tile with 1/N scaler
//   c_6  (cb_eps): 1 tile with epsilon
//   c_7  (cb_rsqrt): 1 tile for norm_factor

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
    constexpr uint32_t cb_norm_w  = tt::CBIndex::c_2;
    constexpr uint32_t cb_x2      = tt::CBIndex::c_24;
    constexpr uint32_t cb_var     = tt::CBIndex::c_4;
    constexpr uint32_t cb_scaler  = tt::CBIndex::c_5;
    constexpr uint32_t cb_eps     = tt::CBIndex::c_6;
    constexpr uint32_t cb_rsqrt   = tt::CBIndex::c_7;
    constexpr uint32_t cb_act     = tt::CBIndex::c_24;  // alias
    constexpr uint32_t cb_out     = tt::CBIndex::c_1;   // separate output CB

    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);
    cb_wait_front(cb_hidden, Kt);

    // Phase 1: Square all hidden tiles → cb_x2
    constexpr uint32_t BLK = 8;
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

    // Phase 2: Reduce x2 → mean(x^2)
    cb_reserve_back(cb_var, 1);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_x2, cb_scaler, cb_var);
    acquire_dst();
    cb_wait_front(cb_x2, Kt);
    for (uint32_t kt = 0; kt < Kt; kt++)
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_x2, cb_scaler, kt, 0, 0);
    cb_pop_front(cb_x2, Kt);
    reduce_uninit();
    pack_tile(0, cb_var);
    cb_push_back(cb_var, 1);
    release_dst();

    // Phase 3: Add epsilon + rsqrt
    cb_wait_front(cb_var, 1);
    acquire_dst();
    add_tiles_init(cb_var, cb_eps);
    add_tiles(cb_var, cb_eps, 0, 0, 0);
    cb_pop_front(cb_var, 1);
    cb_reserve_back(cb_rsqrt, 1);
    rsqrt_tile_init();
    rsqrt_tile(0);
    pack_tile(0, cb_rsqrt);
    cb_push_back(cb_rsqrt, 1);
    release_dst();

    // Phase 4: hidden × rsqrt → cb_act
    cb_wait_front(cb_rsqrt, 1);
    mul_tiles_bcast_scalar_init_short(cb_hidden, cb_rsqrt);
    for (uint32_t kt = 0; kt < Kt; kt += BLK) {
        uint32_t b = (kt + BLK <= Kt) ? BLK : (Kt - kt);
        cb_reserve_back(cb_act, b);
        acquire_dst();
        for (uint32_t i = 0; i < b; i++)
            mul_tiles_bcast_scalar(cb_hidden, cb_rsqrt, kt + i, 0, i);
        for (uint32_t i = 0; i < b; i++)
            pack_tile(i, cb_act);
        cb_push_back(cb_act, b);
        release_dst();
    }
    cb_pop_front(cb_hidden, Kt);
    cb_pop_front(cb_rsqrt, 1);

    // Phase 5: act × norm_weight → cb_out (reuses cb_hidden)
    cb_wait_front(cb_act, Kt);
    cb_wait_front(cb_norm_w, Kt);
    binary_op_init_common(cb_act, cb_norm_w, cb_out);
    mul_tiles_init(cb_act, cb_norm_w);
    for (uint32_t kt = 0; kt < Kt; kt += BLK) {
        uint32_t b = (kt + BLK <= Kt) ? BLK : (Kt - kt);
        cb_reserve_back(cb_out, b);
        acquire_dst();
        for (uint32_t i = 0; i < b; i++)
            mul_tiles(cb_act, cb_norm_w, kt + i, kt + i, i);
        for (uint32_t i = 0; i < b; i++)
            pack_tile(i, cb_out);
        cb_push_back(cb_out, b);
        release_dst();
    }
    cb_pop_front(cb_act, Kt);
    cb_pop_front(cb_norm_w, Kt);

    // Release constant CBs
    cb_pop_front(cb_scaler, 1);
    cb_pop_front(cb_eps, 1);
}
