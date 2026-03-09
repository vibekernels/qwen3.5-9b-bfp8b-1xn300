// SPDX-License-Identifier: Apache-2.0
// Fused RMSNorm + GEMV compute kernel using Tensix FPU.
//
// Phase 1: RMSNorm
//   a. Square hidden tiles → cb_x2 (all Kt tiles)
//   b. REDUCE_SCALAR across all cb_x2 tiles → sum(x^2) * (1/N)
//   c. Add epsilon, rsqrt → norm_factor
//   d. Broadcast-scalar multiply hidden by norm_factor → temp
//   e. Element-wise multiply temp by norm_weights → cb_act
//
// Phase 2: GEMV matmul using cb_act as normalized activations
//
// Compile-time args: [Kt, BLOCK]
// Runtime args: [Mt_per_core, n_elements]
//
// CBs:
//   c_0  (cb_hidden): Kt tiles of hidden input
//   c_1  (cb_weight): GEMV weight tiles (streamed, double-buffered)
//   c_2  (cb_norm_w): Kt tiles of norm weights
//   c_24 (cb_x2/cb_act): Kt tiles (shared: x^2 during reduce, normalized during GEMV)
//   c_4  (cb_var):    1 tile for reduced variance
//   c_5  (cb_scaler): 1 tile with 1/N scaler for reduce
//   c_6  (cb_eps):    1 tile with epsilon value
//   c_7  (cb_rsqrt):  1 tile for norm_factor (1/sqrt(var+eps))
//   c_16 (cb_out):    GEMV output tiles

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/compute_kernel_api.h"

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_SCALAR
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"

void kernel_main() {
    uint32_t Mt = get_arg_val<uint32_t>(0);

    constexpr uint32_t Kt    = get_compile_time_arg_val(0);
    constexpr uint32_t BLOCK = get_compile_time_arg_val(1);

    constexpr uint32_t cb_hidden  = tt::CBIndex::c_0;
    constexpr uint32_t cb_weight  = tt::CBIndex::c_1;
    constexpr uint32_t cb_norm_w  = tt::CBIndex::c_2;
    constexpr uint32_t cb_x2      = tt::CBIndex::c_24;  // shared with cb_act
    constexpr uint32_t cb_var     = tt::CBIndex::c_4;
    constexpr uint32_t cb_scaler  = tt::CBIndex::c_5;
    constexpr uint32_t cb_eps     = tt::CBIndex::c_6;
    constexpr uint32_t cb_rsqrt   = tt::CBIndex::c_7;
    constexpr uint32_t cb_act     = tt::CBIndex::c_24;  // alias for cb_x2
    constexpr uint32_t cb_out     = tt::CBIndex::c_16;

    // Wait for constant tiles from reader
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);

    // Wait for all hidden tiles
    cb_wait_front(cb_hidden, Kt);

    // ======== Phase 1a: Square all hidden tiles → cb_x2 ========
    constexpr uint32_t SQ_BLK = 8;
    mul_tiles_init(cb_hidden, cb_hidden);
    for (uint32_t kt = 0; kt < Kt; kt += SQ_BLK) {
        uint32_t blk = (kt + SQ_BLK <= Kt) ? SQ_BLK : (Kt - kt);
        cb_reserve_back(cb_x2, blk);
        acquire_dst();
        for (uint32_t i = 0; i < blk; i++) {
            mul_tiles(cb_hidden, cb_hidden, kt + i, kt + i, i);
        }
        for (uint32_t i = 0; i < blk; i++) {
            pack_tile(i, cb_x2);
        }
        cb_push_back(cb_x2, blk);
        release_dst();
    }

    // ======== Phase 1b: Reduce x2 → sum(x^2) * (1/N) ========
    // Scaler has 1/N, so reduce gives mean(x^2) directly.
    cb_reserve_back(cb_var, 1);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_x2, cb_scaler, cb_var);
    acquire_dst();

    cb_wait_front(cb_x2, Kt);
    for (uint32_t kt = 0; kt < Kt; kt++) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_x2, cb_scaler, kt, 0, 0);
    }
    cb_pop_front(cb_x2, Kt);  // free x2 tiles (cb_x2 now empty, can be reused as cb_act)
    reduce_uninit();

    // dst[0] = mean(x^2)
    pack_tile(0, cb_var);
    cb_push_back(cb_var, 1);
    release_dst();

    // ======== Phase 1c: Add epsilon + rsqrt ========
    cb_wait_front(cb_var, 1);

    acquire_dst();
    add_tiles_init(cb_var, cb_eps);
    add_tiles(cb_var, cb_eps, 0, 0, 0);  // dst[0] = mean(x^2) + eps
    cb_pop_front(cb_var, 1);

    cb_reserve_back(cb_rsqrt, 1);
    rsqrt_tile_init();
    rsqrt_tile(0);  // dst[0] = 1/sqrt(mean(x^2) + eps)
    pack_tile(0, cb_rsqrt);
    cb_push_back(cb_rsqrt, 1);
    release_dst();

    // ======== Phase 1d: Normalize: act = hidden * rsqrt * norm_weight ========
    cb_wait_front(cb_rsqrt, 1);
    cb_wait_front(cb_norm_w, Kt);

    // Step 1: hidden * rsqrt (scalar broadcast) → cb_act (same physical CB as cb_x2, now empty)
    mul_tiles_bcast_scalar_init_short(cb_hidden, cb_rsqrt);
    for (uint32_t kt = 0; kt < Kt; kt += SQ_BLK) {
        uint32_t blk = (kt + SQ_BLK <= Kt) ? SQ_BLK : (Kt - kt);
        cb_reserve_back(cb_act, blk);
        acquire_dst();
        for (uint32_t i = 0; i < blk; i++) {
            mul_tiles_bcast_scalar(cb_hidden, cb_rsqrt, kt + i, 0, i);
        }
        for (uint32_t i = 0; i < blk; i++) {
            pack_tile(i, cb_act);
        }
        cb_push_back(cb_act, blk);
        release_dst();
    }

    // Release hidden (no longer needed) and rsqrt
    cb_pop_front(cb_hidden, Kt);
    cb_pop_front(cb_rsqrt, 1);

    // Step 2: act * norm_weight → overwrite cb_act in-place
    // Read from cb_act, multiply by cb_norm_w, write to cb_var (temp), then move back
    // Actually, we can't read and write the same CB simultaneously.
    // Instead: pop from cb_act, multiply, push back to cb_act.
    // But after pop, the space is freed and push reuses it.
    // We process tile-by-tile: pop 1 from cb_act, multiply with norm_w[kt], push 1 to cb_act.
    // Wait: this doesn't work because we pop from the front and push to the back.
    // After popping all and pushing all, we'd need the CB to wrap around correctly.
    //
    // Better approach: use cb_hidden (now freed) as temp buffer.
    // Read from cb_act → multiply by norm_w → write to cb_hidden.
    // Then use cb_hidden as the activation for GEMV.

    constexpr uint32_t cb_norm_out = cb_hidden;  // reuse freed cb_hidden for normalized output

    cb_wait_front(cb_act, Kt);
    binary_op_init_common(cb_act, cb_norm_w, cb_norm_out);
    mul_tiles_init(cb_act, cb_norm_w);
    for (uint32_t kt = 0; kt < Kt; kt += SQ_BLK) {
        uint32_t blk = (kt + SQ_BLK <= Kt) ? SQ_BLK : (Kt - kt);
        cb_reserve_back(cb_norm_out, blk);
        acquire_dst();
        for (uint32_t i = 0; i < blk; i++) {
            mul_tiles(cb_act, cb_norm_w, kt + i, kt + i, i);
        }
        for (uint32_t i = 0; i < blk; i++) {
            pack_tile(i, cb_norm_out);
        }
        cb_push_back(cb_norm_out, blk);
        release_dst();
    }

    // Release intermediate buffers
    cb_pop_front(cb_act, Kt);
    cb_pop_front(cb_norm_w, Kt);

    // ======== Phase 2: GEMV matmul ========
    // Use cb_norm_out (= cb_hidden = c_0) as activations
    constexpr uint32_t num_blocks = (Kt + BLOCK - 1) / BLOCK;

    mm_init(cb_norm_out, cb_weight, cb_out);
    cb_wait_front(cb_norm_out, Kt);

    for (uint32_t mt = 0; mt < Mt; mt++) {
        acquire_dst();

        for (uint32_t blk = 0; blk < num_blocks; blk++) {
            constexpr uint32_t full_blocks = Kt / BLOCK;
            uint32_t batch = (blk < full_blocks) ? BLOCK : (Kt - blk * BLOCK);

            cb_wait_front(cb_weight, batch);

            for (uint32_t b = 0; b < batch; b++) {
                matmul_tiles(cb_norm_out, cb_weight, blk * BLOCK + b, b, 0);
            }

            cb_pop_front(cb_weight, batch);
        }

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        release_dst();
    }

    cb_pop_front(cb_norm_out, Kt);

    // Release scaler and eps CBs (balanced for trace replay)
    cb_pop_front(cb_scaler, 1);
    cb_pop_front(cb_eps, 1);
}
