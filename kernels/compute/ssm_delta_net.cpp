// SPDX-License-Identifier: Apache-2.0
// Delta-net linear attention compute kernel for SSM layers.
//
// For single-token decode, each Tensix core handles one v_head.
// The 128x128 state matrix fits in L1 (64KB in f32).
//
// Per-head computation (v_head):
//   1. Load state[128,128] from L1 circular buffer
//   2. g_exp = exp(softplus(alpha + dt_bias) * ssm_a)
//   3. state *= g_exp  (decay)
//   4. sk = state^T @ k  (matrix-vector product)
//   5. d = beta * (v - sk)  (delta)
//   6. state += k outer d  (rank-1 update)
//   7. output = state^T @ q  (query)
//   8. Write state back
//   9. Apply gated RMSNorm: rmsnorm(output) * silu(z)
//
// This kernel uses the FPU matmul for the matrix-vector products
// and SFPU for exp, sigmoid, silu, rsqrt operations.
//
// CB layout:
//   c_0: state tiles [head_k_dim/32, head_v_dim/32] = [4,4] = 16 tiles
//   c_1: q,k,v vectors (as tiles)
//   c_2: gate/beta/alpha scalars
//   c_3: norm weights
//   c_16: output tiles
//   c_17: updated state tiles (written back)

#include "api/compute/compute_api.h"

namespace NAMESPACE {
void MAIN {
    // This is a placeholder for the full delta-net kernel.
    // The actual implementation requires:
    // 1. Multiple matmul_tiles calls for state @ k and state @ q
    // 2. SFPU operations for exp, sigmoid, silu, rsqrt
    // 3. Careful synchronization between tile operations
    //
    // For the initial bring-up, we'll implement a simplified version
    // that processes one element at a time using SFPU operations,
    // then optimize with tiled matmul once correctness is verified.
    //
    // The key challenge is that the state matrix [128,128] is
    // [4 tiles, 4 tiles] = 16 tiles, which fits in L1 (32KB in bf16).
    // We can keep it resident and iterate over it.

    uint32_t head_k_tiles = get_compile_time_arg_val(0);  // 4
    uint32_t head_v_tiles = get_compile_time_arg_val(1);  // 4
    uint32_t state_tiles = head_k_tiles * head_v_tiles;   // 16

    constexpr uint32_t cb_state = tt::CBIndex::c_0;
    constexpr uint32_t cb_qkv   = tt::CBIndex::c_1;
    constexpr uint32_t cb_out   = tt::CBIndex::c_16;

    // For initial bring-up: just pass state through unchanged
    // This validates the data movement path
    for (uint32_t t = 0; t < state_tiles; t++) {
        tile_regs_acquire();
        cb_wait_front(cb_state, 1);

        copy_tile(cb_state, 0, 0);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        cb_pop_front(cb_state, 1);
        tile_regs_release();
    }
}
}  // namespace NAMESPACE
