#include "model.h"
#include "utils.h"
#include "gguf_loader.h"
#include "tokenizer.h"
#include "sampling.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>

// Forward declarations for kernel launchers
// (embedding_to_f32 is defined inline in this file)
void launch_rmsnorm(__nv_bfloat16* output, const __nv_bfloat16* input,
    const float* weight, int n_tokens, int dim, float eps, cudaStream_t stream);
void launch_rmsnorm_f32in(__nv_bfloat16* output, const float* input,
    const float* weight, int n_tokens, int dim, float eps, cudaStream_t stream);
void launch_fused_residual_rmsnorm(__nv_bfloat16* norm_output, float* hidden,
    const float* residual, const float* weight, int n_tokens, int dim, float eps, cudaStream_t stream);
void launch_fused_bf16_residual_rmsnorm(__nv_bfloat16* norm_output, float* hidden,
    const __nv_bfloat16* residual_bf16, const float* weight, int n_tokens, int dim, float eps, cudaStream_t stream);
void launch_bf16_residual_add(float* hidden, const __nv_bfloat16* residual_bf16, int n, cudaStream_t stream);
void launch_rmsnorm_head(__nv_bfloat16* output, const __nv_bfloat16* input,
    const float* weight, int n_tokens, int n_heads, int head_dim,
    float eps, cudaStream_t stream);
void launch_rope(__nv_bfloat16* qk, const int* positions, int n_tokens,
    int n_heads, int head_dim, int rope_dim, float freq_base, cudaStream_t stream);
void launch_swiglu(__nv_bfloat16* output, const __nv_bfloat16* gate,
    const __nv_bfloat16* up, int n_tokens, int n_ff, cudaStream_t stream);
void launch_swiglu_packed(__nv_bfloat16* output, const __nv_bfloat16* packed,
    int n_tokens, int n_ff, cudaStream_t stream);
void launch_sigmoid_mul(__nv_bfloat16* output, const __nv_bfloat16* attn_out,
    const __nv_bfloat16* gate, int n_elements, cudaStream_t stream);
void launch_kv_cache_append(__nv_bfloat16* cache, const __nv_bfloat16* new_kv,
    const int* d_kv_pos, int n_new_tokens, int kv_dim, cudaStream_t stream);
void launch_attention_decode(__nv_bfloat16* output, const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
    const int* d_kv_len, int max_kv_len, int n_head, int n_head_kv, int head_dim, float scale, cudaStream_t stream);
void launch_attention_prefill(__nv_bfloat16* output, const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
    int n_tokens, int kv_start, int n_head, int n_head_kv, int head_dim, float scale, cudaStream_t stream);
void launch_compute_gate(float* gate_out, const float* alpha,
    const float* dt_bias, const float* ssm_a, int n_tokens, int num_v_heads, cudaStream_t stream);
void launch_sigmoid(float* output, const float* input, int n, cudaStream_t stream);
void launch_conv1d_silu(float* output, const float* input,
    const float* conv_state, const float* conv_weight,
    int n_tokens, int channels, int conv_kernel_size, cudaStream_t stream);
void launch_update_conv_state(float* new_state, const float* input,
    const float* old_state, int n_tokens, int channels, int conv_kernel_size, cudaStream_t stream);
void launch_conv1d_silu_update(float* output, float* conv_state, const float* input,
    const float* conv_weight, int channels, int conv_kernel_size, cudaStream_t stream);
void launch_l2_norm(float* output, const float* input,
    int n_vectors, int dim, float eps, cudaStream_t stream);
void launch_delta_net_decode(float* output, float* state,
    const float* q, const float* k, const float* v,
    const float* gate, const float* beta, int num_v_heads, int head_dim,
    float scale, cudaStream_t stream);
void launch_gated_rmsnorm(__nv_bfloat16* output, const float* input,
    const float* weight, const float* gate,
    int num_heads, int head_dim, float eps, cudaStream_t stream);
void launch_repeat_heads(float* output, const float* input,
    int num_k_heads, int num_v_heads, int head_dim, cudaStream_t stream);
void launch_fused_ssm_step(__nv_bfloat16* output, float* state,
    const float* conv_out, const float* alpha, const float* dt_bias,
    const float* ssm_a, const float* beta_raw, const float* z,
    const float* norm_weight,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    float scale, float l2_eps, float rms_eps, cudaStream_t stream);
void launch_fused_ssm_step_batched(__nv_bfloat16* output, float* state,
    const float* conv_out, const float* alpha, const float* dt_bias,
    const float* ssm_a, const float* beta_raw, const float* z,
    const float* norm_weight, int n_tokens,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int conv_channels, int d_inner,
    float scale, float l2_eps, float rms_eps, cudaStream_t stream);

using MC = ModelConfig;

// Vectorized GEMV: y[N] = W[N,K] @ x[K], bf16 weights × bf16 input → bf16/f32 output
// L2-cache optimized: x vector (~8KB) stays in L2, no shared memory needed.
// Each warp processes one row. 8 warps/block = 8 rows/block.
// Requires K divisible by 8 (true for all our GEMM sizes).
template<bool F32_OUTPUT>
__global__ void fast_gemv_kernel(
    void* __restrict__ y,
    const __nv_bfloat16* __restrict__ W,  // [N, K] row-major
    const __nv_bfloat16* __restrict__ x,  // [K]
    int N, int K
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int row = blockIdx.x * 8 + warp_id;

    if (row >= N) return;

    const int K8 = K / 8;
    const uint4* W_v = reinterpret_cast<const uint4*>(&W[(int64_t)row * K]);
    const uint4* x_v = reinterpret_cast<const uint4*>(x);
    float acc = 0.0f;

    for (int i = lane_id; i < K8; i += 32) {
        uint4 wc = __ldcs(&W_v[i]);  // streaming load: weights are read-once, don't pollute L2
        uint4 xc = x_v[i];
        const __nv_bfloat162* wb2 = reinterpret_cast<const __nv_bfloat162*>(&wc);
        const __nv_bfloat162* xb2 = reinterpret_cast<const __nv_bfloat162*>(&xc);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 p = __bfloat1622float2(__hmul2(wb2[j], xb2[j]));
            acc += p.x + p.y;
        }
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane_id == 0) {
        if constexpr (F32_OUTPUT) {
            reinterpret_cast<float*>(y)[row] = acc;
        } else {
            reinterpret_cast<__nv_bfloat16*>(y)[row] = __float2bfloat16(acc);
        }
    }
}

static void launch_gemv_bf16(__nv_bfloat16* y, const __nv_bfloat16* W, const __nv_bfloat16* x,
    int N, int K, cudaStream_t stream) {
    int blocks = cdiv(N, 8);
    fast_gemv_kernel<false><<<blocks, 256, 0, stream>>>(y, W, x, N, K);
}

static void launch_gemv_bf16_f32out(float* y, const __nv_bfloat16* W, const __nv_bfloat16* x,
    int N, int K, cudaStream_t stream) {
    int blocks = cdiv(N, 8);
    fast_gemv_kernel<true><<<blocks, 256, 0, stream>>>(y, W, x, N, K);
}

// ============================================================================
// Tiled GEMM using wmma tensor cores for M>1 (prompt eval)
// C[M,N] = A[M,K] @ B[N,K]^T, all bf16, output bf16 or f32
//
// Block tile: BM=64, BN=64, BK=16.
// 4 warps (2×2), each warp computes 32×32 via 2×2 wmma 16×16 tiles.
// ============================================================================
using namespace nvcuda;

template<bool F32_OUTPUT>
__global__ void tiled_gemm_kernel(
    void* __restrict__ C,
    const __nv_bfloat16* __restrict__ A,  // [M, K] row-major
    const __nv_bfloat16* __restrict__ B,  // [N, K] row-major (transposed access)
    int M, int N, int K
) {
    static constexpr int BM = 64, BN = 64, BK = 16;

    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;

    // 4 warps arranged 2×2
    const int warp_id = threadIdx.x / 32;
    const int warp_row = warp_id / 2;  // 0 or 1
    const int warp_col = warp_id % 2;  // 0 or 1
    const int tid = threadIdx.x;

    // Shared memory for tiles (4KB total)
    __shared__ __nv_bfloat16 As[BM][BK];  // 64×16 = 2KB
    __shared__ __nv_bfloat16 Bs[BN][BK];  // 64×16 = 2KB

    // 2×2 = 4 wmma accumulators per warp (32×32 output)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    // K loop
    for (int bk = 0; bk < K; bk += BK) {
        // Load A tile [BM, BK] — 128 threads load 64×16 = 1024 elements = 128 uint4's
        for (int i = tid; i < BM * (BK / 8); i += 128) {
            int row = i / (BK / 8);
            int col8 = i % (BK / 8);
            int gr = bm + row;
            if (gr < M) {
                *reinterpret_cast<uint4*>(&As[row][col8 * 8]) =
                    *reinterpret_cast<const uint4*>(&A[(int64_t)gr * K + bk + col8 * 8]);
            } else {
                uint4 z = {0, 0, 0, 0};
                *reinterpret_cast<uint4*>(&As[row][col8 * 8]) = z;
            }
        }
        // Load B tile [BN, BK]
        for (int i = tid; i < BN * (BK / 8); i += 128) {
            int row = i / (BK / 8);
            int col8 = i % (BK / 8);
            int gr = bn + row;
            if (gr < N) {
                *reinterpret_cast<uint4*>(&Bs[row][col8 * 8]) =
                    *reinterpret_cast<const uint4*>(&B[(int64_t)gr * K + bk + col8 * 8]);
            } else {
                uint4 z = {0, 0, 0, 0};
                *reinterpret_cast<uint4*>(&Bs[row][col8 * 8]) = z;
            }
        }
        __syncthreads();

        // Each warp computes its 2×2 grid of 16×16 wmma tiles
        #pragma unroll
        for (int wi = 0; wi < 2; wi++) {
            #pragma unroll
            for (int wj = 0; wj < 2; wj++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
                wmma::load_matrix_sync(a_frag, &As[warp_row * 32 + wi * 16][0], BK);
                wmma::load_matrix_sync(b_frag, &Bs[warp_col * 32 + wj * 16][0], BK);
                wmma::mma_sync(acc[wi][wj], a_frag, b_frag, acc[wi][wj]);
            }
        }
        __syncthreads();
    }

    // Store results via shared memory staging
    // Reuse As+Bs = 4KB for staging (4 warps × 256 floats = 4KB)
    float* staging = reinterpret_cast<float*>(As);
    float* warp_staging = staging + warp_id * 256;

    for (int wi = 0; wi < 2; wi++) {
        for (int wj = 0; wj < 2; wj++) {
            int out_row = bm + warp_row * 32 + wi * 16;
            int out_col = bn + warp_col * 32 + wj * 16;
            if (out_row >= M || out_col >= N) continue;

            if constexpr (F32_OUTPUT) {
                if (out_row + 16 <= M && out_col + 16 <= N) {
                    wmma::store_matrix_sync(
                        &reinterpret_cast<float*>(C)[(int64_t)out_row * N + out_col],
                        acc[wi][wj], N, wmma::mem_row_major);
                } else {
                    wmma::store_matrix_sync(warp_staging, acc[wi][wj], 16, wmma::mem_row_major);
                    __syncwarp();
                    const int lane = threadIdx.x % 32;
                    for (int e = lane; e < 256; e += 32) {
                        int r = e / 16, c = e % 16;
                        if (out_row + r < M && out_col + c < N)
                            reinterpret_cast<float*>(C)[(int64_t)(out_row + r) * N + out_col + c] = warp_staging[e];
                    }
                }
            } else {
                wmma::store_matrix_sync(warp_staging, acc[wi][wj], 16, wmma::mem_row_major);
                __syncwarp();
                const int lane = threadIdx.x % 32;
                for (int e = lane; e < 256; e += 32) {
                    int r = e / 16, c = e % 16;
                    if (out_row + r < M && out_col + c < N)
                        reinterpret_cast<__nv_bfloat16*>(C)[(int64_t)(out_row + r) * N + out_col + c] =
                            __float2bfloat16(warp_staging[e]);
                }
            }
        }
    }
}

static void launch_tiled_gemm_bf16(__nv_bfloat16* C, const __nv_bfloat16* A, const __nv_bfloat16* B,
    int M, int N, int K, cudaStream_t stream) {
    dim3 grid(cdiv(N, 64), cdiv(M, 64));
    tiled_gemm_kernel<false><<<grid, 128, 0, stream>>>(C, A, B, M, N, K);
}

static void launch_tiled_gemm_f32out(float* C, const __nv_bfloat16* A, const __nv_bfloat16* B,
    int M, int N, int K, cudaStream_t stream) {
    dim3 grid(cdiv(N, 64), cdiv(M, 64));
    tiled_gemm_kernel<true><<<grid, 128, 0, stream>>>(C, A, B, M, N, K);
}

// Pre-allocated decode buffers (avoid per-token cudaMalloc)
// Packed as [token_id, position, kv_len, kv_len+1] for single memcpy
static int* g_decode_params_d = nullptr;
static int* g_token_d = nullptr;   // points to g_decode_params_d[0]
static int* g_pos_d = nullptr;     // points to g_decode_params_d[1]

// Profiling support
static bool g_profile = false;
static int g_profile_tokens = 0;
struct ProfileTimers {
    double embedding_ms = 0, attn_gemm_ms = 0, attn_kernel_ms = 0;
    double ssm_gemm_ms = 0, ssm_conv_ms = 0, ssm_step_ms = 0;
    double ffn_gate_up_ms = 0, ffn_down_ms = 0, ffn_kernel_ms = 0;
    double lm_head_ms = 0, norm_ms = 0;
};
static ProfileTimers g_prof;

static double sync_and_ms(cudaStream_t s, cudaEvent_t start, cudaEvent_t stop) {
    cudaEventRecord(stop, s);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return (double)ms;
}

// GEMM wrapper: C = A @ B^T  (row-major), bf16 output
// A: [M, K] bf16, B: [N, K] bf16, C: [M, N] bf16
// M=1: custom GEMV, M>1: wmma tiled GEMM
static cudaStream_t g_compute_stream = 0;  // set from model.compute_stream

static void gemm_bf16(
    __nv_bfloat16* C, const __nv_bfloat16* A, const __nv_bfloat16* B,
    int M, int N, int K
) {
    if (M == 1) {
        launch_gemv_bf16(C, B, A, N, K, g_compute_stream);
    } else {
        launch_tiled_gemm_bf16(C, A, B, M, N, K, g_compute_stream);
    }
}

// GEMM wrapper: C = A @ B^T, f32 output
// M=1: custom GEMV with f32 output, M>1: wmma tiled GEMM with f32 output
static void gemm_bf16_f32out(
    float* C, const __nv_bfloat16* A, const __nv_bfloat16* B,
    int M, int N, int K
) {
    if (M == 1) {
        launch_gemv_bf16_f32out(C, B, A, N, K, g_compute_stream);
    } else {
        launch_tiled_gemm_f32out(C, A, B, M, N, K, g_compute_stream);
    }
}


// f32 -> bf16 cast kernel
__global__ void f32_to_bf16_kernel(
    __nv_bfloat16* __restrict__ output,
    const float* __restrict__ input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2bfloat16(input[idx]);
    }
}

static void cast_f32_to_bf16(__nv_bfloat16* output, const float* input, int n, cudaStream_t stream = 0) {
    f32_to_bf16_kernel<<<cdiv(n, 256), 256, 0, stream>>>(output, input, n);
}

// Deinterleave combined SSM projection from bf16 [n_tokens, combined_n] -> 4 separate f32 buffers
__global__ void deinterleave_ssm_proj_bf16_kernel(
    float* __restrict__ qkv,      // [n_tokens, conv_channels]
    float* __restrict__ z,         // [n_tokens, d_inner]
    float* __restrict__ alpha,     // [n_tokens, dt_rank]
    float* __restrict__ beta,      // [n_tokens, dt_rank]
    const __nv_bfloat16* __restrict__ combined,  // [n_tokens, combined_n]
    int n_tokens, int conv_channels, int d_inner, int dt_rank, int combined_n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * combined_n;
    if (tid >= total) return;
    int t = tid / combined_n;
    int c = tid % combined_n;
    float val = __bfloat162float(combined[tid]);
    if (c < conv_channels) {
        qkv[t * conv_channels + c] = val;
    } else if (c < conv_channels + d_inner) {
        z[t * d_inner + (c - conv_channels)] = val;
    } else if (c < conv_channels + d_inner + dt_rank) {
        alpha[t * dt_rank + (c - conv_channels - d_inner)] = val;
    } else {
        beta[t * dt_rank + (c - conv_channels - d_inner - dt_rank)] = val;
    }
}

// Deinterleave Q and Gate from packed [n_heads, head_dim*2] -> Q [n_heads, head_dim] + Gate [n_heads, head_dim]
__global__ void deinterleave_qg_kernel(
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ packed,
    int n_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_heads * head_dim;
    if (idx >= total) return;
    int head = idx / head_dim;
    int d = idx % head_dim;
    q_out[idx] = packed[head * head_dim * 2 + d];
    gate_out[idx] = packed[head * head_dim * 2 + head_dim + d];
}

// Allocate inference buffers
static void allocate_buffers(Model& model, int max_batch, int max_kv_len) {
    // max_batch = max prompt length for batched processing
    model.max_tokens = max_batch;
    model.max_kv_len = max_kv_len;
    model.kv_len = 0;

    // Hidden state in f32 for residual stream precision (matching ggml behavior)
    model.hidden_state = cuda_alloc<float>(max_batch * MC::n_embd);
    model.hidden_bf16  = cuda_alloc<__nv_bfloat16>(max_batch * MC::n_embd);
    model.norm_out     = cuda_alloc<__nv_bfloat16>(max_batch * MC::n_embd);
    model.norm_out_f32 = cuda_alloc<float>(max_batch * MC::n_embd);
    model.attn_out     = cuda_alloc<__nv_bfloat16>(max_batch * MC::n_embd);
    int gemm_max = MC::n_ff > MC::n_embd ? MC::n_ff : MC::n_embd;
    gemm_max = gemm_max > MC::n_vocab ? gemm_max : MC::n_vocab;
    // n_head * head_dim * 2 = 8192, ssm_conv_channels = 8192 - use max
    int qkv_max = MC::n_head * MC::head_dim * 2 + 2 * MC::n_head_kv * MC::head_dim;  // 10240 for packed Q+Gate+K+V
    if (MC::ssm_conv_channels > qkv_max) qkv_max = MC::ssm_conv_channels;
    model.gemm_out     = cuda_alloc<float>(max_batch * gemm_max);
    model.gemm_out2    = cuda_alloc<float>(max_batch * gemm_max);
    model.ffn_buf      = cuda_alloc<__nv_bfloat16>(max_batch * 2 * MC::n_ff);  // sized for packed gate+up
    model.ffn_buf2     = cuda_alloc<__nv_bfloat16>(max_batch * MC::n_ff);

    // QKV temp buffer (sized for batched attention)
    model.qkv_buf = cuda_alloc<__nv_bfloat16>(max_batch * qkv_max);
    // SSM f32 projection buffer for conv state precision
    model.ssm_proj_f32 = cuda_alloc<float>(max_batch * MC::ssm_conv_channels);

    model.logits_f32 = cuda_alloc<float>(max_batch * MC::n_vocab);

    // KV caches for attention layers
    int kv_dim = MC::n_head_kv * MC::head_dim;
    for (int i = 0; i < 8; i++) {
        model.k_cache[i] = cuda_alloc<__nv_bfloat16>(max_kv_len * kv_dim);
        model.v_cache[i] = cuda_alloc<__nv_bfloat16>(max_kv_len * kv_dim);
    }

    // Pre-allocated SSM temp buffers (sized for batched prompt)
    model.ssm_gate_buf     = cuda_alloc<float>(max_batch * MC::ssm_dt_rank);
    model.ssm_beta_buf     = cuda_alloc<float>(max_batch * MC::ssm_dt_rank);
    model.ssm_conv_out_buf = cuda_alloc<float>(max_batch * MC::ssm_conv_channels);
    model.ssm_q_rep_buf    = cuda_alloc<float>(max_batch * MC::ssm_dt_rank * MC::ssm_d_state);
    model.ssm_k_rep_buf    = cuda_alloc<float>(max_batch * MC::ssm_dt_rank * MC::ssm_d_state);
    model.ssm_delta_out_buf = cuda_alloc<float>(max_batch * MC::ssm_dt_rank * MC::ssm_head_v_dim);

    // SSM states
    int conv_state_size = (MC::ssm_conv_kernel - 1) * MC::ssm_conv_channels;
    int recurrent_state_size = MC::ssm_dt_rank * MC::ssm_head_v_dim * MC::ssm_head_v_dim;
    for (int i = 0; i < 24; i++) {
        model.ssm_conv_state[i] = cuda_alloc<float>(conv_state_size);
        CUDA_CHECK(cudaMemset(model.ssm_conv_state[i], 0, conv_state_size * sizeof(float)));
        model.ssm_recurrent_state[i] = cuda_alloc<float>(recurrent_state_size);
        CUDA_CHECK(cudaMemset(model.ssm_recurrent_state[i], 0, recurrent_state_size * sizeof(float)));
    }

    // CUDA graph support
    // Packed decode params: [token_id, position, kv_len, kv_len+1] — single memcpy per decode
    g_decode_params_d = cuda_alloc<int>(4);
    g_token_d = g_decode_params_d;
    g_pos_d = g_decode_params_d + 1;
    model.d_kv_len = g_decode_params_d + 2;
    CUDA_CHECK(cudaStreamCreate(&model.compute_stream));
    g_compute_stream = model.compute_stream;
    g_profile = (getenv("PROFILE") != nullptr);
    model.decode_graph = nullptr;
    model.decode_graph_exec = nullptr;
    model.decode_graph_captured = false;
}

// Residual add: f32 output = f32 a + f32 b
__global__ void residual_add_f32_kernel(float* output, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] + b[idx];
    }
}

static void residual_add_f32(float* output, const float* a, const float* b, int n, cudaStream_t stream = 0) {
    residual_add_f32_kernel<<<cdiv(n, 256), 256, 0, stream>>>(output, a, b, n);
}

static void forward_attention_layer_profiled(Model& model, int layer_idx, int attn_idx,
    float* hidden, int* positions_d, __nv_bfloat16* pending_bf16, cudaEvent_t t0, cudaEvent_t t1);

// Forward pass for one full-attention layer
// hidden is f32 (residual stream). Internal computation uses bf16 for GEMMs.
// pending_bf16: if non-null, fuse bf16→f32 residual add with input norm (saves 1 kernel launch)
static void forward_attention_layer(Model& model, int layer_idx, int attn_idx,
    float* hidden, int n_tokens, int* positions_d, __nv_bfloat16* pending_bf16 = nullptr) {
    auto& lw = model.attn_layers[attn_idx];

    cudaStream_t s = model.compute_stream;

    // 1. RMSNorm (f32 in, bf16 out) — optionally fused with pending residual
    if (pending_bf16) {
        launch_fused_bf16_residual_rmsnorm(model.norm_out, hidden, pending_bf16,
            lw.attn_norm, n_tokens, MC::n_embd, MC::rms_norm_eps, s);
    } else {
        launch_rmsnorm_f32in(model.norm_out, hidden, lw.attn_norm, n_tokens, MC::n_embd, MC::rms_norm_eps, s);
    }

    int kv_dim = MC::n_head_kv * MC::head_dim;
    int q_gate_dim = MC::n_head * MC::head_dim * 2;
    __nv_bfloat16* k_proj;
    __nv_bfloat16* v_proj;

    if (n_tokens == 1) {
        // Decode: packed Q+Gate+K+V in single GEMM (saves 1 GEMV call vs separate wq+wkv)
        int qkv_dim = q_gate_dim + 2 * kv_dim;  // 10240
        gemm_bf16(model.qkv_buf, model.norm_out, lw.wqkv, 1, qkv_dim, MC::n_embd);
        k_proj = model.qkv_buf + q_gate_dim;
        v_proj = model.qkv_buf + q_gate_dim + kv_dim;
    } else {
        // Batched: separate GEMMs for Q and K/V
        gemm_bf16(model.qkv_buf, model.norm_out, lw.wq, n_tokens, q_gate_dim, MC::n_embd);
        k_proj = model.attn_out;
        gemm_bf16(k_proj, model.norm_out, lw.wk, n_tokens, kv_dim, MC::n_embd);
        v_proj = model.ffn_buf2;
        gemm_bf16(v_proj, model.norm_out, lw.wv, n_tokens, kv_dim, MC::n_embd);
    }

    // 5. Deinterleave Q and Gate with a single kernel
    __nv_bfloat16* q_contiguous = model.norm_out;
    __nv_bfloat16* gate_buf = model.hidden_bf16;
    {
        int total = n_tokens * MC::n_head * MC::head_dim;
        deinterleave_qg_kernel<<<cdiv(total, 256), 256, 0, s>>>(
            q_contiguous, gate_buf, model.qkv_buf, n_tokens * MC::n_head, MC::head_dim);
    }

    // Q/K norm
    launch_rmsnorm_head(q_contiguous, q_contiguous, lw.attn_q_norm,
        n_tokens, MC::n_head, MC::head_dim, MC::rms_norm_eps, s);
    launch_rmsnorm_head(k_proj, k_proj, lw.attn_k_norm,
        n_tokens, MC::n_head_kv, MC::head_dim, MC::rms_norm_eps, s);

    // 6. RoPE on Q and K
    launch_rope(q_contiguous, positions_d, n_tokens, MC::n_head, MC::head_dim,
        MC::rope_dim, MC::rope_freq_base, s);
    launch_rope(k_proj, positions_d, n_tokens, MC::n_head_kv, MC::head_dim,
        MC::rope_dim, MC::rope_freq_base, s);

    // 7. Append K, V to cache (d_kv_len has current kv_len value)
    launch_kv_cache_append(model.k_cache[attn_idx], k_proj, model.d_kv_len, n_tokens, kv_dim, s);
    launch_kv_cache_append(model.v_cache[attn_idx], v_proj, model.d_kv_len, n_tokens, kv_dim, s);

    // 8. Attention (use prefill kernel for n_tokens > 1)
    if (n_tokens > 1) {
        launch_attention_prefill(model.attn_out, q_contiguous,
            model.k_cache[attn_idx], model.v_cache[attn_idx],
            n_tokens, model.kv_len, MC::n_head, MC::n_head_kv, MC::head_dim, MC::attn_scale, s);
    } else {
        // d_kv_len[1] has kv_len + n_tokens (total entries after append)
        launch_attention_decode(model.attn_out, q_contiguous,
            model.k_cache[attn_idx], model.v_cache[attn_idx],
            model.d_kv_len + 1, model.max_kv_len, MC::n_head, MC::n_head_kv, MC::head_dim, MC::attn_scale, s);
    }

    // 9. Sigmoid gate
    launch_sigmoid_mul(model.attn_out, model.attn_out, gate_buf,
        n_tokens * MC::n_head * MC::head_dim, s);

    // 10. Output projection -> bf16
    gemm_bf16(model.attn_out, model.attn_out, lw.wo, n_tokens, MC::n_embd, MC::n_head * MC::head_dim);

    // 11. Fused bf16 cast + residual + post-attention norm (saves cast kernel)
    launch_fused_bf16_residual_rmsnorm(model.norm_out, hidden, model.attn_out,
        lw.post_attn_norm, n_tokens, MC::n_embd, MC::rms_norm_eps, s);

    // FFN: fused gate+up GEMM, then SwiGLU from packed layout
    gemm_bf16(model.ffn_buf, model.norm_out, lw.ffn_gate_up, n_tokens, 2 * MC::n_ff, MC::n_embd);
    launch_swiglu_packed(model.ffn_buf, model.ffn_buf, n_tokens, MC::n_ff, s);

    // down_proj -> bf16
    gemm_bf16(model.attn_out, model.ffn_buf, lw.ffn_down, n_tokens, MC::n_embd, MC::n_ff);
    // Decode (n_tokens==1): residual add deferred — fused with next layer's norm
    // Prompt (n_tokens>1): apply residual add here (no cross-layer fusion)
    if (n_tokens > 1) {
        launch_bf16_residual_add(hidden, model.attn_out, n_tokens * MC::n_embd, s);
    }
}

// Forward pass for one SSM (delta-net) layer
// hidden is f32 (residual stream), n_tokens can be > 1 for prompt batching
static void forward_ssm_layer(Model& model, int layer_idx, int ssm_idx,
    float* hidden, int n_tokens, __nv_bfloat16* pending_bf16 = nullptr) {
    auto& lw = model.ssm_layers[ssm_idx];

    cudaStream_t s = model.compute_stream;

    // 1. RMSNorm (f32 in, bf16 out) — optionally fused with pending residual
    if (pending_bf16) {
        launch_fused_bf16_residual_rmsnorm(model.norm_out, hidden, pending_bf16,
            lw.attn_norm, n_tokens, MC::n_embd, MC::rms_norm_eps, s);
    } else {
        launch_rmsnorm_f32in(model.norm_out, hidden, lw.attn_norm, n_tokens, MC::n_embd, MC::rms_norm_eps, s);
    }

    // Pointers to SSM projection outputs
    float* qkv_proj;     // [n_tokens, ssm_conv_channels=8192]
    float* z_buf;        // [n_tokens, ssm_d_inner=4096]
    float* alpha_f32;    // [n_tokens, ssm_dt_rank=32]
    float* beta_raw_f32; // [n_tokens, ssm_dt_rank=32]

    if (n_tokens == 1) {
        // Decode: single combined GEMM, use pointer offsets (no copies!)
        static constexpr int combined_n = MC::ssm_conv_channels + MC::ssm_d_inner + MC::ssm_dt_rank + MC::ssm_dt_rank;
        gemm_bf16_f32out(model.gemm_out, model.norm_out, lw.w_combined, 1, combined_n, MC::n_embd);

        qkv_proj    = model.gemm_out;
        z_buf       = model.gemm_out + MC::ssm_conv_channels;
        alpha_f32   = model.gemm_out + MC::ssm_conv_channels + MC::ssm_d_inner;
        beta_raw_f32= model.gemm_out + MC::ssm_conv_channels + MC::ssm_d_inner + MC::ssm_dt_rank;
    } else {
        // Batched: bf16 combined GEMM + fused deinterleave+cast
        static constexpr int combined_n = MC::ssm_conv_channels + MC::ssm_d_inner + MC::ssm_dt_rank + MC::ssm_dt_rank;
        qkv_proj = model.ssm_proj_f32;
        z_buf = model.norm_out_f32;
        alpha_f32 = model.gemm_out2;
        beta_raw_f32 = model.gemm_out2 + n_tokens * MC::ssm_dt_rank;

        // bf16 GEMM into bf16 temp buffer (reuse ffn_buf which is max_batch * 2*n_ff bf16 — large enough)
        __nv_bfloat16* combined_bf16 = model.ffn_buf;
        gemm_bf16(combined_bf16, model.norm_out, lw.w_combined, n_tokens, combined_n, MC::n_embd);
        // Fused deinterleave + bf16→f32 cast
        int total = n_tokens * combined_n;
        deinterleave_ssm_proj_bf16_kernel<<<cdiv(total, 256), 256, 0, s>>>(
            qkv_proj, z_buf, alpha_f32, beta_raw_f32,
            combined_bf16, n_tokens, MC::ssm_conv_channels, MC::ssm_d_inner, MC::ssm_dt_rank, combined_n);
    }

    // 6. Conv1d + SiLU on QKV mixed (f32 in, f32 out)
    if (n_tokens == 1) {
        // Decode: fused conv1d+SiLU+state update (1 kernel instead of 2)
        launch_conv1d_silu_update(model.ssm_conv_out_buf, model.ssm_conv_state[ssm_idx],
            qkv_proj, lw.ssm_conv1d, MC::ssm_conv_channels, MC::ssm_conv_kernel, s);
    } else {
        launch_conv1d_silu(model.ssm_conv_out_buf, qkv_proj, model.ssm_conv_state[ssm_idx],
            lw.ssm_conv1d, n_tokens, MC::ssm_conv_channels, MC::ssm_conv_kernel, s);
        launch_update_conv_state(model.ssm_conv_state[ssm_idx], qkv_proj,
            model.ssm_conv_state[ssm_idx], n_tokens, MC::ssm_conv_channels, MC::ssm_conv_kernel, s);
    }

    // 7-13. Fused SSM step
    float scale = 1.0f / sqrtf((float)MC::ssm_d_state);

    if (n_tokens == 1) {
        launch_fused_ssm_step(
            model.norm_out,
            model.ssm_recurrent_state[ssm_idx],
            model.ssm_conv_out_buf,
            alpha_f32, lw.ssm_dt_bias, lw.ssm_a, beta_raw_f32,
            z_buf, lw.ssm_norm,
            MC::ssm_n_group, MC::ssm_dt_rank,
            MC::ssm_d_state, MC::ssm_head_v_dim,
            scale, MC::rms_norm_eps, MC::rms_norm_eps, s);
    } else {
        launch_fused_ssm_step_batched(
            model.norm_out,
            model.ssm_recurrent_state[ssm_idx],
            model.ssm_conv_out_buf,
            alpha_f32, lw.ssm_dt_bias, lw.ssm_a, beta_raw_f32,
            z_buf, lw.ssm_norm, n_tokens,
            MC::ssm_n_group, MC::ssm_dt_rank,
            MC::ssm_d_state, MC::ssm_head_v_dim,
            MC::ssm_conv_channels, MC::ssm_d_inner,
            scale, MC::rms_norm_eps, MC::rms_norm_eps, s);
    }

    // 14. Output projection -> bf16
    gemm_bf16(model.attn_out, model.norm_out, lw.ssm_out, n_tokens, MC::n_embd, MC::ssm_d_inner);

    // 15. Fused bf16 cast + residual + post-attention norm
    launch_fused_bf16_residual_rmsnorm(model.norm_out, hidden, model.attn_out,
        lw.post_attn_norm, n_tokens, MC::n_embd, MC::rms_norm_eps, s);

    // FFN: fused gate+up GEMM, then SwiGLU from packed layout
    gemm_bf16(model.ffn_buf, model.norm_out, lw.ffn_gate_up, n_tokens, 2 * MC::n_ff, MC::n_embd);
    launch_swiglu_packed(model.ffn_buf, model.ffn_buf, n_tokens, MC::n_ff, s);

    // down_proj -> bf16
    gemm_bf16(model.attn_out, model.ffn_buf, lw.ffn_down, n_tokens, MC::n_embd, MC::n_ff);
    // Decode: residual add deferred — fused with next layer's norm
    // Prompt: apply residual add here
    if (n_tokens > 1) {
        launch_bf16_residual_add(hidden, model.attn_out, n_tokens * MC::n_embd, s);
    }
}

// Profiled attention layer (decode only, n_tokens=1)
static void forward_attention_layer_profiled(Model& model, int layer_idx, int attn_idx,
    float* hidden, int* positions_d, __nv_bfloat16* pending_bf16, cudaEvent_t t0, cudaEvent_t t1) {
    if (!g_profile) {
        forward_attention_layer(model, layer_idx, attn_idx, hidden, 1, positions_d, pending_bf16);
        return;
    }
    auto& lw = model.attn_layers[attn_idx];

    cudaStream_t s = model.compute_stream;
#define PS() cudaEventRecord(t0, s)
#define PE(f) g_prof.f += sync_and_ms(s, t0, t1)

    PS();
    if (pending_bf16) {
        launch_fused_bf16_residual_rmsnorm(model.norm_out, hidden, pending_bf16,
            lw.attn_norm, 1, MC::n_embd, MC::rms_norm_eps, s);
    } else {
        launch_rmsnorm_f32in(model.norm_out, hidden, lw.attn_norm, 1, MC::n_embd, MC::rms_norm_eps, s);
    }
    PE(norm_ms);

    int kv_dim = MC::n_head_kv * MC::head_dim;
    int q_gate_dim = MC::n_head * MC::head_dim * 2;
    int qkv_dim = q_gate_dim + 2 * kv_dim;

    PS();
    gemm_bf16(model.qkv_buf, model.norm_out, lw.wqkv, 1, qkv_dim, MC::n_embd);
    PE(attn_gemm_ms);
    __nv_bfloat16* k_proj = model.qkv_buf + q_gate_dim;
    __nv_bfloat16* v_proj = model.qkv_buf + q_gate_dim + kv_dim;

    PS();
    __nv_bfloat16* q_contiguous = model.norm_out;
    __nv_bfloat16* gate_buf = model.hidden_bf16;
    int total = MC::n_head * MC::head_dim;
    deinterleave_qg_kernel<<<cdiv(total, 256), 256, 0, s>>>(
        q_contiguous, gate_buf, model.qkv_buf, MC::n_head, MC::head_dim);
    launch_rmsnorm_head(q_contiguous, q_contiguous, lw.attn_q_norm, 1, MC::n_head, MC::head_dim, MC::rms_norm_eps, s);
    launch_rmsnorm_head(k_proj, k_proj, lw.attn_k_norm, 1, MC::n_head_kv, MC::head_dim, MC::rms_norm_eps, s);
    launch_rope(q_contiguous, positions_d, 1, MC::n_head, MC::head_dim, MC::rope_dim, MC::rope_freq_base, s);
    launch_rope(k_proj, positions_d, 1, MC::n_head_kv, MC::head_dim, MC::rope_dim, MC::rope_freq_base, s);
    launch_kv_cache_append(model.k_cache[attn_idx], k_proj, model.d_kv_len, 1, kv_dim, s);
    launch_kv_cache_append(model.v_cache[attn_idx], v_proj, model.d_kv_len, 1, kv_dim, s);
    launch_attention_decode(model.attn_out, q_contiguous,
        model.k_cache[attn_idx], model.v_cache[attn_idx],
        model.d_kv_len + 1, model.max_kv_len, MC::n_head, MC::n_head_kv, MC::head_dim, MC::attn_scale, s);
    launch_sigmoid_mul(model.attn_out, model.attn_out, gate_buf, MC::n_head * MC::head_dim, s);
    PE(attn_kernel_ms);

    PS();
    gemm_bf16(model.attn_out, model.attn_out, lw.wo, 1, MC::n_embd, MC::n_head * MC::head_dim);
    PE(attn_gemm_ms);

    PS();
    launch_fused_bf16_residual_rmsnorm(model.norm_out, hidden, model.attn_out,
        lw.post_attn_norm, 1, MC::n_embd, MC::rms_norm_eps, s);
    PE(norm_ms);

    PS();
    gemm_bf16(model.ffn_buf, model.norm_out, lw.ffn_gate_up, 1, 2 * MC::n_ff, MC::n_embd);
    PE(ffn_gate_up_ms);

    PS();
    launch_swiglu_packed(model.ffn_buf, model.ffn_buf, 1, MC::n_ff, s);
    PE(ffn_kernel_ms);

    PS();
    gemm_bf16(model.attn_out, model.ffn_buf, lw.ffn_down, 1, MC::n_embd, MC::n_ff);
    PE(ffn_down_ms);

    // bf16_residual_add deferred — fused with next layer's norm
#undef PS
#undef PE
}

// Profiled SSM layer (decode only, n_tokens=1)
static void forward_ssm_layer_profiled(Model& model, int layer_idx, int ssm_idx,
    float* hidden, __nv_bfloat16* pending_bf16, cudaEvent_t t0, cudaEvent_t t1) {
    if (!g_profile) {
        forward_ssm_layer(model, layer_idx, ssm_idx, hidden, 1, pending_bf16);
        return;
    }
    auto& lw = model.ssm_layers[ssm_idx];

    cudaStream_t s = model.compute_stream;
#define PS() cudaEventRecord(t0, s)
#define PE(f) g_prof.f += sync_and_ms(s, t0, t1)

    PS();
    if (pending_bf16) {
        launch_fused_bf16_residual_rmsnorm(model.norm_out, hidden, pending_bf16,
            lw.attn_norm, 1, MC::n_embd, MC::rms_norm_eps, s);
    } else {
        launch_rmsnorm_f32in(model.norm_out, hidden, lw.attn_norm, 1, MC::n_embd, MC::rms_norm_eps, s);
    }
    PE(norm_ms);

    PS();
    static constexpr int combined_n = MC::ssm_conv_channels + MC::ssm_d_inner + MC::ssm_dt_rank + MC::ssm_dt_rank;
    gemm_bf16_f32out(model.gemm_out, model.norm_out, lw.w_combined, 1, combined_n, MC::n_embd);
    PE(ssm_gemm_ms);

    float* qkv_proj = model.gemm_out;
    float* z_buf = model.gemm_out + MC::ssm_conv_channels;
    float* alpha_f32 = model.gemm_out + MC::ssm_conv_channels + MC::ssm_d_inner;
    float* beta_raw_f32 = model.gemm_out + MC::ssm_conv_channels + MC::ssm_d_inner + MC::ssm_dt_rank;

    PS();
    launch_conv1d_silu_update(model.ssm_conv_out_buf, model.ssm_conv_state[ssm_idx],
        qkv_proj, lw.ssm_conv1d, MC::ssm_conv_channels, MC::ssm_conv_kernel, s);
    PE(ssm_conv_ms);

    PS();
    float scale = 1.0f / sqrtf((float)MC::ssm_d_state);
    launch_fused_ssm_step(
        model.norm_out, model.ssm_recurrent_state[ssm_idx], model.ssm_conv_out_buf,
        alpha_f32, lw.ssm_dt_bias, lw.ssm_a, beta_raw_f32,
        z_buf, lw.ssm_norm,
        MC::ssm_n_group, MC::ssm_dt_rank, MC::ssm_d_state, MC::ssm_head_v_dim,
        scale, MC::rms_norm_eps, MC::rms_norm_eps, s);
    PE(ssm_step_ms);

    PS();
    gemm_bf16(model.attn_out, model.norm_out, lw.ssm_out, 1, MC::n_embd, MC::ssm_d_inner);
    PE(ssm_gemm_ms);

    PS();
    launch_fused_bf16_residual_rmsnorm(model.norm_out, hidden, model.attn_out,
        lw.post_attn_norm, 1, MC::n_embd, MC::rms_norm_eps, s);
    PE(norm_ms);

    PS();
    gemm_bf16(model.ffn_buf, model.norm_out, lw.ffn_gate_up, 1, 2 * MC::n_ff, MC::n_embd);
    PE(ffn_gate_up_ms);

    PS();
    launch_swiglu_packed(model.ffn_buf, model.ffn_buf, 1, MC::n_ff, s);
    PE(ffn_kernel_ms);

    PS();
    gemm_bf16(model.attn_out, model.ffn_buf, lw.ffn_down, 1, MC::n_embd, MC::n_ff);
    PE(ffn_down_ms);

    // bf16_residual_add deferred — fused with next layer's norm
#undef PS
#undef PE
}

// Global temperature (set from command line)
static float g_temperature = 0.8f;

// Embedding lookup to f32: look up bf16 embedding, convert to f32
__global__ void embedding_to_f32_kernel(
    float* __restrict__ output,
    const __nv_bfloat16* __restrict__ embed_table,
    const int* __restrict__ token_ids,
    int dim
) {
    const int token_idx = blockIdx.x;
    const int token_id = token_ids[token_idx];
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const __nv_bfloat16* src = embed_table + (int64_t)token_id * dim;
    float* dst = output + (int64_t)token_idx * dim;
    for (int i = tid; i < dim; i += stride) {
        dst[i] = __bfloat162float(src[i]);
    }
}

static void ensure_decode_bufs() {
    // g_decode_params_d is allocated during model init
}

// Execute decode forward pass body (embedding through LM head)
// All operations go to model.compute_stream
static void forward_decode_body(Model& model) {
    cudaStream_t s = model.compute_stream;

    cudaEvent_t t0, t1;
    if (g_profile) {
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
    }

#define PROF_START() if (g_profile) cudaEventRecord(t0, s)
#define PROF_END(field) if (g_profile) g_prof.field += sync_and_ms(s, t0, t1)

    // Embedding lookup -> f32 hidden state
    PROF_START();
    embedding_to_f32_kernel<<<1, 1024, 0, s>>>(
        model.hidden_state, model.tok_embd, g_token_d, MC::n_embd);
    PROF_END(embedding_ms);

    // Process all layers — each layer defers its final bf16 residual add,
    // which gets fused with the next layer's input norm
    __nv_bfloat16* pending_bf16 = nullptr;
    for (int il = 0; il < MC::n_layers; il++) {
        if (MC::is_recurrent(il)) {
            forward_ssm_layer_profiled(model, il, model.layer_subidx[il], model.hidden_state, pending_bf16, t0, t1);
        } else {
            forward_attention_layer_profiled(model, il, model.layer_subidx[il], model.hidden_state, g_pos_d, pending_bf16, t0, t1);
        }
        pending_bf16 = model.attn_out;  // FFN down output, to be fused with next layer's norm
    }

    // Final norm: fuse last layer's residual add with output norm
    PROF_START();
    launch_fused_bf16_residual_rmsnorm(model.norm_out, model.hidden_state, pending_bf16,
        model.output_norm, 1, MC::n_embd, MC::rms_norm_eps, s);
    PROF_END(norm_ms);

    // LM head: [1, 4096] -> [1, 248320] -> f32 logits
    PROF_START();
    gemm_bf16_f32out(model.logits_f32, model.norm_out, model.output,
        1, MC::n_vocab, MC::n_embd);
    PROF_END(lm_head_ms);

    if (g_profile) {
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        g_profile_tokens++;
    }
#undef PROF_START
#undef PROF_END
}

// Full forward pass for n_tokens=1 (decode step) with CUDA graph acceleration
static int forward_decode(Model& model, int token_id, int position) {
    ensure_decode_bufs();
    cudaStream_t s = model.compute_stream;

    // Upload all decode parameters in a single memcpy (token_id, position, kv_len, kv_len+1)
    int decode_params[4] = { token_id, position, model.kv_len, model.kv_len + 1 };
    CUDA_CHECK(cudaMemcpyAsync(g_decode_params_d, decode_params, 4 * sizeof(int), cudaMemcpyHostToDevice, s));

    if (getenv("NO_GRAPH") || g_profile) {
        forward_decode_body(model);
    } else if (!model.decode_graph_captured) {
        // First decode: capture the compute graph
        CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        forward_decode_body(model);
        CUDA_CHECK(cudaStreamEndCapture(s, &model.decode_graph));
        CUDA_CHECK(cudaGraphInstantiate(&model.decode_graph_exec, model.decode_graph, nullptr, nullptr, 0));
        model.decode_graph_captured = true;
        CUDA_CHECK(cudaGraphLaunch(model.decode_graph_exec, s));
    } else {
        // Subsequent decodes: replay the captured graph
        CUDA_CHECK(cudaGraphLaunch(model.decode_graph_exec, s));
    }

    model.kv_len += 1;

    // Sample: for greedy, launch argmax on compute_stream (avoids sync gap)
    if (g_temperature <= 0.0f) {
        return gpu_argmax_on_stream(model.logits_f32, MC::n_vocab, s);
    }
    CUDA_CHECK(cudaStreamSynchronize(s));
    return sample_token(model.logits_f32, MC::n_vocab, g_temperature);
}

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s -m <model_path> -p <prompt> [-n <max_tokens>] [-t <temperature>]\n", prog);
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string prompt;
    int max_gen_tokens = 128;
    float temperature = 0.8f;

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            max_gen_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            temperature = atof(argv[++i]);
            g_temperature = temperature;
        }
    }

    if (model_path.empty()) {
        model_path = "/workspace/models/Qwen3.5-9B-BF16.gguf";
    }
    if (prompt.empty()) {
        prompt = "Hello, world!";
    }

    printf("Model: %s\n", model_path.c_str());
    printf("Prompt: %s\n", prompt.c_str());
    printf("Max tokens: %d\n", max_gen_tokens);
    printf("Temperature: %.2f\n\n", temperature);

    // Load tokenizer
    Tokenizer tokenizer;
    if (!tokenizer.load(model_path)) {
        fprintf(stderr, "Failed to load tokenizer\n");
        return 1;
    }

    // Tokenize prompt
    std::vector<int> prompt_tokens = tokenizer.encode(prompt);
    printf("Prompt tokens (%zu): ", prompt_tokens.size());
    for (int t : prompt_tokens) printf("%d ", t);
    printf("\n\n");

    // Load model
    Model model;
    memset(&model, 0, sizeof(model));
    if (!load_model(model_path, model)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Allocate buffers (max_batch = prompt size for batched prefill)
    int max_batch = (int)prompt_tokens.size();
    if (max_batch < 1) max_batch = 1;
    int max_kv_len = (int)prompt_tokens.size() + max_gen_tokens + 16;
    allocate_buffers(model, max_batch, max_kv_len);

    // Warm up GPU kernels on compute stream
    {
        __nv_bfloat16* dummy_a = model.norm_out;
        __nv_bfloat16* dummy_c = model.attn_out;
        // Warm up GEMV with representative sizes
        gemm_bf16(dummy_c, dummy_a, model.ssm_layers[0].ssm_out, 1, MC::n_embd, MC::ssm_d_inner);
        gemm_bf16(model.ffn_buf, dummy_c, model.ssm_layers[0].ffn_gate_up, 1, 2 * MC::n_ff, MC::n_embd);
        gemm_bf16(dummy_c, model.ffn_buf, model.ssm_layers[0].ffn_down, 1, MC::n_embd, MC::n_ff);
        gemm_bf16_f32out(model.logits_f32, dummy_a, model.output, 1, MC::n_vocab, MC::n_embd);
        CUDA_CHECK(cudaStreamSynchronize(model.compute_stream));
    }

    printf("\nGenerating...\n");

    // Batched prefill: process all prompt tokens through each layer together
    auto t_start = std::chrono::high_resolution_clock::now();

    int n_prompt = (int)prompt_tokens.size();
    int next_token = -1;
    {
        cudaStream_t s = model.compute_stream;

        // Upload all token IDs and positions
        int* tokens_d = cuda_alloc<int>(n_prompt);
        int* pos_d = cuda_alloc<int>(n_prompt);
        cuda_upload(tokens_d, prompt_tokens.data(), n_prompt);
        std::vector<int> positions(n_prompt);
        for (int i = 0; i < n_prompt; i++) positions[i] = i;
        cuda_upload(pos_d, positions.data(), n_prompt);

        // Upload kv_len for attention layers (kv_len=0 at start)
        int kv_params[2] = { model.kv_len, model.kv_len + n_prompt };
        cuda_upload(model.d_kv_len, kv_params, 2);

        // Embedding lookup for all prompt tokens -> f32 hidden state
        embedding_to_f32_kernel<<<n_prompt, 1024, 0, s>>>(
            model.hidden_state, model.tok_embd, tokens_d, MC::n_embd);

        // Process all layers with batched tokens
        for (int il = 0; il < MC::n_layers; il++) {
            if (MC::is_recurrent(il)) {
                forward_ssm_layer(model, il, model.layer_subidx[il], model.hidden_state, n_prompt);
            } else {
                forward_attention_layer(model, il, model.layer_subidx[il], model.hidden_state, n_prompt, pos_d);
            }
        }

        // Update KV cache position
        model.kv_len += n_prompt;

        // Final norm on last token only
        float* last_hidden = model.hidden_state + (n_prompt - 1) * MC::n_embd;
        launch_rmsnorm_f32in(model.norm_out, last_hidden, model.output_norm,
            1, MC::n_embd, MC::rms_norm_eps, s);

        // LM head on last token
        gemm_bf16_f32out(model.logits_f32, model.norm_out, model.output,
            1, MC::n_vocab, MC::n_embd);

        CUDA_CHECK(cudaStreamSynchronize(s));
        next_token = sample_token(model.logits_f32, MC::n_vocab, g_temperature);

        cudaFree(tokens_d);
        cudaFree(pos_d);
    }

    auto t_prompt = std::chrono::high_resolution_clock::now();
    double prompt_ms = std::chrono::duration<double, std::milli>(t_prompt - t_start).count();

    // Print prompt
    printf("%s", prompt.c_str());
    fflush(stdout);

    // Generate tokens
    std::vector<int> generated;
    auto t_gen_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < max_gen_tokens; i++) {
        if (next_token == tokenizer.eos_token_id()) break;

        generated.push_back(next_token);
        std::string tok_str = tokenizer.decode(next_token);
        printf("%s", tok_str.c_str());
        fflush(stdout);

        int pos = (int)prompt_tokens.size() + i;
        next_token = forward_decode(model, next_token, pos);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(t_end - t_gen_start).count();

    printf("\n\n--- Performance ---\n");
    printf("Prompt tokens: %zu (%.1f ms, %.1f tok/s)\n",
        prompt_tokens.size(), prompt_ms,
        prompt_tokens.size() * 1000.0 / prompt_ms);
    printf("Generated tokens: %zu (%.1f ms, %.1f tok/s)\n",
        generated.size(), gen_ms,
        generated.size() * 1000.0 / gen_ms);

    if (g_profile && g_profile_tokens > 0) {
        int n = g_profile_tokens;
        double ffn_gemm = g_prof.ffn_gate_up_ms + g_prof.ffn_down_ms;
        double total = g_prof.attn_gemm_ms + g_prof.attn_kernel_ms + g_prof.ssm_gemm_ms +
            g_prof.ssm_conv_ms + g_prof.ssm_step_ms + ffn_gemm + g_prof.ffn_kernel_ms +
            g_prof.lm_head_ms + g_prof.norm_ms + g_prof.embedding_ms;
        double gemm_total = g_prof.attn_gemm_ms + g_prof.ssm_gemm_ms + ffn_gemm + g_prof.lm_head_ms;
        printf("\n--- Profile (%d tokens, %.1f ms total, %.1f ms/tok) ---\n", n, total, total / n);
        printf("  FFN gate+up:  %7.1f ms (%5.1f%%, %.2f ms/tok)  [32×, N=24576 K=4096]\n", g_prof.ffn_gate_up_ms, 100*g_prof.ffn_gate_up_ms/total, g_prof.ffn_gate_up_ms/n);
        printf("  FFN down:     %7.1f ms (%5.1f%%, %.2f ms/tok)  [32×, N=4096 K=12288]\n", g_prof.ffn_down_ms, 100*g_prof.ffn_down_ms/total, g_prof.ffn_down_ms/n);
        printf("  SSM GEMM:     %7.1f ms (%5.1f%%, %.2f ms/tok)  [24× combined+out]\n", g_prof.ssm_gemm_ms, 100*g_prof.ssm_gemm_ms/total, g_prof.ssm_gemm_ms/n);
        printf("  LM head:      %7.1f ms (%5.1f%%, %.2f ms/tok)  [1×, N=248320 K=4096]\n", g_prof.lm_head_ms, 100*g_prof.lm_head_ms/total, g_prof.lm_head_ms/n);
        printf("  Attn GEMM:    %7.1f ms (%5.1f%%, %.2f ms/tok)  [8× wq+wkv+wo]\n", g_prof.attn_gemm_ms, 100*g_prof.attn_gemm_ms/total, g_prof.attn_gemm_ms/n);
        printf("  SSM conv:     %7.1f ms (%5.1f%%, %.2f ms/tok)\n", g_prof.ssm_conv_ms, 100*g_prof.ssm_conv_ms/total, g_prof.ssm_conv_ms/n);
        printf("  SSM step:     %7.1f ms (%5.1f%%, %.2f ms/tok)\n", g_prof.ssm_step_ms, 100*g_prof.ssm_step_ms/total, g_prof.ssm_step_ms/n);
        printf("  Norms:        %7.1f ms (%5.1f%%, %.2f ms/tok)\n", g_prof.norm_ms, 100*g_prof.norm_ms/total, g_prof.norm_ms/n);
        printf("  Attn kernels: %7.1f ms (%5.1f%%, %.2f ms/tok)\n", g_prof.attn_kernel_ms, 100*g_prof.attn_kernel_ms/total, g_prof.attn_kernel_ms/n);
        printf("  FFN kernels:  %7.1f ms (%5.1f%%, %.2f ms/tok)\n", g_prof.ffn_kernel_ms, 100*g_prof.ffn_kernel_ms/total, g_prof.ffn_kernel_ms/n);
        printf("  Embedding:    %7.1f ms (%5.1f%%, %.2f ms/tok)\n", g_prof.embedding_ms, 100*g_prof.embedding_ms/total, g_prof.embedding_ms/n);
        printf("  ---\n");
        printf("  GEMM total:   %7.1f ms (%5.1f%%, %.2f ms/tok)\n", gemm_total, 100*gemm_total/total, gemm_total/n);
        // Bandwidth calculation
        double weight_bytes = (
            32.0 * (24576 + 4096) * 4096 +  // FFN gate_up + down (all layers)
            24.0 * (12352 + 4096) * 4096 +  // SSM combined + out
            8.0 * (8192 + 2048 + 4096) * 4096 +  // Attn wq + wkv + wo
            248320.0 * 4096  // LM head
        ) * 2;  // bf16 = 2 bytes
        printf("  Weight bytes:  %.1f MB/tok, BW util: %.0f%% of 1792 GB/s\n",
            weight_bytes / 1e6, 100.0 * weight_bytes / (gemm_total / n / 1000.0) / 1.792e12);
    }

    free_model(model);
    return 0;
}
