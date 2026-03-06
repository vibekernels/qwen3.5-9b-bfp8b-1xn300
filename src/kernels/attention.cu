#include "../utils.h"
#include "../model.h"
#include <cuda_bf16.h>
#include <cstdint>
#include <cfloat>

// ============================================================
// GQA Attention for full-attention layers
// Q: [n_tokens, n_head, head_dim]  (16 heads)
// K: [n_tokens, n_head_kv, head_dim]  (4 heads)
// V: [n_tokens, n_head_kv, head_dim]  (4 heads)
// GQA ratio = n_head / n_head_kv = 4
//
// For the attention layer forward pass:
// 1. Project: Q+Gate = wq(x), K = wk(x), V = wv(x)
// 2. Split Q from Gate (packed in wq output)
// 3. Q/K norm (RMSNorm per head)
// 4. RoPE on Q, K
// 5. Append K,V to cache
// 6. Compute attention: softmax(Q @ K^T / sqrt(d)) @ V
// 7. Gate: output = sigmoid(gate) * attn_output
// 8. Output projection: wo(output)
// ============================================================

// Sigmoid activation kernel
__global__ void sigmoid_mul_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ attn_out,
    const __nv_bfloat16* __restrict__ gate,
    int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    float a = __bfloat162float(attn_out[idx]);
    float g = __bfloat162float(gate[idx]);
    float sig_g = 1.0f / (1.0f + expf(-g));
    output[idx] = __float2bfloat16(a * sig_g);
}

void launch_sigmoid_mul(
    __nv_bfloat16* output,
    const __nv_bfloat16* attn_out,
    const __nv_bfloat16* gate,
    int n_elements,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = cdiv(n_elements, threads);
    sigmoid_mul_kernel<<<blocks, threads, 0, stream>>>(output, attn_out, gate, n_elements);
}

// Append K,V to cache at position kv_pos
// K_new: [n_new_tokens, n_head_kv * head_dim]
// k_cache: [max_kv_len, n_head_kv * head_dim]
// kv_pos read from device pointer (for CUDA graph compatibility)
__global__ void kv_cache_append_kernel(
    __nv_bfloat16* __restrict__ cache,
    const __nv_bfloat16* __restrict__ new_kv,
    const int* __restrict__ d_kv_pos,
    int n_new_tokens,
    int kv_dim           // n_head_kv * head_dim
) {
    int kv_pos = *d_kv_pos;
    int token = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    __nv_bfloat16* dst = cache + (int64_t)(kv_pos + token) * kv_dim;
    const __nv_bfloat16* src = new_kv + (int64_t)token * kv_dim;

    for (int i = tid; i < kv_dim; i += stride) {
        dst[i] = src[i];
    }
}

void launch_kv_cache_append(
    __nv_bfloat16* cache,
    const __nv_bfloat16* new_kv,
    const int* d_kv_pos,
    int n_new_tokens,
    int kv_dim,
    cudaStream_t stream
) {
    int threads = (kv_dim < 1024) ? kv_dim : 1024;
    threads = ((threads + 31) / 32) * 32;
    kv_cache_append_kernel<<<n_new_tokens, threads, 0, stream>>>(
        cache, new_kv, d_kv_pos, n_new_tokens, kv_dim);
}

// Naive attention: Q @ K^T with causal masking, softmax, then @ V
// For decode (n_tokens=1), this is just a dot product per head
//
// This is a simple implementation; will be replaced with FlashAttention later.
//
// Q:  [n_tokens, n_head, head_dim]
// K_cache: [kv_len, n_head_kv, head_dim]
// V_cache: [kv_len, n_head_kv, head_dim]
// Output: [n_tokens, n_head, head_dim]

// Compute attention scores for one query head against one KV head
// score[i] = Q[q_token, head] . K[i, kv_head] / scale
// Then softmax, then weighted sum of V
// kv_len read from device pointer (for CUDA graph compatibility)
__global__ void attention_decode_kernel(
    __nv_bfloat16* __restrict__ output,    // [n_head, head_dim]
    const __nv_bfloat16* __restrict__ q,   // [n_head, head_dim]
    const __nv_bfloat16* __restrict__ k_cache,  // [kv_len, n_head_kv, head_dim]
    const __nv_bfloat16* __restrict__ v_cache,  // [kv_len, n_head_kv, head_dim]
    const int* __restrict__ d_kv_len,
    int n_head,
    int n_head_kv,
    int head_dim,
    float scale
) {
    int kv_len = *d_kv_len;
    // Each block handles one Q head
    const int head = blockIdx.x;
    const int kv_head = head / (n_head / n_head_kv); // GQA mapping
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    extern __shared__ float smem[];
    float* scores = smem;  // [kv_len]

    const __nv_bfloat16* q_head = q + head * head_dim;

    // Compute Q . K^T for all KV positions
    for (int kv_pos = tid; kv_pos < kv_len; kv_pos += stride) {
        const __nv_bfloat16* k_vec = k_cache + (int64_t)kv_pos * n_head_kv * head_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += __bfloat162float(q_head[d]) * __bfloat162float(k_vec[d]);
        }
        scores[kv_pos] = dot * scale;
    }
    __syncthreads();

    // Softmax: find max
    float max_val = -FLT_MAX;
    for (int i = tid; i < kv_len; i += stride) {
        max_val = fmaxf(max_val, scores[i]);
    }
    // Warp reduce max
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    __shared__ float s_max;
    if (tid == 0) s_max = max_val;
    // Cross-warp reduce if needed
    __shared__ float warp_maxes[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) warp_maxes[warp_id] = max_val;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane_id < (stride / 32)) ? warp_maxes[lane_id] : -FLT_MAX;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
        }
        if (lane_id == 0) s_max = v;
    }
    __syncthreads();

    // Softmax: exp and sum
    float sum_exp = 0.0f;
    for (int i = tid; i < kv_len; i += stride) {
        scores[i] = expf(scores[i] - s_max);
        sum_exp += scores[i];
    }
    // Reduce sum
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }
    __shared__ float s_sum;
    __shared__ float warp_sums[32];
    if (lane_id == 0) warp_sums[warp_id] = sum_exp;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane_id < (stride / 32)) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, offset);
        }
        if (lane_id == 0) s_sum = v;
    }
    __syncthreads();

    // Normalize
    float inv_sum = 1.0f / s_sum;
    for (int i = tid; i < kv_len; i += stride) {
        scores[i] *= inv_sum;
    }
    __syncthreads();

    // Weighted sum: output = sum(scores[i] * V[i])
    __nv_bfloat16* out_head = output + head * head_dim;
    for (int d = tid; d < head_dim; d += stride) {
        float acc = 0.0f;
        for (int kv_pos = 0; kv_pos < kv_len; kv_pos++) {
            const __nv_bfloat16* v_vec = v_cache + (int64_t)kv_pos * n_head_kv * head_dim + kv_head * head_dim;
            acc += scores[kv_pos] * __bfloat162float(v_vec[d]);
        }
        out_head[d] = __float2bfloat16(acc);
    }
}

// Batched attention for prefill: handle n_tokens > 1 with causal masking
// Q: [n_tokens, n_head, head_dim]
// K_cache, V_cache: already contain [kv_start + n_tokens] entries
// Each query q_t attends to kv positions 0..(kv_start + t), inclusive
__global__ void attention_prefill_kernel(
    __nv_bfloat16* __restrict__ output,    // [n_tokens, n_head, head_dim]
    const __nv_bfloat16* __restrict__ q,   // [n_tokens, n_head, head_dim]
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    int n_tokens,
    int kv_start,        // KV positions before this prefill
    int n_head,
    int n_head_kv,
    int head_dim,
    float scale
) {
    // blockIdx.x = token * n_head + head
    const int block_id = blockIdx.x;
    const int token = block_id / n_head;
    const int head = block_id % n_head;
    const int kv_head = head / (n_head / n_head_kv);
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    int kv_len = kv_start + token + 1; // causal: can see up to own position

    extern __shared__ float smem[];
    float* scores = smem;

    const __nv_bfloat16* q_head = q + ((int64_t)token * n_head + head) * head_dim;

    // Compute Q . K^T for all valid KV positions
    for (int kv_pos = tid; kv_pos < kv_len; kv_pos += stride) {
        const __nv_bfloat16* k_vec = k_cache + (int64_t)kv_pos * n_head_kv * head_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += __bfloat162float(q_head[d]) * __bfloat162float(k_vec[d]);
        }
        scores[kv_pos] = dot * scale;
    }
    __syncthreads();

    // Softmax: find max
    float max_val = -FLT_MAX;
    for (int i = tid; i < kv_len; i += stride) {
        max_val = fmaxf(max_val, scores[i]);
    }
    __shared__ float warp_maxes[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    if (lane_id == 0) warp_maxes[warp_id] = max_val;
    __syncthreads();
    __shared__ float s_max;
    if (warp_id == 0) {
        float v = (lane_id < (stride / 32)) ? warp_maxes[lane_id] : -FLT_MAX;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
        }
        if (lane_id == 0) s_max = v;
    }
    __syncthreads();

    // Exp and sum
    float sum_exp = 0.0f;
    for (int i = tid; i < kv_len; i += stride) {
        scores[i] = expf(scores[i] - s_max);
        sum_exp += scores[i];
    }
    __shared__ float warp_sums[32];
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }
    if (lane_id == 0) warp_sums[warp_id] = sum_exp;
    __syncthreads();
    __shared__ float s_sum;
    if (warp_id == 0) {
        float v = (lane_id < (stride / 32)) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, offset);
        }
        if (lane_id == 0) s_sum = v;
    }
    __syncthreads();

    float inv_sum = 1.0f / s_sum;
    for (int i = tid; i < kv_len; i += stride) {
        scores[i] *= inv_sum;
    }
    __syncthreads();

    // Weighted sum of V
    __nv_bfloat16* out_head = output + ((int64_t)token * n_head + head) * head_dim;
    for (int d = tid; d < head_dim; d += stride) {
        float acc = 0.0f;
        for (int kv_pos = 0; kv_pos < kv_len; kv_pos++) {
            const __nv_bfloat16* v_vec = v_cache + (int64_t)kv_pos * n_head_kv * head_dim + kv_head * head_dim;
            acc += scores[kv_pos] * __bfloat162float(v_vec[d]);
        }
        out_head[d] = __float2bfloat16(acc);
    }
}

void launch_attention_prefill(
    __nv_bfloat16* output,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache,
    int n_tokens,
    int kv_start,
    int n_head,
    int n_head_kv,
    int head_dim,
    float scale,
    cudaStream_t stream
) {
    int threads = 256;
    int total_kv = kv_start + n_tokens;
    size_t smem = total_kv * sizeof(float);
    attention_prefill_kernel<<<n_tokens * n_head, threads, smem, stream>>>(
        output, q, k_cache, v_cache, n_tokens, kv_start, n_head, n_head_kv, head_dim, scale);
}

void launch_attention_decode(
    __nv_bfloat16* output,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache,
    const int* d_kv_len,
    int max_kv_len,
    int n_head,
    int n_head_kv,
    int head_dim,
    float scale,
    cudaStream_t stream
) {
    // One block per head, allocate smem for max possible kv_len (for graph capture)
    int threads = 256;
    size_t smem = max_kv_len * sizeof(float);
    attention_decode_kernel<<<n_head, threads, smem, stream>>>(
        output, q, k_cache, v_cache, d_kv_len, n_head, n_head_kv, head_dim, scale);
}
