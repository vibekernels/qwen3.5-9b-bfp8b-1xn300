# CUDA Migration Progress: Qwen3.5-9B Inference

## Status: Phase 6 — cuBLAS Eliminated

Custom CUDA inference engine for Qwen3.5-9B (BF16) with **zero external BLAS dependencies**. Beats llama.cpp on both prompt eval and generation using only custom GEMV and wmma tensor-core GEMM kernels.

## Performance (RTX 5090, BF16)

| Metric | Our Implementation | llama.cpp | Speedup |
|--------|-------------------|-----------|---------|
| Prompt eval (112 tok) | **1790 tok/s** | 563 tok/s | **3.18×** |
| Generation | **94.9 tok/s** | 77.8 tok/s | **1.22×** |
| Binary size | **532 KB** | ~150 MB | 283× smaller |
| Dependencies | cudart only | cuBLAS, cuDNN, etc. | — |

## Phase 2 Optimizations Applied

### Weight Packing (Reduced GEMM launches)
- **Attention K+V packed**: Single GEMM for K+V during decode (saves 1 cuBLAS call per layer)
- **FFN gate+up packed**: Single GEMM for gate+up with `swiglu_packed` kernel (saves 1 per layer)
- **SSM combined projection**: Packed QKV+Z+alpha+beta into single GEMM with fused bf16 deinterleave (4→1 GEMM for batched SSM)

### Fused Kernels (Reduced kernel launches)
- **Fused residual + RMSNorm**: Combines residual add + norm in one kernel (2 launches → 1)
- **Fused bf16 residual + RMSNorm**: Cast bf16→f32 + residual + norm in one kernel (3 launches → 1)
- **Fused bf16 residual add**: Cast + add for FFN down output (2 launches → 1)
- **Fused SSM step**: Combines gate compute, beta sigmoid, L2 norm, repeat heads, delta-net decode, gated RMSNorm (7 ops → 1 kernel)
- **SwiGLU packed**: Works directly on interleaved gate+up GEMM output

### SSM Recurrence Optimization
- **Batched SSM kernel**: All prompt tokens processed in a single kernel launch (n_tokens launches → 1)
- **Shared memory state**: 64KB state matrix (128×128) kept in shared memory during batched recurrence (4.8× faster — 93ms→19ms for 24 SSM layers on 111 tokens)

### Adaptive GEMM Strategy
- **M≤4**: Direct bf16→f32 output (decode path, saves cast)
- **M>4**: bf16→bf16 GEMM + fused bf16→f32 cast in downstream kernel (faster cuBLAS kernels)
- **cuBLAS warm-up**: Pre-warm cuBLAS workspace before timing

### CUDA Graph Decode (Phase 3)
- **Full decode graph capture**: Entire decode forward pass captured as CUDA graph on first token, replayed for all subsequent tokens
- **Device-side kv_len**: Attention kernels read kv_len from device memory so graph stays valid as kv_len changes
- **Named compute stream**: All operations routed through a non-default stream for graph capture compatibility
- **Stream-aware argmax**: Greedy sampling runs on compute stream, avoiding sync gap between graph and sampling

### SSM Decode Shared Memory (Phase 4)
- **Single-token SSM step in shared memory**: 64KB state matrix (128x128) loaded into shared memory for decode, matching batched version pattern (43us -> 15us per call, 2.9x faster)

### Cross-Layer Fusion & Profiling (Phase 5)
- **Packed Q+Gate+K+V GEMM**: Single GEMM for all attention projections during decode (wq+wkv → wqkv [10240, 4096], saves 8 cuBLAS calls, 137→129 per token)
- **Cross-layer residual fusion**: FFN down's bf16 residual add fused with next layer's input RMSNorm (saves 32 kernel launches per token)
- **Fused conv1d+state update**: Combined conv1d_silu + update_conv_state into single kernel for decode (2→1 launch per SSM layer)
- **Packed decode params**: Single 16-byte memcpy for token_id+position+kv_len (3→1 memcpy)
- **Built-in profiling**: `PROFILE=1` env var for per-category GPU timing breakdown

### cuBLAS Elimination (Phase 6)
- **Custom GEMV kernel**: Vectorized 128-bit loads from DRAM, L2-cached x vector, bf162 packed operations, warp-level reduction. 8 warps/block = 8 rows/block. **Streaming loads (`__ldcs`)** for weight data bypass L2 cache, preserving L2 for x-vector and intermediates across ~129 GEMV calls per decode step (+5% tok/s).
- **wmma Tensor Core GEMM**: 64×64×16 tiled GEMM using `nvcuda::wmma` API for prompt eval. 4 warps (2×2), each computing 32×32 via 2×2 wmma 16×16 tiles. Achieves 97.5% of cuBLAS prompt throughput.
- **Zero BLAS dependency**: Only links `libcudart.so`. Binary: 532KB (vs ~150MB with cuBLAS). No cuBLAS workspace allocation (~120ms startup saved).

### Other
- GPU argmax sampling (avoids downloading 248K float logits)
- Pre-allocated decode buffers (avoids per-token cudaMalloc)
- Batched prompt processing through all layers

## Architecture

Qwen3.5-9B is a **hybrid Mamba-Attention** model (delta-net linear attention):

- 32 layers: 24 SSM (delta-net) + 8 full attention (at positions 3,7,11,...,31)
- Hidden dim: 4096, FFN dim: 12288, vocab: 248,320
- Attention: 16 Q heads, 4 KV heads (GQA 4:1), head_dim=256
- SSM: d_state=128, n_group=16, dt_rank=32, conv_kernel=4

## Decode Profile Breakdown (PROFILE=1, 128 tokens)

| Category | ms/tok | % | Notes |
|----------|--------|---|-------|
| FFN gate+up | 3.94 | 35.2% | 32×, N=24576 K=4096, 92% BW |
| FFN down | 2.08 | 18.6% | 32×, N=4096 K=12288, 86% BW |
| SSM GEMM | 2.12 | 19.0% | 24× combined+out |
| LM head | 1.20 | 10.7% | 1×, N=248320 K=4096, 95% BW |
| Attn GEMM | 0.61 | 5.5% | 8× packed wqkv+wo |
| SSM step | 0.44 | 3.9% | 24× fused SSM (shared mem) |
| Norms | 0.31 | 2.7% | Fused residual+norm |
| Attn kernels | 0.29 | 2.6% | rope, deinterleave, kv_append, attention_decode |
| FFN kernels | 0.09 | 0.8% | swiglu |
| SSM conv | 0.10 | 0.9% | Fused conv1d+update |
| **GEMM total** | **9.95** | **89.1%** | 13.7GB/tok, 77% of 1792 GB/s |

## Bandwidth Analysis

- **Total weight bytes**: 13.7 GB/token (BF16)
- **Theoretical minimum**: 13.7 GB / 1792 GB/s = 7.65 ms/tok (131 tok/s)
- **Actual (graph mode, streaming GEMV)**: ~10.5 ms/tok (94.9 tok/s) = **80% bandwidth utilization**
- **Actual (graph mode, non-streaming)**: ~11.0 ms/tok (90.7 tok/s) = **76% bandwidth utilization**
- **Streaming loads**: `__ldcs` on weight data bypasses L2, improving both GEMV BW and non-GEMM kernel speed (intermediates stay in L2)

### cuBLAS Overhead Investigation

Benchmarked persistent GEMV kernel (cooperative groups + grid sync) and custom vectorized GEMV vs cuBLAS:

| Approach | Cold L2 (realistic) | Warm L2 | Notes |
|----------|-------------------|---------|-------|
| cuBLAS GemmEx | 71.8 µs | 69.4 µs | Tensor cores, hand-tuned SASS |
| Custom GEMV (vectorized 128-bit) | 71.7 µs | 20.5 µs | 3.4× faster on warm L2, equal on cold |
| Persistent GEMV (grid sync) | — | ~same | 2.4 µs/sync × 129 = 0.31ms overhead |

**Key finding**: The "per-kernel overhead" is not cuBLAS dispatch cost — it's fundamental DRAM access latency for cold cache lines. Custom kernels perform identically to cuBLAS when L2 is cold (the realistic scenario). The ~21% gap from theoretical peak BW is inherent to the memory access pattern, not the kernel implementation.

**Replacing cuBLAS**: Technically feasible (custom GEMV matches cuBLAS for cold L2), but would not improve inference speed. Benefits would be: eliminating ~150MB library dependency, enabling GEMV+post-op fusion, reducing binary size.

## Build & Run

```sh
make -j
./qwen-inference /path/to/qwen3.5-9b-bf16.gguf -p "Your prompt here" -n 128 -t 0
```
