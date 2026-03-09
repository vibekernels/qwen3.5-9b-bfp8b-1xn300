# Qwen3.5-9B on Tenstorrent N300 — Optimization Progress

**Date:** 2026-03-09
**Current performance:** ~65 ms/tok (~15.4 tok/s)
**Target:** 50 ms/tok (20 tok/s)
**Gap:** ~15 ms/tok

---

## Architecture Overview

- **Hardware:** Tenstorrent N300 (2× Wormhole chips, 12 DRAM banks per chip, 56 Tensix cores per chip)
- **Model:** Qwen3.5-9B (BF16 weights stored as BFP8_B on device)
  - 32 layers: 24 SSM (DeltaNet) + 8 full attention
  - n_embd=4096, n_ff=12288, n_head=16, n_head_kv=4, head_dim=256
- **Weight format:** BFP8_B (1088 bytes/tile), total model ~9.5 GB across both chips
- **Execution:** Custom tt-metalium kernels (no ttnn dependency for matmuls). Traced execution for minimal dispatch overhead.

## Current Pipeline (per token, 32 layers)

```
Per-layer pipeline:
  1. Wait for previous FFN chain to complete (device)
  2. TP reduction: read chip 0 + chip 1 partials, host add, write back
  3. Async write hidden to chip 1 (overlaps with step 4)
  4. Replay norm+GEMV trace (device)
  5. Read GEMV result from device (blocking PCIe read)
  6. Host compute (SSM conv1d+deltanet or attention)
  7. Write residual to device
  8. Dispatch FFN chain trace (non-blocking)

After all 32 layers:
  9. TP reduction for final hidden
  10. Output RMSNorm (host)
  11. LM head GEMV (2-chip parallel)
```

## Profiling Breakdown (@60 tokens, steady state)

| Component | ms/tok | Notes |
|---|---|---|
| **norm_mm** | 54 | Dominant cost — includes FFN wait + GEMV read |
| &nbsp;&nbsp;ffn_wait | 35.8 | Waiting for previous layer's FFN chain (device-bound) |
| &nbsp;&nbsp;gemv_read | 18.3 | PCIe read of pre-layer GEMV output (includes device latency) |
| **ffn dispatch** | 3 | Host-side trace replay dispatch time |
| **host compute** | 3 | conv1d=0.5, deltanet=2.0, attn=0.3 |
| **reswrite** | 2 | PCIe write of residual to device |
| **lmhead** | 6 | Output norm + LM head GEMV + PCIe read |
| | | |
| **TP detail:** | | |
| &nbsp;&nbsp;ffn_device | 29.3 | Total FFN device execution (all 32 layers) |
| &nbsp;&nbsp;tp_reduce | 6.4 | Host-side TP reduction overhead (31 layers) |

**Per-layer averages:**
- FFN device time: ~0.9 ms/layer (≈71 MB weights @ ~75 GB/s DRAM BW)
- Host pipeline: ~1.0 ms/layer (TP reduce + GEMV read + host + reswr)
- System is **balanced** — device and host pipeline are nearly equal duration

## Completed Optimizations (chronological)

1. **Custom DRAM-sharded GEMV kernels** — 12 cores, one per DRAM bank, TRID pipelining with BLOCK=16
2. **BFP8_B weight packing** — 8-bit block floating point, 47% less DRAM bandwidth than BF16
3. **Multi-threaded weight packing** — 5 threads per layer during model load
4. **Pre-allocated device buffers** — Zero-alloc GEMV dispatch
5. **Dual-chip tensor parallelism** — FFN weights split 50/50 across chips (gate/up by M, down by K)
6. **Traced execution** — norm+GEMV and FFN chains captured as mesh traces, replayed with minimal dispatch
7. **LoFi math fidelity** — All GEMVs use MathFidelity::LoFi
8. **Pre-computed RoPE tables** — Static sin/cos lookup
9. **Transposed SSM state layout** — Better memory access patterns for deltanet recurrence
10. **Raw bf16 host ops** — AVX-512 vectorized conv1d, deltanet, attention
11. **Async chip 1 hidden write** — Overlaps with chip 0 GEMV execution
12. **Fused gate+up GEMV** — Single weight matrix with split writer kernel
13. **Custom RMSNorm (FPU single-core)** — Replaces ttnn::rms_norm
14. **Custom eltwise kernels** — add, multiply, SwiGLU on device
15. **Fused GEMV+resadd** — Output projection adds to hidden in single kernel pass
16. **2-chip LM head** — Split vocabulary rows across both chips for 2x bandwidth

## Failed/Reverted Optimizations

1. **Fused norm+GEMV kernel** (reader_gemv_fused_norm.cpp) — Each of 12 GEMV cores independently computes full RMSNorm by reading all 128 hidden tiles from DRAM. Caused 3x regression (179ms vs 64ms) due to massive DRAM bank contention from 12× redundant reads. Reverted to separate rmsnorm_fpu + GEMV dispatch.

## Remaining Optimization Opportunities

### High impact (~5 ms savings each)

1. **On-device TP reduction** — Currently host reads partials from both chips, adds in f32, writes back. Moving the add to device could save ~5 ms by eliminating one PCIe round-trip. Challenges:
   - Still need PCIe to move chip 1's partial to chip 0 (no direct Ethernet data path available)
   - Would add single-core eltwise add to device pipeline (~0.1 ms/layer × 31 = 3.1 ms)
   - Multi-core eltwise add could reduce to ~0.3 ms total
   - Net savings after restructuring: estimated 3-5 ms

2. **Reduce GEMV read overhead** (18.3 ms total) — The blocking `EnqueueReadMeshBuffer` includes waiting for the device to finish norm+GEMV. Could potentially overlap the read with other work, or reduce the amount of data transferred (currently reading full padded tile output).

3. **LM head optimization** (6 ms) — Currently sequential: host rmsnorm → write → 2-chip GEMV → 2-chip read. Could overlap norm write to chip 1 with chip 0 GEMV, or use on-device rmsnorm.

### Medium impact (~2-3 ms each)

4. **Reduce PCIe write overhead** (reswrite = 2 ms) — Writing 4096 floats (~16 KB) should be <0.1 ms at PCIe Gen4 speeds. The overhead is likely from tilization + EnqueueWriteMeshBuffer overhead. Could pre-tilize or keep data in tiled format.

5. **Custom 1×K dot-product compute kernel** — Current GEMV compute kernel does full 32×32 × 32×32 tile multiply (`matmul_tiles`), wasting 31/32 of compute on zero-padded activation rows. A specialized kernel that only computes the first row could reduce compute time by ~30×, making the system purely DRAM-bandwidth-bound.

### Low impact / speculative

6. **Increase DRAM bandwidth utilization** — Currently at ~75 GB/s per chip (~29% of 258 GB/s theoretical, ~93% of measured achievable ~80 GB/s). Already near hardware limits.

7. **Reduce host compute** (3 ms) — Already highly optimized with AVX-512. Marginal gains possible from further parallelization.

## DRAM Bandwidth Analysis

| Metric | Value |
|---|---|
| Theoretical peak (12 GT/s) | 258 GB/s per chip |
| Measured achievable | ~80 GB/s per chip |
| Current utilization | ~75 GB/s per chip (~93% of achievable) |
| Weight data per FFN layer per chip | ~71 MB (BFP8_B) |
| Pre-layer GEMV weight data per layer | ~38-46 MB (varies by layer type) |
| Total weight data per token | ~3.5 GB per chip |

## Key Files

| File | Description |
|---|---|
| `tt_metal/host/engine.cpp` | Main inference engine (~3430 lines) |
| `tt_metal/host/engine.h` | Public API |
| `tt_metal/host/model_config.h` | Model hyperparameters |
| `tt_metal/host/gguf_loader.{h,cpp}` | Weight loading |
| `tt_metal/kernels/dataflow/reader_gemv_dram_sharded.cpp` | DRAM-sharded GEMV reader |
| `tt_metal/kernels/dataflow/reader_gemv_fused_norm.cpp` | Fused norm reader (unused — reverted) |
| `tt_metal/kernels/dataflow/writer_gemv_split.cpp` | Split writer for fused gate+up |
| `tt_metal/kernels/compute/gemv.cpp` | GEMV compute kernel |

## Build & Test

```sh
cd tt_metal/build && make test_forward -j$(nproc)
sudo TT_METAL_RUNTIME_ROOT=/home/ubuntu/tt-metal \
  ./build/test_forward /home/ubuntu/qwen3.5-9b-bf16-1x5090/models/Qwen3.5-9B-BF16.gguf \
  "The capital of France is" 40
```

## Performance History

| Date | ms/tok | tok/s | Key change |
|---|---|---|---|
| Initial | ~700 | ~1.4 | ttnn::matmul on device |
| +BFP8_B packing | ~480 | ~2.1 | 47% less weight bandwidth |
| +Pre-alloc buffers | ~290 | ~3.4 | Zero-alloc dispatch |
| +Dual chip | ~165 | ~6.1 | 2× DRAM bandwidth |
| +Custom GEMV kernels | ~140 | ~7.1 | DRAM-sharded, TRID pipeline |
| +Traced execution | ~90 | ~11.1 | Minimal dispatch overhead |
| +LoFi math | ~65 | ~15.4 | Faster Tensix compute |
| **Current** | **~65** | **~15.4** | — |
| **Target** | **50** | **20** | — |
