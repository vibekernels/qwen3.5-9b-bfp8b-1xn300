# qwen3.5-9b-bf16-1xn300d

Custom inference engine for Qwen3.5-9B on a single Tenstorrent N300 card (two Wormhole chips), built from scratch with tt-metalium APIs. No ttnn dependency — all matmuls, norms, and element-wise ops use custom Tensix kernels.

## Architecture

Qwen3.5-9B is a hybrid architecture with 32 layers: 8 full attention layers (every 4th) and 24 SSM delta-net recurrent layers.

- **Custom DRAM-sharded GEMV kernels** on Tensix cores (12 cores per chip, one per DRAM bank, TRID pipelining)
- **Tensor-parallel FFN** across both chips: gate/up weights split by M, down weights split by K, host-side TP reduction
- **Custom on-device RMSNorm** (single-core FPU kernel, eliminates PCIe round-trip)
- **Custom eltwise kernels**: add, multiply, SwiGLU, fused GEMV+residual-add
- **On-device FFN chain**: outproj+resadd → RMSNorm → gate+up GEMV split → SwiGLU → down GEMV+resadd, all as a single traced operation per chip
- **Fused gate+up GEMV** with split writer kernel (single weight matrix, two output buffers)
- **SSM recurrence** and **attention** (online softmax + KV cache) run on host CPU in f32 with AVX-512
- **Metal Traces** capture and replay device op sequences with near-zero dispatch overhead
- **BFP8_B weights** (8-bit block floating point, 1088 bytes/tile) — 47% less DRAM bandwidth than BF16

## Performance

| Metric | Value |
|--------|-------|
| Decode latency | ~65 ms/tok |
| Decode throughput | ~15.4 tok/s |
| Model size (device) | ~9.5 GB BFP8_B across 2 chips |
| DRAM bandwidth utilization | ~75 GB/s per chip (~93% of achievable) |

Performance measured on Tenstorrent N300 (2x Wormhole chips, 1000 MHz AI clock, 12 Gbps DRAM). The first few tokens are slower while traces are captured; subsequent tokens stabilize at ~65ms.

### Decode time breakdown (steady state)

| Component | Time | Notes |
|-----------|------|-------|
| norm + pre-layer GEMV | 54 ms | Includes waiting for previous FFN chain (35.8ms) + GEMV read (18.3ms) |
| FFN chain (device) | 29.3 ms | Overlaps with host pipeline; ~0.9 ms/layer |
| TP reduction | 6.4 ms | Host-side: read both chips, f32 add, write back |
| Host compute | 3 ms | conv1d (0.5ms), deltanet (2.0ms), attention (0.3ms) |
| Residual writes | 2 ms | 32 PCIe writes per token |
| LM head | 6 ms | Output norm + 2-chip parallel GEMV + read |
| FFN dispatch | 3 ms | Non-blocking trace replay |

### Performance history

| Milestone | ms/tok | tok/s | Key change |
|-----------|--------|-------|------------|
| Initial | ~700 | ~1.4 | ttnn::matmul on device |
| +BFP8_B packing | ~480 | ~2.1 | 47% less weight bandwidth |
| +Pre-alloc buffers | ~290 | ~3.4 | Zero-alloc dispatch |
| +Dual chip TP | ~165 | ~6.1 | 2x DRAM bandwidth for FFN |
| +Custom GEMV kernels | ~140 | ~7.1 | DRAM-sharded, TRID pipeline |
| +Traced execution | ~90 | ~11.1 | Minimal dispatch overhead |
| +LoFi math fidelity | ~65 | ~15.4 | Faster Tensix compute |

## Building

Requires tt-metal built from source and clang-20.

```sh
cd tt_metal
mkdir -p build && cd build
cmake .. -DTT_METAL_BUILD=/path/to/tt-metal/build_Release -DCMAKE_CXX_COMPILER=clang++-20
make -j$(nproc)
```

This produces test binaries: `test_device`, `test_matmul`, `test_load_weights`, `test_forward`.

## Running inference

Requires `TT_METAL_RUNTIME_ROOT` pointing to your tt-metal source tree:

```sh
TT_METAL_RUNTIME_ROOT=/path/to/tt-metal \
  ./build/test_forward /path/to/Qwen3.5-9B-BF16.gguf "What is the capital of France?" 128
```

Pass `--raw` as a 4th argument to skip the chat template and send the prompt directly.

## Tests

All tests require `TT_METAL_RUNTIME_ROOT`:

```sh
TT_METAL_RUNTIME_ROOT=/path/to/tt-metal ./build/test_device        # validate N300 device opens
TT_METAL_RUNTIME_ROOT=/path/to/tt-metal ./build/test_matmul        # basic matmul
TT_METAL_RUNTIME_ROOT=/path/to/tt-metal ./build/test_load_weights  # GGUF weight loading
TT_METAL_RUNTIME_ROOT=/path/to/tt-metal ./build/test_forward       # full generation test
```

## Project structure

```
src/
  engine.cpp                  # inference engine: forward pass, generate loop
  engine.h                    # public API: load_model_and_tokenizer(), generate(), etc.
  gguf_loader.{h,cpp}        # GGUF weight loading into device DRAM MeshBuffers
  model_config.h              # Qwen3.5-9B hyperparameters and tile dimensions
  tokenizer.{h,cpp}          # BPE tokenizer (GPT-2 byte-level)
  download.{h,cpp}           # HuggingFace model download
  kernels/
    compute/
      gemv.cpp                # GEMV compute kernel (matmul_tiles accumulation)
      rmsnorm.cpp             # RMSNorm compute kernel
      eltwise_binary.cpp      # Element-wise add/multiply compute kernel
      swiglu.cpp              # SwiGLU activation compute kernel
    dataflow/
      reader_gemv_dram_sharded.cpp  # DRAM-sharded GEMV reader (TRID pipelining)
      writer_gemv.cpp               # Standard GEMV writer
      writer_gemv_split.cpp         # Split writer (gate+up → two buffers)
      writer_gemv_resadd.cpp        # GEMV writer with fused residual add
  tests/
    test_device.cpp           # device validation
    test_matmul.cpp           # basic matmul test
    test_load_weights.cpp     # weight loading test
    test_forward.cpp          # end-to-end generation test
    test_inference.cpp        # integration test suite
  third_party/
    json.hpp                  # nlohmann/json header
Makefile                      # build system
```

## Key optimizations

- **Custom DRAM-sharded GEMV**: 12 reader cores (one per DRAM bank) with TRID-pipelined weight reads for maximum bandwidth
- **BFP8_B weights** (1088 bytes/tile vs 2048 for BF16) with native hardware decompression
- **Tensor-parallel FFN** across 2 chips: each chip handles half the FFN width
- **Fused gate+up GEMV**: single weight matrix with split writer avoids redundant activation reads
- **Fused GEMV+residual-add**: output projection and residual addition in one kernel pass
- **Metal Traces** for norm+GEMV and FFN chain ops (near-zero dispatch overhead on replay)
- **LoFi math fidelity** for all GEMVs (faster Tensix multiply-accumulate)
- **Async chip 1 dispatch**: hidden state write to chip 1 overlaps with chip 0 GEMV
- **2-chip LM head**: vocabulary rows split across chips for 2x read bandwidth
- **AVX-512 host compute**: vectorized conv1d, deltanet recurrence, attention, RoPE
- **Pre-allocated device buffers** for zero-allocation GEMV dispatch
- **Pre-computed RoPE tables** (avoids trig calls per token)
- **Raw bf16 bit operations** for f32<->bf16 conversion (no bfloat16 class overhead)
- **Static scratch buffers** for all forward pass intermediates (no heap allocation per token)
