# qwen3.5-9b-bfp8-1xn300

Custom inference engine for Qwen3.5-9B on a single Tenstorrent N300 card (two Wormhole chips), built from scratch with tt-metalium APIs. No ttnn dependency — all matmuls, norms, and element-wise ops use custom Tensix kernels.

[![Deploy to Koyeb](https://www.koyeb.com/static/images/deploy/button.svg)](https://app.koyeb.com/deploy?name=qwen3-5-9b-n300&type=docker&image=ghcr.io%2Fvibekernels%2Fqwen3.5-9b-bfp8-1xn300:latest&instance_type=gpu-tenstorrent-n300s&regions=na&instances_min=1&hc_grace_period%5B8888%5D=900&ports=8888;http;/&ports=22;tcp;;true;tcp&env%5BPUBLIC_KEY%5D=REPLACE_ME)

To enable SSH access, set `PUBLIC_KEY` to your SSH public key (e.g. contents of `~/.ssh/id_ed25519.pub`). Koyeb will assign a TCP proxy domain and port for SSH connections.

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

## Getting started

Requires clang-20 and a Tenstorrent N300 device.

```sh
make setup             # init tt-metal submodule + build SDK (~13 min, first time only)
make -j$(nproc)        # build everything
make test              # run integration tests (13 tests, ~60s)
make quicktest         # fast smoke test: "The capital of France is" -> Paris
```

The model (`unsloth/Qwen3.5-9B-GGUF:BF16`) is automatically downloaded from HuggingFace on first run and cached in `~/.cache/qwen-models/`. To use a local model file instead:

```sh
MODEL_PATH=/path/to/Qwen3.5-9B-BF16.gguf make test
```

## Running inference

```sh
# Interactive chat:
make chat

# HTTP server with chat UI (port 8888):
make serve

# Custom port:
PORT=9090 make serve

# Single prompt (auto-downloads model if needed):
make quicktest

# Manual run:
TT_METAL_RUNTIME_ROOT=$(pwd)/third_party/tt-metal \
  ./build/test_forward "unsloth/Qwen3.5-9B-GGUF:BF16" "What is the capital of France?" 128
```

Pass `--raw` as a 4th argument to skip the chat template and send the prompt directly.

### HTTP server

`make serve` starts an OpenAI-compatible HTTP server with a built-in chat UI:

- `GET /` — Chat UI (dark theme, markdown rendering, streaming responses)
- `POST /v1/chat/completions` — OpenAI-compatible API (streaming and non-streaming)
- `GET /v1/models` — List available models
- `GET /health` — Health check
- `GET /api/status` — Model loading progress (downloading/loading/ready/failed)

The model downloads and loads in the background while the server is already accepting connections. The chat UI shows a loading overlay with download progress until the model is ready.

```sh
# Use as an OpenAI-compatible API:
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-9b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":128}'
```

## Project structure

```
src/
  engine.cpp                  # inference engine: forward pass, generate loop
  engine.h                    # public API: load_model_and_tokenizer(), generate(), etc.
  server.cpp                  # HTTP server with chat UI and OpenAI-compatible API
  chat.cpp                    # interactive CLI chat
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
    httplib.h                 # cpp-httplib HTTP server
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
