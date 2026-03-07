# qwen3.5-9b-bf16-1xn300d

Custom inference engine for Qwen3.5-9B on a single Tenstorrent N300 card (two Wormhole chips), built from scratch with hand-written tt-metal kernels (no frameworks).

## Architecture

Qwen3.5-9B is a hybrid architecture with 32 layers: 8 full attention layers (every 4th) and 24 SSM delta-net recurrent layers.

- **Matrix multiplications** run on-device via `ttnn::matmul` on Tensix cores
- **Chip 0** holds layer weights (attention QKV, SSM combined, FFN gate/up/down projections)
- **Chip 1** holds output projections (attention Wo, SSM out, LM head)
- **Small element-wise ops** (RoPE, attention scores, SSM recurrence, gating, RMSNorm) run on host CPU in f32
- **Weights** stored on device DRAM as tiled BF16 MeshBuffers (~12 GB total)

## Performance

| Metric | Value |
|--------|-------|
| Decode latency | ~290 ms/tok |
| Decode throughput | ~3.4 tok/s |
| Model size | ~18 GB BF16 |

Performance measured on Tenstorrent N300 (2x Wormhole chips, 1000 MHz AI clock, 12 Gbps DRAM).

## Building

Requires tt-metal built from source (with `_ttnncpp.so`).

```sh
cd tt_metal
mkdir -p build && cd build
cmake .. -DTT_METAL_BUILD=/path/to/tt-metal/build_Release
make -j$(nproc)
```

This produces test binaries: `test_device`, `test_matmul`, `test_load_weights`, `test_forward`.

## Running inference

```sh
./build/test_forward /path/to/Qwen3.5-9B-BF16.gguf "What is the capital of France?" 128
```

Pass `--raw` as a 4th argument to skip the chat template and send the prompt directly.

## Tests

```sh
./build/test_device        # validate N300 device opens correctly
./build/test_matmul        # basic ttnn::matmul on device
./build/test_load_weights  # load GGUF weights into device DRAM
./build/test_forward       # full generation test
```

## Project structure

```
tt_metal/
  CMakeLists.txt              # build system
  host/
    engine.cpp                # inference engine (forward pass, generate loop)
    engine.h                  # public API: load_model_and_tokenizer(), generate(), etc.
    gguf_loader.cpp           # GGUF weight loading into device DRAM MeshBuffers
    gguf_loader.h             # loader interface
    model_config.h            # Qwen3.5-9B hyperparameters and tile dimensions
  kernels/
    compute/                  # Tensix compute kernels (unused — matmuls via ttnn)
    dataflow/                 # data movement kernels (unused — using ttnn API)
  tests/
    test_device.cpp           # device validation
    test_matmul.cpp           # basic matmul test
    test_load_weights.cpp     # weight loading test
    test_forward.cpp          # end-to-end generation test
src/
  tokenizer.{h,cpp}          # BPE tokenizer (GPT-2 byte-level)
  download.{h,cpp}           # HuggingFace model download
```

## Key optimizations

- **Pre-allocated device buffers** for zero-allocation GEMV dispatch (no per-call MeshBuffer create/destroy)
- **Two-chip parallelism**: layer weights on chip 0, output projections on chip 1
- **Pre-computed RoPE tables** (avoids trig calls per token)
- **Raw bf16 bit operations** for f32<->bf16 conversion (avoids bfloat16 class overhead)
- **Static scratch buffers** for all forward pass intermediates (no heap allocation per token)
- **Transposed SSM state layout** `[head_v, head_k]` for contiguous inner-loop access
- **Program cache enabled** for faster repeated matmul dispatch
