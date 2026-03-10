# Project goals

The point of this project is to write a Tenstorrent accelerator kernel. ALL computation should run on the Tenstorrent N300 device (two Wormhole chips). Do not fall back to pure CPU compute — use the device for matmuls and as many operations as possible. If quantization is needed, do not go below INT8.

# Project structure

- `src/engine.cpp` — inference engine: forward pass, `generate()`, `load_model_and_tokenizer()`
- `src/engine.h` — public API: `generate()`, `load_model_and_tokenizer()`, `reset_state()`, `shutdown()`, `get_tokenizer()`
- `src/gguf_loader.{h,cpp}` — GGUF weight loading into device DRAM MeshBuffers
- `src/model_config.h` — model hyperparameters and tile dimensions
- `src/tokenizer.{h,cpp}` — BPE tokenizer (GPT-2 byte-level)
- `src/download.{h,cpp}` — HuggingFace model download
- `src/kernels/compute/` — Tensix compute kernels (gemv, rmsnorm, swiglu, etc.)
- `src/kernels/dataflow/` — data movement kernels (readers/writers)
- `src/tests/` — test suite (test_inference.cpp, test_forward.cpp, benchmarks)
- `src/third_party/` — third-party headers (json.hpp)
- `CMakeLists.txt` — CMake build system (root level)

## Build & test

```sh
make -j$(nproc)        # build everything
make test              # run integration tests
make clean             # remove build artifacts
```

Environment variables:
- `TT_METAL_HOME` — tt-metal source tree (default: `/home/ubuntu/tt-metal`)
- `TT_METAL_BUILD` — tt-metal build dir (default: `$(TT_METAL_HOME)/build_Release`)
- `MODEL_PATH` — path to .gguf model file

## Test inference

```sh
TT_METAL_RUNTIME_ROOT=/home/ubuntu/tt-metal \
  ./build/test_forward /home/ubuntu/qwen3.5-9b-bf16-1x5090/models/Qwen3.5-9B-BF16.gguf "Your prompt here" 128
```

## Reference model (llama.cpp)

For comparison against llama.cpp:

```sh
llama-completion -hf unsloth/Qwen3.5-9B-GGUF:BF16 -p "Your prompt here" -n 128 -ngl 99
```
