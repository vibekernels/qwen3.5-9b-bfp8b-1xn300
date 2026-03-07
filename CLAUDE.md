# Project goals

The point of this project is to write a Tenstorrent accelerator kernel. ALL computation should run on the Tenstorrent N300 device (two Wormhole chips). Do not fall back to pure CPU compute — use the device for matmuls and as many operations as possible. If quantization is needed, do not go below INT8.

# Project structure

- `tt_metal/host/engine.cpp` — inference engine: forward pass, `generate()`, `load_model_and_tokenizer()`
- `tt_metal/host/engine.h` — public API: `generate()`, `load_model_and_tokenizer()`, `reset_state()`, `shutdown()`, `get_tokenizer()`
- `tt_metal/host/gguf_loader.{h,cpp}` — GGUF weight loading into device DRAM MeshBuffers
- `tt_metal/host/model_config.h` — model hyperparameters and tile dimensions
- `tt_metal/kernels/` — Tensix compute/dataflow kernels (currently unused, matmuls via ttnn API)
- `tt_metal/tests/` — test suite (device, matmul, weight loading, forward pass)
- `tt_metal/CMakeLists.txt` — CMake build system
- `src/tokenizer.{h,cpp}` — BPE tokenizer (GPT-2 byte-level)
- `src/download.{h,cpp}` — HuggingFace model download

## Build & test

```sh
cd tt_metal && mkdir -p build && cd build
cmake .. -DTT_METAL_BUILD=/home/ubuntu/tt-metal/build_Release -DCMAKE_CXX_COMPILER=clang++-20
make -j$(nproc)
```

## Test inference

Requires `sudo` and `TT_METAL_RUNTIME_ROOT`:

```sh
sudo TT_METAL_RUNTIME_ROOT=/home/ubuntu/tt-metal \
  ./build/test_forward /path/to/Qwen3.5-9B-BF16.gguf "Your prompt here" 128
```

## Reference model (llama.cpp)

For comparison against llama.cpp:

```sh
llama-completion -hf unsloth/Qwen3.5-9B-GGUF:BF16 -p "Your prompt here" -n 128 -ngl 99
```
