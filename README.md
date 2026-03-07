# qwen3.5-9b-1x5090

Custom CUDA inference engine for Qwen3.5-9B on a single RTX 5090, built from scratch with hand-written kernels (no frameworks).

## RunPod (Docker)

Every push to `main` builds a Docker image at `ghcr.io/vibekernels/qwen3.5-9b-1x5090:latest`.

Create a RunPod GPU pod with:

- **Container Image:** `ghcr.io/vibekernels/qwen3.5-9b-1x5090:latest`
- **Container Disk:** 40 GB+ (for model weights)
- **Expose HTTP Ports:** `8080`
- **Expose TCP Ports:** `22` (for SSH)

The container automatically downloads the model from HuggingFace and starts the server on port 8080. Set the `HF_MODEL` environment variable to override the default model (default: `unsloth/Qwen3.5-9B-GGUF:BF16`).

For SSH access, set the `PUBLIC_KEY` environment variable to your public key.

## Building from source

Requires CUDA 12.8+ and an SM 12.0 GPU (RTX 5090).

```sh
make -j$(nproc)
```

This produces two binaries: `qwen-inference` (CLI) and `qwen-server` (HTTP server).

## CLI (`qwen-inference`)

One-shot text completion.

```sh
./qwen-inference -m unsloth/Qwen3.5-9B-GGUF:BF16 -p "The capital of France is" -n 128
```

| Flag | Description | Default |
|------|-------------|---------|
| `-m` | Model path or HuggingFace tag (e.g. `org/repo:quant`) | *required* |
| `-p` | Prompt text | `Hello, world!` |
| `-n` | Max tokens to generate | `128` |
| `-t` | Temperature | `0.8` |
| `--model-dir` | Local model cache directory | `~/.cache/qwen-models` |

## Server (`qwen-server`)

OpenAI-compatible HTTP server.

```sh
./qwen-server -m unsloth/Qwen3.5-9B-GGUF:BF16
```

| Flag | Description | Default |
|------|-------------|---------|
| `-m` | Model path or HuggingFace tag (e.g. `org/repo:quant`) | *required* |
| `--host` | Listen address | `0.0.0.0` |
| `--port` | Listen port | `8080` |
| `--ctx-size` | Max context length | `262144` |
| `--model-dir` | Local model cache directory | `~/.cache/qwen-models` |

### API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |
| `/api/status` | GET | Download/loading progress |

### Example request

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "temperature": 0.7
  }'
```
