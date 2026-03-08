#pragma once

#include "model_config.h"
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <cstdint>

// Forward declarations for tt-metal types
namespace tt::tt_metal {
class Program;
namespace distributed {
class MeshDevice;
class MeshBuffer;
class MeshCommandQueue;
}  // namespace distributed
}  // namespace tt::tt_metal

// Token callback for streaming. Return false to stop generation.
using TokenCallback = std::function<bool(int token_id, const std::string& text)>;

enum StopReason { STOP_EOS, STOP_LENGTH, STOP_CALLBACK };

// Weights for a single full-attention layer.
// Small norm weights: BF16 MeshBuffers on device.
// Large matmul weights: stored as host bf16 vectors, freed after BFP8_B upload.
struct AttentionLayerBuffers {
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> attn_norm;
    std::vector<uint16_t> wqkv_host;         // [qkv_rows, n_embd] BF16 — freed after upload
    std::vector<uint16_t> wo_host;           // [n_head*head_dim, n_embd] BF16 — freed after upload
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> attn_q_norm;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> attn_k_norm;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> post_attn_norm;
    std::vector<uint16_t> ffn_gate_host;     // [n_ff, n_embd] BF16 — freed after upload
    std::vector<uint16_t> ffn_up_host;       // [n_ff, n_embd] BF16 — freed after upload
    std::vector<uint16_t> ffn_down_host;     // [n_embd, n_ff] BF16 — freed after upload
};

// Weights for a single SSM (delta-net) layer.
struct SSMLayerBuffers {
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> attn_norm;
    std::vector<uint16_t> w_combined_host;   // [combined_rows, n_embd] BF16 — freed after upload
    std::vector<float> ssm_a_host;           // [ssm_dt_rank] = [32]
    std::vector<float> ssm_conv1d_host;      // [channels * kernel] = [8192 * 4]
    std::vector<float> ssm_dt_bias_host;     // [ssm_dt_rank] = [32]
    std::vector<float> ssm_norm_host;        // [ssm_head_v_dim] = [128]
    std::vector<uint16_t> ssm_out_host;      // [n_embd, ssm_d_inner] BF16 — freed after upload
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> post_attn_norm;
    std::vector<uint16_t> ffn_gate_host;     // [n_ff, n_embd] BF16 — freed after upload
    std::vector<uint16_t> ffn_up_host;       // [n_ff, n_embd] BF16 — freed after upload
    std::vector<uint16_t> ffn_down_host;     // [n_embd, n_ff] BF16 — freed after upload
};

// Full model weight buffers on device DRAM.
// Large tables (tok_embd, output) stored in host memory to save device DRAM.
struct ModelBuffers {
    // Host-side storage for large lookup tables
    std::vector<uint16_t> tok_embd_host;  // [n_vocab * n_embd] BF16 as raw uint16
    std::vector<uint16_t> output_host;    // [n_vocab * n_embd] BF16 as raw uint16

    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> output_norm;

    AttentionLayerBuffers attn_layers[8];
    SSMLayerBuffers ssm_layers[24];
};

// Load model weights + tokenizer from GGUF, allocate device buffers.
bool load_model_and_tokenizer(const char* model_path, int max_ctx);

// Run generation: prefill prompt_tokens, then decode up to max_tokens.
int generate(const std::vector<int>& prompt_tokens, int max_tokens,
             float temperature, TokenCallback cb, StopReason* stop_reason = nullptr);

// Reset all inference state between requests.
void reset_state();

// Release all device resources and close device. Call before exit.
void shutdown();

// Get the loaded tokenizer for encoding prompts.
class Tokenizer;
const Tokenizer& get_tokenizer();
