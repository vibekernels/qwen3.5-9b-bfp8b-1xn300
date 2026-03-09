#pragma once

#include <cstdint>

// Model hyperparameters for Qwen3.5-9B (from GGUF metadata)
// Same constants as src/model.h but without CUDA types.
struct ModelConfig {
    static constexpr int n_layers          = 32;
    static constexpr int n_embd            = 4096;
    static constexpr int n_ff              = 12288;
    static constexpr int n_vocab           = 248320;
    static constexpr int n_ctx             = 262144;

    // Attention params (for full attention layers)
    static constexpr int n_head            = 16;
    static constexpr int n_head_kv         = 4;
    static constexpr int head_dim          = 256;
    static constexpr float rms_norm_eps    = 1e-6f;

    // RoPE
    static constexpr float rope_freq_base  = 10000000.0f;
    static constexpr int rope_dim          = 64;

    // SSM / Delta-net params (for recurrent layers)
    static constexpr int ssm_d_inner       = 4096;
    static constexpr int ssm_d_state       = 128;
    static constexpr int ssm_n_group       = 16;
    static constexpr int ssm_dt_rank       = 32;
    static constexpr int ssm_head_v_dim    = ssm_d_inner / ssm_dt_rank; // = 128
    static constexpr int ssm_conv_kernel   = 4;
    static constexpr int ssm_conv_channels = ssm_d_inner + 2 * ssm_n_group * ssm_d_state; // 8192

    // Layer pattern: full attention at (i+1) % 4 == 0
    static constexpr int full_attn_interval = 4;

    static bool is_recurrent(int layer) {
        return (layer + 1) % full_attn_interval != 0;
    }

    static constexpr float attn_scale = 1.0f / 16.0f; // 1/sqrt(head_dim)

    // Tile dimensions for tt-metal (32x32 tiles)
    static constexpr int TILE_H = 32;
    static constexpr int TILE_W = 32;

    // Tiled dimensions (each dimension in number of tiles)
    static constexpr int n_embd_tiles      = n_embd / TILE_W;       // 128
    static constexpr int n_ff_tiles        = n_ff / TILE_W;         // 384
    static constexpr int n_vocab_tiles     = (n_vocab + TILE_W - 1) / TILE_W; // 7760
    static constexpr int head_dim_tiles    = head_dim / TILE_W;     // 8
    static constexpr int ssm_d_state_tiles = ssm_d_state / TILE_W;  // 4

    // Padded vocab to tile boundary
    static constexpr int n_vocab_padded    = n_vocab_tiles * TILE_W; // 248320 (already aligned)

    // BF16 tile size in bytes
    static constexpr int tile_size_bf16    = TILE_H * TILE_W * 2;   // 2048 bytes
    static constexpr int tile_size_f32     = TILE_H * TILE_W * 4;   // 4096 bytes
};
