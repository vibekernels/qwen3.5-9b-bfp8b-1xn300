#pragma once

#include <cuda_bf16.h>
#include <cstdint>

// Model hyperparameters (from GGUF metadata)
struct ModelConfig {
    static constexpr int n_layers          = 32;
    static constexpr int n_embd            = 4096;
    static constexpr int n_ff              = 12288;
    static constexpr int n_vocab           = 248320;
    static constexpr int n_ctx             = 262144;

    // Attention params (for full attention layers)
    static constexpr int n_head            = 16;
    static constexpr int n_head_kv         = 4;
    static constexpr int head_dim          = 256;   // n_embd_head_k = n_embd_head_v
    static constexpr float rms_norm_eps    = 1e-6f;

    // RoPE
    static constexpr float rope_freq_base  = 10000000.0f;
    static constexpr int rope_dim          = 64;    // dimension_count
    static constexpr int rope_sections[4]  = {11, 11, 10, 0};

    // SSM / Delta-net params (for recurrent layers)
    static constexpr int ssm_d_inner       = 4096;
    static constexpr int ssm_d_state       = 128;   // head_k_dim for SSM
    static constexpr int ssm_n_group       = 16;    // num_k_heads for SSM
    static constexpr int ssm_dt_rank       = 32;    // num_v_heads for SSM
    static constexpr int ssm_head_v_dim    = ssm_d_inner / ssm_dt_rank; // = 128
    static constexpr int ssm_conv_kernel   = 4;
    static constexpr int ssm_conv_channels = ssm_d_inner + 2 * ssm_n_group * ssm_d_state; // 4096 + 4096 = 8192

    // Layer pattern: full attention at (i+1) % 4 == 0, i.e., layers 3,7,11,15,19,23,27,31
    static constexpr int full_attn_interval = 4;

    static bool is_recurrent(int layer) {
        return (layer + 1) % full_attn_interval != 0;
    }

    // Attention scale for full attention layers
    static constexpr float attn_scale = 1.0f / 16.0f; // 1/sqrt(head_dim) = 1/sqrt(256) = 1/16
};

// Full attention layer weights (layers 3,7,11,15,19,23,27,31)
struct AttentionLayerWeights {
    float*         attn_norm;        // [n_embd] (F32 in GGUF)
    __nv_bfloat16* wq;               // [n_embd, n_head * head_dim * 2] = [4096, 8192] (Q + gate packed)
    __nv_bfloat16* wk;               // [n_embd, n_head_kv * head_dim] = [4096, 1024]
    __nv_bfloat16* wv;               // [n_embd, n_head_kv * head_dim] = [4096, 1024]
    __nv_bfloat16* wkv;              // packed K+V [n_embd, 2 * n_head_kv * head_dim] = [4096, 2048]
    __nv_bfloat16* wqkv;             // packed Q+Gate+K+V [n_embd, n_head*head_dim*2 + 2*n_head_kv*head_dim] = [4096, 10240]
    __nv_bfloat16* wo;               // [n_head * head_dim, n_embd] = [4096, 4096]
    float*         attn_q_norm;      // [head_dim] = [256] (F32 in GGUF)
    float*         attn_k_norm;      // [head_dim] = [256] (F32 in GGUF)
    float*         post_attn_norm;   // [n_embd] (F32 in GGUF)

    // FFN
    __nv_bfloat16* ffn_gate;         // [n_embd, n_ff] = [4096, 12288]
    __nv_bfloat16* ffn_up;           // [n_embd, n_ff] = [4096, 12288]
    __nv_bfloat16* ffn_down;         // [n_ff, n_embd] = [12288, 4096]
    __nv_bfloat16* ffn_gate_up;      // [n_embd, 2*n_ff] = [4096, 24576] packed gate+up
};

// SSM (delta-net) layer weights
struct SSMLayerWeights {
    float*         attn_norm;        // [n_embd] (F32 in GGUF)
    __nv_bfloat16* wqkv;             // [n_embd, ssm_conv_channels] = [4096, 8192]
    __nv_bfloat16* wqkv_gate;        // [n_embd, ssm_d_inner] = [4096, 4096]  (attn_gate in GGUF)
    __nv_bfloat16* w_combined;       // packed [n_embd, 8192+4096+32+32] = [4096, 12352]

    // SSM params
    float*         ssm_a;            // [ssm_dt_rank] = [32]
    float*         ssm_conv1d;       // [ssm_conv_kernel, ssm_conv_channels] = [4, 8192]
    float*         ssm_dt_bias;      // [ssm_dt_rank] = [32]
    __nv_bfloat16* ssm_alpha;        // [n_embd, ssm_dt_rank] = [4096, 32]
    __nv_bfloat16* ssm_beta;         // [n_embd, ssm_dt_rank] = [4096, 32]
    float*         ssm_norm;         // [ssm_d_state] = [128] (F32 in GGUF)
    __nv_bfloat16* ssm_out;          // [ssm_d_inner, n_embd] = [4096, 4096]

    float*         post_attn_norm;   // [n_embd] (F32 in GGUF)

    // FFN
    __nv_bfloat16* ffn_gate;         // [n_embd, n_ff]
    __nv_bfloat16* ffn_up;           // [n_embd, n_ff]
    __nv_bfloat16* ffn_down;         // [n_ff, n_embd]
    __nv_bfloat16* ffn_gate_up;      // [n_embd, 2*n_ff] packed gate+up
};

// Combined layer (union-like, selected by is_recurrent)
struct LayerWeights {
    bool is_recurrent;
    union {
        AttentionLayerWeights attn;
        SSMLayerWeights ssm;
    };

    // Common weights (duplicated in both structs for convenience)
    __nv_bfloat16* attn_norm;
    __nv_bfloat16* post_attn_norm;
    __nv_bfloat16* ffn_gate;
    __nv_bfloat16* ffn_up;
    __nv_bfloat16* ffn_down;
};

// Full model weights
struct Model {
    __nv_bfloat16* tok_embd;         // [n_vocab, n_embd] = [248320, 4096]
    __nv_bfloat16* output;           // [n_embd, n_vocab] = [4096, 248320]
    float*         output_norm;      // [n_embd] (F32 in GGUF)

    // Per-layer weights
    AttentionLayerWeights attn_layers[8];  // For the 8 full-attention layers
    SSMLayerWeights ssm_layers[24];        // For the 24 SSM layers

    // Flat index: layer_index -> {type, index_within_type}
    int layer_type[ModelConfig::n_layers];     // 0 = attention, 1 = SSM
    int layer_subidx[ModelConfig::n_layers];   // index within attn_layers or ssm_layers

    // Inference state buffers (hidden state in f32 for precision, like ggml)
    float*         hidden_state;     // [max_tokens, n_embd]  working buffer (f32 residual stream)
    __nv_bfloat16* hidden_bf16;      // [max_tokens, n_embd]  bf16 copy for GEMM inputs
    __nv_bfloat16* norm_out;         // [max_tokens, n_embd]
    float*         norm_out_f32;     // [max_tokens, n_embd]  f32 norm output for ggml-matching precision
    __nv_bfloat16* attn_out;         // [max_tokens, n_embd]
    float*         gemm_out;         // [max_tokens, max(n_embd, n_ff)] f32 GEMM output
    float*         gemm_out2;        // [max_tokens, max(n_embd, n_ff)] f32 GEMM output
    __nv_bfloat16* ffn_buf;          // [max_tokens, n_ff]
    __nv_bfloat16* ffn_buf2;        // [max_tokens, n_ff]
    float*         logits_f32;       // [max_tokens, n_vocab]

    // SSM f32 intermediates (matches ggml precision)
    float*         ssm_proj_f32;     // [max_tokens, ssm_conv_channels] f32 QKV projection for conv

    // Attention-specific buffers
    __nv_bfloat16* qkv_buf;          // large temp buffer for QKV projections
    float*         attn_scores;      // [n_head, max_tokens, max_kv_len] for attention

    // KV cache for full attention layers
    __nv_bfloat16* k_cache[8];       // [max_kv_len, n_head_kv * head_dim] per attention layer
    __nv_bfloat16* v_cache[8];       // [max_kv_len, n_head_kv * head_dim] per attention layer

    // SSM state for delta-net layers
    float* ssm_conv_state[24];       // [(conv_kernel-1) * conv_channels] per SSM layer
    float* ssm_recurrent_state[24];  // [num_v_heads * head_v_dim * head_v_dim] = [32*128*128] per SSM layer

    // Pre-allocated SSM temp buffers (avoid cudaMalloc/Free per token)
    float* ssm_gate_buf;             // [max_tokens * ssm_dt_rank]
    float* ssm_beta_buf;             // [max_tokens * ssm_dt_rank]
    float* ssm_conv_out_buf;         // [max_tokens * ssm_conv_channels]
    float* ssm_q_rep_buf;            // [ssm_dt_rank * ssm_d_state]
    float* ssm_k_rep_buf;            // [ssm_dt_rank * ssm_d_state]
    float* ssm_delta_out_buf;        // [ssm_dt_rank * ssm_head_v_dim]

    int max_tokens;                  // max batch tokens allocated
    int max_kv_len;                  // max KV cache length
    int kv_len;                      // current KV cache length

    // CUDA graph for decode acceleration
    int* d_kv_len;                   // kv_len on device (for graph-captured attention)
    cudaStream_t compute_stream;     // non-default stream for graph capture
    cudaGraph_t decode_graph;
    cudaGraphExec_t decode_graph_exec;
    bool decode_graph_captured;

    void init_layer_mapping() {
        int attn_idx = 0, ssm_idx = 0;
        for (int i = 0; i < ModelConfig::n_layers; i++) {
            if (ModelConfig::is_recurrent(i)) {
                layer_type[i] = 1;
                layer_subidx[i] = ssm_idx++;
            } else {
                layer_type[i] = 0;
                layer_subidx[i] = attn_idx++;
            }
        }
    }
};
