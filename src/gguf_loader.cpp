#include "gguf_loader.h"
#include "utils.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

// GGUF format constants
enum GGUFType {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

enum GGMLType {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_BF16 = 30,
};

struct TensorInfo {
    std::string name;
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;
    uint64_t offset;
    uint64_t size_bytes; // computed
};

static std::string read_string(FILE* f) {
    uint64_t len;
    fread(&len, 8, 1, f);
    std::string s(len, '\0');
    fread(&s[0], 1, len, f);
    return s;
}

static void skip_value(FILE* f, uint32_t vtype) {
    switch (vtype) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            fseek(f, 1, SEEK_CUR); break;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            fseek(f, 2, SEEK_CUR); break;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            fseek(f, 4, SEEK_CUR); break;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            fseek(f, 8, SEEK_CUR); break;
        case GGUF_TYPE_STRING:
            read_string(f); break;
        case GGUF_TYPE_ARRAY: {
            uint32_t arr_type;
            uint64_t arr_len;
            fread(&arr_type, 4, 1, f);
            fread(&arr_len, 8, 1, f);
            for (uint64_t i = 0; i < arr_len; i++) {
                skip_value(f, arr_type);
            }
            break;
        }
    }
}

static size_t ggml_type_size(uint32_t type) {
    switch (type) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:  return 2;
        case GGML_TYPE_BF16: return 2;
        default:
            fprintf(stderr, "Unsupported GGML type: %u\n", type);
            exit(1);
    }
}

// Upload a tensor from host memory to GPU as bf16
// Handles f32->bf16 conversion if needed
static __nv_bfloat16* upload_tensor_bf16(const void* host_data, uint64_t n_elements, uint32_t src_type) {
    __nv_bfloat16* gpu_ptr = cuda_alloc<__nv_bfloat16>(n_elements);

    if (src_type == GGML_TYPE_BF16) {
        // Direct upload
        cuda_upload(gpu_ptr, (const __nv_bfloat16*)host_data, n_elements);
    } else if (src_type == GGML_TYPE_F32) {
        // Convert f32 -> bf16 on host, then upload
        const float* src = (const float*)host_data;
        std::vector<__nv_bfloat16> buf(n_elements);
        for (uint64_t i = 0; i < n_elements; i++) {
            buf[i] = __float2bfloat16(src[i]);
        }
        cuda_upload(gpu_ptr, buf.data(), n_elements);
    } else {
        fprintf(stderr, "Cannot convert type %u to bf16\n", src_type);
        exit(1);
    }
    return gpu_ptr;
}

// Upload a tensor as f32 (for SSM params that need f32 precision)
static float* upload_tensor_f32(const void* host_data, uint64_t n_elements, uint32_t src_type) {
    float* gpu_ptr = cuda_alloc<float>(n_elements);

    if (src_type == GGML_TYPE_F32) {
        cuda_upload(gpu_ptr, (const float*)host_data, n_elements);
    } else {
        fprintf(stderr, "Expected f32 tensor, got type %u\n", src_type);
        exit(1);
    }
    return gpu_ptr;
}

bool load_model(const std::string& path, Model& model) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path.c_str());
        return false;
    }

    // Read header
    char magic[4];
    fread(magic, 1, 4, f);
    if (memcmp(magic, "GGUF", 4) != 0) {
        fprintf(stderr, "Not a GGUF file\n");
        fclose(f);
        return false;
    }

    uint32_t version;
    fread(&version, 4, 1, f);
    printf("GGUF version: %u\n", version);

    uint64_t n_tensors, n_kv;
    fread(&n_tensors, 8, 1, f);
    fread(&n_kv, 8, 1, f);
    printf("Tensors: %lu, KV pairs: %lu\n", n_tensors, n_kv);

    // Skip KV metadata
    for (uint64_t i = 0; i < n_kv; i++) {
        read_string(f); // key
        uint32_t vtype;
        fread(&vtype, 4, 1, f);
        skip_value(f, vtype);
    }

    // Read tensor info
    std::vector<TensorInfo> tensors(n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        tensors[i].name = read_string(f);
        fread(&tensors[i].n_dims, 4, 1, f);
        memset(tensors[i].dims, 0, sizeof(tensors[i].dims));
        for (uint32_t d = 0; d < tensors[i].n_dims; d++) {
            fread(&tensors[i].dims[d], 8, 1, f);
        }
        fread(&tensors[i].type, 4, 1, f);
        fread(&tensors[i].offset, 8, 1, f);

        // Compute total elements and size
        uint64_t n_elems = 1;
        for (uint32_t d = 0; d < tensors[i].n_dims; d++) {
            n_elems *= tensors[i].dims[d];
        }
        tensors[i].size_bytes = n_elems * ggml_type_size(tensors[i].type);
    }

    // Data section starts at alignment boundary after header
    long header_end = ftell(f);
    const int ALIGNMENT = 32;
    long data_start = ((header_end + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;

    // Build name -> tensor map
    std::unordered_map<std::string, size_t> tensor_map;
    for (size_t i = 0; i < tensors.size(); i++) {
        tensor_map[tensors[i].name] = i;
    }

    // Helper to load a tensor by name
    auto get_tensor = [&](const std::string& name) -> const TensorInfo& {
        auto it = tensor_map.find(name);
        if (it == tensor_map.end()) {
            fprintf(stderr, "Tensor not found: %s\n", name.c_str());
            exit(1);
        }
        return tensors[it->second];
    };

    auto has_tensor = [&](const std::string& name) -> bool {
        return tensor_map.find(name) != tensor_map.end();
    };

    // Read and upload a tensor as bf16
    auto load_bf16 = [&](const std::string& name) -> __nv_bfloat16* {
        const TensorInfo& ti = get_tensor(name);
        uint64_t n_elems = 1;
        for (uint32_t d = 0; d < ti.n_dims; d++) n_elems *= ti.dims[d];

        std::vector<uint8_t> buf(ti.size_bytes);
        fseek(f, data_start + ti.offset, SEEK_SET);
        fread(buf.data(), 1, ti.size_bytes, f);

        return upload_tensor_bf16(buf.data(), n_elems, ti.type);
    };

    // Read and upload a tensor as f32
    auto load_f32 = [&](const std::string& name) -> float* {
        const TensorInfo& ti = get_tensor(name);
        uint64_t n_elems = 1;
        for (uint32_t d = 0; d < ti.n_dims; d++) n_elems *= ti.dims[d];

        std::vector<uint8_t> buf(ti.size_bytes);
        fseek(f, data_start + ti.offset, SEEK_SET);
        fread(buf.data(), 1, ti.size_bytes, f);

        return upload_tensor_f32(buf.data(), n_elems, ti.type);
    };

    printf("Loading model weights...\n");

    // Global weights
    model.tok_embd    = load_bf16("token_embd.weight");
    model.output      = load_bf16("output.weight");
    model.output_norm = load_f32("output_norm.weight");

    // Initialize layer mapping
    model.init_layer_mapping();

    // Load per-layer weights
    for (int il = 0; il < ModelConfig::n_layers; il++) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "blk.%d.", il);
        auto pname = [&](const char* suffix) -> std::string {
            return std::string(prefix) + suffix;
        };

        if (ModelConfig::is_recurrent(il)) {
            // SSM / delta-net layer
            int si = model.layer_subidx[il];
            SSMLayerWeights& lw = model.ssm_layers[si];

            lw.attn_norm     = load_f32(pname("attn_norm.weight"));
            lw.wqkv          = load_bf16(pname("attn_qkv.weight"));
            lw.wqkv_gate     = load_bf16(pname("attn_gate.weight"));
            lw.ssm_a         = load_f32(pname("ssm_a"));
            lw.ssm_conv1d    = load_f32(pname("ssm_conv1d.weight"));
            lw.ssm_dt_bias   = load_f32(pname("ssm_dt.bias"));
            lw.ssm_alpha     = load_bf16(pname("ssm_alpha.weight"));
            lw.ssm_beta      = load_bf16(pname("ssm_beta.weight"));
            lw.ssm_norm      = load_f32(pname("ssm_norm.weight"));
            lw.ssm_out       = load_bf16(pname("ssm_out.weight"));
            lw.post_attn_norm= load_f32(pname("post_attention_norm.weight"));
            lw.ffn_gate      = load_bf16(pname("ffn_gate.weight"));
            lw.ffn_up        = load_bf16(pname("ffn_up.weight"));
            lw.ffn_down      = load_bf16(pname("ffn_down.weight"));

            printf("  Layer %d: SSM (delta-net)\n", il);
        } else {
            // Full attention layer
            int ai = model.layer_subidx[il];
            AttentionLayerWeights& lw = model.attn_layers[ai];

            lw.attn_norm     = load_f32(pname("attn_norm.weight"));
            lw.wq            = load_bf16(pname("attn_q.weight"));
            lw.wk            = load_bf16(pname("attn_k.weight"));
            lw.wv            = load_bf16(pname("attn_v.weight"));
            lw.wo            = load_bf16(pname("attn_output.weight"));
            lw.attn_q_norm   = load_f32(pname("attn_q_norm.weight"));
            lw.attn_k_norm   = load_f32(pname("attn_k_norm.weight"));
            lw.post_attn_norm= load_f32(pname("post_attention_norm.weight"));
            lw.ffn_gate      = load_bf16(pname("ffn_gate.weight"));
            lw.ffn_up        = load_bf16(pname("ffn_up.weight"));
            lw.ffn_down      = load_bf16(pname("ffn_down.weight"));

            printf("  Layer %d: Full attention\n", il);
        }
    }

    fclose(f);

    // Pack attention K+V weights for fused GEMM
    // Also pack Q+Gate+K+V into single wqkv for decode (saves 1 GEMV call per layer)
    {
        int kv_dim = ModelConfig::n_head_kv * ModelConfig::head_dim;  // 1024
        int q_dim = ModelConfig::n_head * ModelConfig::head_dim * 2;  // 8192 (Q+Gate)
        int k = ModelConfig::n_embd;
        for (int il = 0; il < ModelConfig::n_layers; il++) {
            if (ModelConfig::is_recurrent(il)) continue;
            int ai = model.layer_subidx[il];
            auto& lw = model.attn_layers[ai];
            lw.wkv = cuda_alloc<__nv_bfloat16>(2 * kv_dim * k);
            CUDA_CHECK(cudaMemcpy(lw.wkv, lw.wk,
                (size_t)kv_dim * k * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(lw.wkv + (size_t)kv_dim * k, lw.wv,
                (size_t)kv_dim * k * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));

            // Pack wqkv = [Q+Gate(8192) | K+V(2048)] = [10240, 4096]
            int qkv_dim = q_dim + 2 * kv_dim;  // 10240
            lw.wqkv = cuda_alloc<__nv_bfloat16>((size_t)qkv_dim * k);
            CUDA_CHECK(cudaMemcpy(lw.wqkv, lw.wq,
                (size_t)q_dim * k * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(lw.wqkv + (size_t)q_dim * k, lw.wkv,
                (size_t)(2 * kv_dim) * k * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
        }
    }

    // Pack SSM combined weights: QKV(8192) + Z(4096) + alpha(32) + beta(32) = 12352
    {
        int combined_n = ModelConfig::ssm_conv_channels + ModelConfig::ssm_d_inner
                       + ModelConfig::ssm_dt_rank + ModelConfig::ssm_dt_rank;  // 12352
        int k = ModelConfig::n_embd;  // 4096

        for (int il = 0; il < ModelConfig::n_layers; il++) {
            if (!ModelConfig::is_recurrent(il)) continue;
            int si = model.layer_subidx[il];
            auto& lw = model.ssm_layers[si];

            lw.w_combined = cuda_alloc<__nv_bfloat16>((size_t)combined_n * k);

            size_t off = 0;
            // QKV: [8192, 4096]
            CUDA_CHECK(cudaMemcpy(lw.w_combined + off, lw.wqkv,
                (size_t)ModelConfig::ssm_conv_channels * k * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToDevice));
            off += (size_t)ModelConfig::ssm_conv_channels * k;

            // Z gate: [4096, 4096]
            CUDA_CHECK(cudaMemcpy(lw.w_combined + off, lw.wqkv_gate,
                (size_t)ModelConfig::ssm_d_inner * k * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToDevice));
            off += (size_t)ModelConfig::ssm_d_inner * k;

            // Alpha: [32, 4096]
            CUDA_CHECK(cudaMemcpy(lw.w_combined + off, lw.ssm_alpha,
                (size_t)ModelConfig::ssm_dt_rank * k * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToDevice));
            off += (size_t)ModelConfig::ssm_dt_rank * k;

            // Beta: [32, 4096]
            CUDA_CHECK(cudaMemcpy(lw.w_combined + off, lw.ssm_beta,
                (size_t)ModelConfig::ssm_dt_rank * k * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToDevice));
        }
    }

    // Pack FFN gate+up weights for fused GEMM
    // gate: [n_ff, n_embd], up: [n_ff, n_embd] → packed: [2*n_ff, n_embd]
    int n_ff = ModelConfig::n_ff;
    int n_embd = ModelConfig::n_embd;
    for (int il = 0; il < ModelConfig::n_layers; il++) {
        if (ModelConfig::is_recurrent(il)) {
            int si = model.layer_subidx[il];
            auto& lw = model.ssm_layers[si];
            lw.ffn_gate_up = cuda_alloc<__nv_bfloat16>(2 * n_ff * n_embd);
            CUDA_CHECK(cudaMemcpy(lw.ffn_gate_up, lw.ffn_gate,
                (size_t)n_ff * n_embd * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(lw.ffn_gate_up + (size_t)n_ff * n_embd, lw.ffn_up,
                (size_t)n_ff * n_embd * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
        } else {
            int ai = model.layer_subidx[il];
            auto& lw = model.attn_layers[ai];
            lw.ffn_gate_up = cuda_alloc<__nv_bfloat16>(2 * n_ff * n_embd);
            CUDA_CHECK(cudaMemcpy(lw.ffn_gate_up, lw.ffn_gate,
                (size_t)n_ff * n_embd * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(lw.ffn_gate_up + (size_t)n_ff * n_embd, lw.ffn_up,
                (size_t)n_ff * n_embd * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
        }
    }

    printf("Model loaded successfully.\n");

    return true;
}

void free_model(Model& model) {
    // Note: in production, track and free all allocations
    // For now, process exit handles cleanup
}
