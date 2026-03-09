// SPDX-License-Identifier: Apache-2.0
// GGUF weight loader for tt-metal: parse → convert → tilize → upload to DRAM

#include "gguf_loader.h"
#include "engine.h"
#include "model_config.h"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/device.hpp>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <unordered_map>

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using MC = ModelConfig;

// ============================================================================
// GGUF format parsing (pure CPU, no device dependency)
// ============================================================================

enum GGUFType {
    GGUF_TYPE_UINT8 = 0, GGUF_TYPE_INT8 = 1, GGUF_TYPE_UINT16 = 2,
    GGUF_TYPE_INT16 = 3, GGUF_TYPE_UINT32 = 4, GGUF_TYPE_INT32 = 5,
    GGUF_TYPE_FLOAT32 = 6, GGUF_TYPE_BOOL = 7, GGUF_TYPE_STRING = 8,
    GGUF_TYPE_ARRAY = 9, GGUF_TYPE_UINT64 = 10, GGUF_TYPE_INT64 = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

enum GGMLType {
    GGML_TYPE_F32 = 0, GGML_TYPE_F16 = 1, GGML_TYPE_BF16 = 30,
};

struct TensorInfo {
    std::string name;
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;
    uint64_t offset;
    uint64_t size_bytes;
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
        case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL:
            fseek(f, 1, SEEK_CUR); break;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16:
            fseek(f, 2, SEEK_CUR); break;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32:
            fseek(f, 4, SEEK_CUR); break;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64:
            fseek(f, 8, SEEK_CUR); break;
        case GGUF_TYPE_STRING: read_string(f); break;
        case GGUF_TYPE_ARRAY: {
            uint32_t arr_type; uint64_t arr_len;
            fread(&arr_type, 4, 1, f);
            fread(&arr_len, 8, 1, f);
            for (uint64_t i = 0; i < arr_len; i++) skip_value(f, arr_type);
            break;
        }
    }
}

static size_t ggml_type_size(uint32_t type) {
    switch (type) {
        case GGML_TYPE_F32: return 4;
        case GGML_TYPE_F16: return 2;
        case GGML_TYPE_BF16: return 2;
        default: fprintf(stderr, "Unsupported GGML type: %u\n", type); exit(1);
    }
}

// ============================================================================
// Upload helpers
// ============================================================================

// Read raw tensor data from file, convert to bf16 vector
static std::vector<bfloat16> read_tensor_bf16(
    FILE* f, long data_start, const TensorInfo& ti)
{
    uint64_t n_elems = 1;
    for (uint32_t d = 0; d < ti.n_dims; d++) n_elems *= ti.dims[d];

    fseek(f, data_start + ti.offset, SEEK_SET);
    std::vector<bfloat16> result(n_elems);

    if (ti.type == GGML_TYPE_BF16) {
        fread(result.data(), 2, n_elems, f);
    } else if (ti.type == GGML_TYPE_F32) {
        std::vector<float> tmp(n_elems);
        fread(tmp.data(), 4, n_elems, f);
        for (uint64_t i = 0; i < n_elems; i++) {
            result[i] = bfloat16(tmp[i]);
        }
    } else {
        fprintf(stderr, "Cannot convert type %u to bf16\n", ti.type);
        exit(1);
    }
    return result;
}

// Read raw tensor data as f32 (no bf16 round-trip)
static std::vector<float> read_tensor_f32(
    FILE* f, long data_start, const TensorInfo& ti)
{
    uint64_t n_elems = 1;
    for (uint32_t d = 0; d < ti.n_dims; d++) n_elems *= ti.dims[d];

    fseek(f, data_start + ti.offset, SEEK_SET);
    std::vector<float> result(n_elems);

    if (ti.type == GGML_TYPE_F32) {
        fread(result.data(), 4, n_elems, f);
    } else if (ti.type == GGML_TYPE_BF16) {
        std::vector<bfloat16> tmp(n_elems);
        fread(tmp.data(), 2, n_elems, f);
        for (uint64_t i = 0; i < n_elems; i++)
            result[i] = static_cast<float>(tmp[i]);
    } else {
        fprintf(stderr, "Cannot convert type %u to f32\n", ti.type);
        exit(1);
    }
    return result;
}

// Pad row-major bf16 data to tile boundaries, tilize, and upload to DRAM
static std::shared_ptr<MeshBuffer> upload_2d_bf16(
    MeshDevice* device, MeshCommandQueue& cq,
    const bfloat16* data, uint32_t rows, uint32_t cols)
{
    uint32_t rows_padded = ((rows + TILE_HEIGHT - 1) / TILE_HEIGHT) * TILE_HEIGHT;
    uint32_t cols_padded = ((cols + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;
    uint32_t num_tiles = (rows_padded / TILE_HEIGHT) * (cols_padded / TILE_WIDTH);
    uint32_t tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;

    // Pad
    std::vector<bfloat16> padded(rows_padded * cols_padded, bfloat16(0.0f));
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < cols; c++) {
            padded[r * cols_padded + c] = data[r * cols + c];
        }
    }

    // Tilize
    auto tiled = tilize_nfaces(padded, rows_padded, cols_padded);

    // Create DRAM buffer and upload
    DeviceLocalBufferConfig dram_config{.page_size = tile_size, .buffer_type = BufferType::DRAM};
    ReplicatedBufferConfig buf_config{.size = num_tiles * tile_size};
    auto buf = MeshBuffer::create(buf_config, dram_config, device);
    EnqueueWriteMeshBuffer(cq, buf, tiled, false);
    return buf;
}

// Upload a 1D vector as a single-row tiled buffer (1×N padded to tile)
static std::shared_ptr<MeshBuffer> upload_1d_bf16(
    MeshDevice* device, MeshCommandQueue& cq,
    const bfloat16* data, uint32_t len)
{
    return upload_2d_bf16(device, cq, data, 1, len);
}

// Concatenate two weight matrices vertically: [rows_a, cols] and [rows_b, cols] → [rows_a+rows_b, cols]
static std::vector<bfloat16> concat_vertical(
    const bfloat16* a, uint32_t rows_a,
    const bfloat16* b, uint32_t rows_b,
    uint32_t cols)
{
    std::vector<bfloat16> result((rows_a + rows_b) * cols);
    memcpy(result.data(), a, rows_a * cols * sizeof(bfloat16));
    memcpy(result.data() + rows_a * cols, b, rows_b * cols * sizeof(bfloat16));
    return result;
}

// ============================================================================
// Main loader
// ============================================================================

bool load_gguf_weights(
    const std::string& path, ModelBuffers& model,
    MeshDevice* device, MeshCommandQueue& cq)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path.c_str());
        return false;
    }

    // Header
    char magic[4];
    fread(magic, 1, 4, f);
    if (memcmp(magic, "GGUF", 4) != 0) {
        fprintf(stderr, "Not a GGUF file\n");
        fclose(f); return false;
    }
    uint32_t version; fread(&version, 4, 1, f);
    uint64_t n_tensors, n_kv;
    fread(&n_tensors, 8, 1, f);
    fread(&n_kv, 8, 1, f);
    printf("GGUF v%u: %lu tensors, %lu KV pairs\n", version, n_tensors, n_kv);

    // Skip KV metadata
    for (uint64_t i = 0; i < n_kv; i++) {
        read_string(f);
        uint32_t vtype; fread(&vtype, 4, 1, f);
        skip_value(f, vtype);
    }

    // Read tensor info
    std::vector<TensorInfo> tensors(n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        tensors[i].name = read_string(f);
        fread(&tensors[i].n_dims, 4, 1, f);
        memset(tensors[i].dims, 0, sizeof(tensors[i].dims));
        for (uint32_t d = 0; d < tensors[i].n_dims; d++)
            fread(&tensors[i].dims[d], 8, 1, f);
        fread(&tensors[i].type, 4, 1, f);
        fread(&tensors[i].offset, 8, 1, f);
        uint64_t n_elems = 1;
        for (uint32_t d = 0; d < tensors[i].n_dims; d++) n_elems *= tensors[i].dims[d];
        tensors[i].size_bytes = n_elems * ggml_type_size(tensors[i].type);
    }

    long header_end = ftell(f);
    long data_start = ((header_end + 31) / 32) * 32;

    // Name → index map
    std::unordered_map<std::string, size_t> tmap;
    for (size_t i = 0; i < tensors.size(); i++) tmap[tensors[i].name] = i;

    auto get = [&](const std::string& name) -> const TensorInfo& {
        auto it = tmap.find(name);
        if (it == tmap.end()) { fprintf(stderr, "Tensor not found: %s\n", name.c_str()); exit(1); }
        return tensors[it->second];
    };

    auto has = [&](const std::string& name) -> bool {
        return tmap.count(name) > 0;
    };

    // Load a tensor as bf16 vector
    auto load_bf16 = [&](const std::string& name) -> std::vector<bfloat16> {
        return read_tensor_bf16(f, data_start, get(name));
    };

    // Load a tensor as f32 vector (for small f32 tensors like SSM params)
    auto load_f32 = [&](const std::string& name) -> std::vector<float> {
        return read_tensor_f32(f, data_start, get(name));
    };

    printf("Loading weights to device DRAM...\n");

    // ===== Global weights =====
    // token_embd: [n_vocab, n_embd] — stored in HOST memory for lookup
    {
        auto data = load_bf16("token_embd.weight");
        model.tok_embd_host.resize(data.size());
        memcpy(model.tok_embd_host.data(), data.data(), data.size() * sizeof(bfloat16));
        printf("  token_embd: [%d, %d] (host memory, %.1f MB)\n",
               MC::n_vocab, MC::n_embd,
               data.size() * 2.0f / (1024 * 1024));
    }

    // output (LM head): [n_vocab, n_embd] — stored in HOST memory
    {
        auto data = load_bf16("output.weight");
        model.output_host.resize(data.size());
        memcpy(model.output_host.data(), data.data(), data.size() * sizeof(bfloat16));
        printf("  output: [%d, %d] (host memory, %.1f MB)\n",
               MC::n_vocab, MC::n_embd,
               data.size() * 2.0f / (1024 * 1024));
    }

    // output_norm: [n_embd] — stored as bf16 1D
    {
        auto data = load_bf16("output_norm.weight");
        model.output_norm = upload_1d_bf16(device, cq, data.data(), MC::n_embd);
        printf("  output_norm: [%d]\n", MC::n_embd);
    }

    // ===== Per-layer weights =====
    int attn_idx = 0, ssm_idx = 0;
    for (int il = 0; il < MC::n_layers; il++) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "blk.%d.", il);
        auto pname = [&](const char* suffix) -> std::string {
            return std::string(prefix) + suffix;
        };

        if (MC::is_recurrent(il)) {
            auto& lw = model.ssm_layers[ssm_idx];

            // attn_norm: [n_embd] — small, upload to device as BF16
            auto norm_data = load_bf16(pname("attn_norm.weight"));
            lw.attn_norm = upload_1d_bf16(device, cq, norm_data.data(), MC::n_embd);

            // QKV + Gate + Alpha + Beta → combined: [12352, 4096] — store on host
            auto qkv_data = load_bf16(pname("attn_qkv.weight"));
            auto gate_data = load_bf16(pname("attn_gate.weight"));
            auto alpha_data = load_bf16(pname("ssm_alpha.weight"));
            auto beta_data = load_bf16(pname("ssm_beta.weight"));

            uint32_t combined_rows = MC::ssm_conv_channels + MC::ssm_d_inner
                                   + MC::ssm_dt_rank + MC::ssm_dt_rank;
            lw.w_combined_host.resize(combined_rows * MC::n_embd);
            uint16_t* dst = lw.w_combined_host.data();
            size_t off = 0;
            memcpy(dst + off, qkv_data.data(), MC::ssm_conv_channels * MC::n_embd * 2);
            off += MC::ssm_conv_channels * MC::n_embd;
            memcpy(dst + off, gate_data.data(), MC::ssm_d_inner * MC::n_embd * 2);
            off += MC::ssm_d_inner * MC::n_embd;
            memcpy(dst + off, alpha_data.data(), MC::ssm_dt_rank * MC::n_embd * 2);
            off += MC::ssm_dt_rank * MC::n_embd;
            memcpy(dst + off, beta_data.data(), MC::ssm_dt_rank * MC::n_embd * 2);

            // SSM parameters — stored on host as f32
            lw.ssm_a_host = load_f32(pname("ssm_a"));
            // Load conv1d weights and transpose from [channels, 4] to [4, channels]
            // for contiguous AVX-512 loads in the conv1d hot loop
            {
                auto raw = load_f32(pname("ssm_conv1d.weight"));
                int channels = MC::ssm_conv_channels;
                int kernel = MC::ssm_conv_kernel;
                lw.ssm_conv1d_host.resize(raw.size());
                for (int ch = 0; ch < channels; ch++)
                    for (int k = 0; k < kernel; k++)
                        lw.ssm_conv1d_host[k * channels + ch] = raw[ch * kernel + k];
            }
            lw.ssm_dt_bias_host = load_f32(pname("ssm_dt.bias"));
            lw.ssm_norm_host = load_f32(pname("ssm_norm.weight"));

            auto ssm_out_data = load_bf16(pname("ssm_out.weight"));
            lw.ssm_out_host.resize(ssm_out_data.size());
            memcpy(lw.ssm_out_host.data(), ssm_out_data.data(), ssm_out_data.size() * 2);

            // Post-attention norm — small, upload to device as BF16
            auto post_norm = load_bf16(pname("post_attention_norm.weight"));
            lw.post_attn_norm = upload_1d_bf16(device, cq, post_norm.data(), MC::n_embd);

            // FFN: gate and up stored separately on host
            auto ffn_gate = load_bf16(pname("ffn_gate.weight"));
            lw.ffn_gate_host.resize(ffn_gate.size());
            memcpy(lw.ffn_gate_host.data(), ffn_gate.data(), ffn_gate.size() * 2);

            auto ffn_up = load_bf16(pname("ffn_up.weight"));
            lw.ffn_up_host.resize(ffn_up.size());
            memcpy(lw.ffn_up_host.data(), ffn_up.data(), ffn_up.size() * 2);

            auto ffn_down_data = load_bf16(pname("ffn_down.weight"));
            lw.ffn_down_host.resize(ffn_down_data.size());
            memcpy(lw.ffn_down_host.data(), ffn_down_data.data(), ffn_down_data.size() * 2);

            printf("  Layer %d: SSM (delta-net) [%d]\n", il, ssm_idx);
            ssm_idx++;
        } else {
            auto& lw = model.attn_layers[attn_idx];

            // attn_norm — small, upload to device as BF16
            auto norm_data = load_bf16(pname("attn_norm.weight"));
            lw.attn_norm = upload_1d_bf16(device, cq, norm_data.data(), MC::n_embd);

            // QKV: pack Q+K+V → [qkv_rows, n_embd] — store on host
            int q_dim = MC::n_head * MC::head_dim * 2;
            int kv_dim = MC::n_head_kv * MC::head_dim;
            int qkv_rows = q_dim + 2 * kv_dim;

            auto wq = load_bf16(pname("attn_q.weight"));
            auto wk = load_bf16(pname("attn_k.weight"));
            auto wv = load_bf16(pname("attn_v.weight"));

            lw.wqkv_host.resize(qkv_rows * MC::n_embd);
            memcpy(lw.wqkv_host.data(), wq.data(), q_dim * MC::n_embd * 2);
            memcpy(lw.wqkv_host.data() + q_dim * MC::n_embd, wk.data(), kv_dim * MC::n_embd * 2);
            memcpy(lw.wqkv_host.data() + (q_dim + kv_dim) * MC::n_embd, wv.data(), kv_dim * MC::n_embd * 2);

            // Output projection — store on host
            auto wo = load_bf16(pname("attn_output.weight"));
            lw.wo_host.resize(wo.size());
            memcpy(lw.wo_host.data(), wo.data(), wo.size() * 2);

            // Q/K norms — small, upload to device as BF16
            auto qn = load_bf16(pname("attn_q_norm.weight"));
            lw.attn_q_norm = upload_1d_bf16(device, cq, qn.data(), MC::head_dim);
            auto kn = load_bf16(pname("attn_k_norm.weight"));
            lw.attn_k_norm = upload_1d_bf16(device, cq, kn.data(), MC::head_dim);

            // Post-attention norm — small, upload to device as BF16
            auto post_norm = load_bf16(pname("post_attention_norm.weight"));
            lw.post_attn_norm = upload_1d_bf16(device, cq, post_norm.data(), MC::n_embd);

            // FFN: gate and up stored separately on host
            auto ffn_gate = load_bf16(pname("ffn_gate.weight"));
            lw.ffn_gate_host.resize(ffn_gate.size());
            memcpy(lw.ffn_gate_host.data(), ffn_gate.data(), ffn_gate.size() * 2);

            auto ffn_up = load_bf16(pname("ffn_up.weight"));
            lw.ffn_up_host.resize(ffn_up.size());
            memcpy(lw.ffn_up_host.data(), ffn_up.data(), ffn_up.size() * 2);

            auto ffn_down_data = load_bf16(pname("ffn_down.weight"));
            lw.ffn_down_host.resize(ffn_down_data.size());
            memcpy(lw.ffn_down_host.data(), ffn_down_data.data(), ffn_down_data.size() * 2);

            printf("  Layer %d: Attention [%d]\n", il, attn_idx);
            attn_idx++;
        }
    }

    fclose(f);
    printf("All weights loaded to device DRAM.\n");
    return true;
}
