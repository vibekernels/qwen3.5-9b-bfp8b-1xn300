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
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using MC = ModelConfig;

// ============================================================================
// GGUF format constants
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
    GGML_TYPE_TT_BFP8B_TILED = 200,
};

// ============================================================================
// mmap-based cursor — no fseek, no buffer flushes
// ============================================================================

struct MmapCursor {
    const uint8_t* base;
    size_t pos;

    uint32_t read_u32() { uint32_t v; memcpy(&v, base+pos, 4); pos+=4; return v; }
    uint64_t read_u64() { uint64_t v; memcpy(&v, base+pos, 8); pos+=8; return v; }
    std::string read_string() {
        uint64_t len = read_u64();
        std::string s(reinterpret_cast<const char*>(base + pos), len);
        pos += len;
        return s;
    }
    void skip(size_t n) { pos += n; }
};

static void skip_value(MmapCursor& c, uint32_t vtype) {
    switch (vtype) {
        case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: c.skip(1); break;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16:                    c.skip(2); break;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: c.skip(4); break;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: c.skip(8); break;
        case GGUF_TYPE_STRING: c.read_string(); break;
        case GGUF_TYPE_ARRAY: {
            uint32_t arr_type = c.read_u32();
            uint64_t arr_len  = c.read_u64();
            for (uint64_t i = 0; i < arr_len; i++) skip_value(c, arr_type);
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
// mmap helpers: open file and parse GGUF header
// ============================================================================

static bool mmap_gguf_file(const std::string& path, void*& mapped, size_t& file_size) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Cannot open %s\n", path.c_str());
        return false;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "Cannot stat %s\n", path.c_str());
        close(fd); return false;
    }
    file_size = (size_t)st.st_size;
    mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "mmap failed for %s\n", path.c_str());
        return false;
    }
    madvise(mapped, file_size, MADV_SEQUENTIAL);
    return true;
}

static bool parse_gguf_header(const uint8_t* base, size_t file_size,
                               std::unordered_map<std::string, GGUFTensorInfo>& tmap,
                               size_t& data_start) {
    MmapCursor c{base, 0};

    char magic[4];
    memcpy(magic, c.base, 4); c.pos = 4;
    if (memcmp(magic, "GGUF", 4) != 0) {
        fprintf(stderr, "Not a GGUF file\n");
        return false;
    }
    uint32_t version  = c.read_u32();
    uint64_t n_tensors = c.read_u64();
    uint64_t n_kv      = c.read_u64();
    printf("GGUF v%u: %lu tensors, %lu KV pairs\n", version, n_tensors, n_kv);

    for (uint64_t i = 0; i < n_kv; i++) {
        c.read_string();
        uint32_t vtype = c.read_u32();
        skip_value(c, vtype);
    }

    for (uint64_t i = 0; i < n_tensors; i++) {
        GGUFTensorInfo ti;
        ti.name = c.read_string();
        ti.n_dims = c.read_u32();
        memset(ti.dims, 0, sizeof(ti.dims));
        for (uint32_t d = 0; d < ti.n_dims; d++) ti.dims[d] = c.read_u64();
        ti.type   = c.read_u32();
        ti.offset = c.read_u64();
        uint64_t n_elems = 1;
        for (uint32_t d = 0; d < ti.n_dims; d++) n_elems *= ti.dims[d];
        if (ti.type == GGML_TYPE_TT_BFP8B_TILED && ti.n_dims == 2) {
            uint64_t M = ti.dims[1], K = ti.dims[0];
            ti.size_bytes = (M / TILE_HEIGHT) * (K / TILE_WIDTH) * BFLOAT8_B_TILE_HW;
        } else {
            ti.size_bytes = n_elems * ggml_type_size(ti.type);
        }
        tmap[ti.name] = std::move(ti);
    }

    size_t header_end = c.pos;
    data_start = ((header_end + 31) / 32) * 32;
    return true;
}

// ============================================================================
// Tensor load helpers (operate on the mmap'd data section)
// ============================================================================

static std::vector<bfloat16> do_load_bf16(const uint8_t* base, size_t data_start,
                                           const GGUFTensorInfo& ti)
{
    uint64_t n_elems = 1;
    for (uint32_t d = 0; d < ti.n_dims; d++) n_elems *= ti.dims[d];
    const uint8_t* src = base + data_start + ti.offset;
    std::vector<bfloat16> result(n_elems);
    if (ti.type == GGML_TYPE_BF16) {
        memcpy(result.data(), src, n_elems * 2);
    } else if (ti.type == GGML_TYPE_F32) {
        const float* fsrc = reinterpret_cast<const float*>(src);
        for (uint64_t i = 0; i < n_elems; i++) result[i] = bfloat16(fsrc[i]);
    } else {
        fprintf(stderr, "Cannot convert type %u to bf16\n", ti.type); exit(1);
    }
    return result;
}

static std::vector<float> do_load_f32(const uint8_t* base, size_t data_start,
                                       const GGUFTensorInfo& ti)
{
    uint64_t n_elems = 1;
    for (uint32_t d = 0; d < ti.n_dims; d++) n_elems *= ti.dims[d];
    const uint8_t* src = base + data_start + ti.offset;
    std::vector<float> result(n_elems);
    if (ti.type == GGML_TYPE_F32) {
        memcpy(result.data(), src, n_elems * 4);
    } else if (ti.type == GGML_TYPE_BF16) {
        const bfloat16* bsrc = reinterpret_cast<const bfloat16*>(src);
        for (uint64_t i = 0; i < n_elems; i++) result[i] = static_cast<float>(bsrc[i]);
    } else {
        fprintf(stderr, "Cannot convert type %u to f32\n", ti.type); exit(1);
    }
    return result;
}

static std::vector<uint32_t> do_load_packed(const uint8_t* base, size_t data_start,
                                             const GGUFTensorInfo& ti)
{
    const uint8_t* src = base + data_start + ti.offset;
    std::vector<uint32_t> result(ti.size_bytes / sizeof(uint32_t));
    memcpy(result.data(), src, ti.size_bytes);
    return result;
}

// ============================================================================
// GGUFContext methods
// ============================================================================

void GGUFContext::close() {
    if (mapped) {
        munmap(mapped, file_size);
        mapped = nullptr;
    }
}

bool GGUFContext::has(const std::string& name) const {
    return tmap.count(name) > 0;
}

std::vector<uint32_t> GGUFContext::load_packed(const std::string& name) const {
    auto it = tmap.find(name);
    if (it == tmap.end()) { fprintf(stderr, "Tensor not found: %s\n", name.c_str()); exit(1); }
    return do_load_packed(reinterpret_cast<const uint8_t*>(mapped), data_start, it->second);
}

void GGUFContext::load_packed_into(const std::string& name, std::vector<uint32_t>& dst) const {
    auto it = tmap.find(name);
    if (it == tmap.end()) { fprintf(stderr, "Tensor not found: %s\n", name.c_str()); exit(1); }
    const auto& ti = it->second;
    size_t n_words = ti.size_bytes / sizeof(uint32_t);
    dst.resize(n_words);
    const uint8_t* src = reinterpret_cast<const uint8_t*>(mapped) + data_start + ti.offset;
    memcpy(dst.data(), src, ti.size_bytes);
}

std::vector<bfloat16> GGUFContext::load_bf16(const std::string& name) const {
    auto it = tmap.find(name);
    if (it == tmap.end()) { fprintf(stderr, "Tensor not found: %s\n", name.c_str()); exit(1); }
    return do_load_bf16(reinterpret_cast<const uint8_t*>(mapped), data_start, it->second);
}

void GGUFContext::load_bf16_into(const std::string& name, std::vector<bfloat16>& dst) const {
    auto it = tmap.find(name);
    if (it == tmap.end()) { fprintf(stderr, "Tensor not found: %s\n", name.c_str()); exit(1); }
    const auto& ti = it->second;
    uint64_t n_elems = 1;
    for (uint32_t d = 0; d < ti.n_dims; d++) n_elems *= ti.dims[d];
    dst.resize(n_elems);
    const uint8_t* src = reinterpret_cast<const uint8_t*>(mapped) + data_start + ti.offset;
    if (ti.type == GGML_TYPE_BF16) {
        memcpy(dst.data(), src, n_elems * 2);
    } else if (ti.type == GGML_TYPE_F32) {
        const float* fsrc = reinterpret_cast<const float*>(src);
        for (uint64_t i = 0; i < n_elems; i++) dst[i] = bfloat16(fsrc[i]);
    } else {
        fprintf(stderr, "Cannot convert type %u to bf16\n", ti.type); exit(1);
    }
}

void GGUFContext::load_bf16_raw_into(const std::string& name, std::vector<uint16_t>& dst) const {
    auto it = tmap.find(name);
    if (it == tmap.end()) { fprintf(stderr, "Tensor not found: %s\n", name.c_str()); exit(1); }
    const auto& ti = it->second;
    uint64_t n_elems = 1;
    for (uint32_t d = 0; d < ti.n_dims; d++) n_elems *= ti.dims[d];
    dst.resize(n_elems);
    const uint8_t* src = reinterpret_cast<const uint8_t*>(mapped) + data_start + ti.offset;
    if (ti.type == GGML_TYPE_BF16) {
        memcpy(dst.data(), src, n_elems * 2);
    } else if (ti.type == GGML_TYPE_F32) {
        const float* fsrc = reinterpret_cast<const float*>(src);
        for (uint64_t i = 0; i < n_elems; i++) {
            uint32_t bits; memcpy(&bits, &fsrc[i], 4);
            dst[i] = static_cast<uint16_t>(bits >> 16);
        }
    } else {
        fprintf(stderr, "Cannot convert type %u to bf16_raw\n", ti.type); exit(1);
    }
}

std::vector<float> GGUFContext::load_f32(const std::string& name) const {
    auto it = tmap.find(name);
    if (it == tmap.end()) { fprintf(stderr, "Tensor not found: %s\n", name.c_str()); exit(1); }
    return do_load_f32(reinterpret_cast<const uint8_t*>(mapped), data_start, it->second);
}

// ============================================================================
// Upload helpers (same as before)
// ============================================================================

static std::shared_ptr<MeshBuffer> upload_2d_bf16(
    MeshDevice* device, MeshCommandQueue& cq,
    const bfloat16* data, uint32_t rows, uint32_t cols)
{
    uint32_t rows_padded = ((rows + TILE_HEIGHT - 1) / TILE_HEIGHT) * TILE_HEIGHT;
    uint32_t cols_padded = ((cols + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;
    uint32_t num_tiles = (rows_padded / TILE_HEIGHT) * (cols_padded / TILE_WIDTH);
    uint32_t tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;

    std::vector<bfloat16> padded(rows_padded * cols_padded, bfloat16(0.0f));
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t c = 0; c < cols; c++)
            padded[r * cols_padded + c] = data[r * cols + c];

    auto tiled = tilize_nfaces(padded, rows_padded, cols_padded);
    DeviceLocalBufferConfig dram_config{.page_size = tile_size, .buffer_type = BufferType::DRAM};
    ReplicatedBufferConfig buf_config{.size = num_tiles * tile_size};
    auto buf = MeshBuffer::create(buf_config, dram_config, device);
    EnqueueWriteMeshBuffer(cq, buf, tiled, false);
    return buf;
}

static std::shared_ptr<MeshBuffer> upload_1d_bf16(
    MeshDevice* device, MeshCommandQueue& cq,
    const bfloat16* data, uint32_t len)
{
    return upload_2d_bf16(device, cq, data, 1, len);
}

// ============================================================================
// open_gguf: open file, parse header, load global + per-layer SMALL weights.
// Large packed tensors are NOT loaded here — use ctx.load_packed() on-demand.
// ============================================================================

bool open_gguf(
    const std::string& path, ModelBuffers& model,
    MeshDevice* device, MeshCommandQueue& cq,
    GGUFContext& ctx)
{
    void* mapped = nullptr;
    size_t file_size = 0;
    if (!mmap_gguf_file(path, mapped, file_size)) return false;

    const uint8_t* base = reinterpret_cast<const uint8_t*>(mapped);
    if (!parse_gguf_header(base, file_size, ctx.tmap, ctx.data_start)) {
        munmap(mapped, file_size); return false;
    }
    ctx.mapped    = mapped;
    ctx.file_size = file_size;

    // Kick off async OS read-ahead for the entire data section
    size_t data_size = file_size - ctx.data_start;
    madvise(const_cast<uint8_t*>(base) + ctx.data_start, data_size, MADV_WILLNEED);

    printf("Loading GGUF global + norm weights...\n");

    // token_embd: [n_vocab, n_embd] — HOST memory for embedding lookup
    // Use load_bf16_raw_into to load directly into tok_embd_host (uint16_t), avoiding a 2GB intermediate copy.
    {
        ctx.load_bf16_raw_into("token_embd.weight", model.tok_embd_host);
        printf("  token_embd: [%d, %d] (host memory, %.1f MB)\n",
               MC::n_vocab, MC::n_embd, model.tok_embd_host.size() * 2.0f / (1024 * 1024));
    }

    // output_norm: [n_embd] — small, upload to device now
    {
        auto data = ctx.load_bf16("output_norm.weight");
        model.output_norm = upload_1d_bf16(device, cq, data.data(), MC::n_embd);
        printf("  output_norm: [%d]\n", MC::n_embd);
    }

    // Per-layer norm weights (small, upload immediately)
    int attn_idx = 0, ssm_idx = 0;
    for (int il = 0; il < MC::n_layers; il++) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "blk.%d.", il);
        auto pname = [&](const char* suffix) { return std::string(prefix) + suffix; };

        if (MC::is_recurrent(il)) {
            auto& lw = model.ssm_layers[ssm_idx];
            auto nd = ctx.load_bf16(pname("attn_norm.weight"));
            lw.attn_norm = upload_1d_bf16(device, cq, nd.data(), MC::n_embd);
            auto pn = ctx.load_bf16(pname("post_attention_norm.weight"));
            lw.post_attn_norm = upload_1d_bf16(device, cq, pn.data(), MC::n_embd);

            // SSM host params (small f32 arrays)
            lw.ssm_a_host    = ctx.load_f32(pname("ssm_a"));
            lw.ssm_dt_bias_host = ctx.load_f32(pname("ssm_dt.bias"));
            lw.ssm_norm_host = ctx.load_f32(pname("ssm_norm.weight"));
            {
                auto raw = ctx.load_f32(pname("ssm_conv1d.weight"));
                int channels = MC::ssm_conv_channels, kernel = MC::ssm_conv_kernel;
                lw.ssm_conv1d_host.resize(raw.size());
                for (int ch = 0; ch < channels; ch++)
                    for (int k = 0; k < kernel; k++)
                        lw.ssm_conv1d_host[k * channels + ch] = raw[ch * kernel + k];
            }
            ssm_idx++;
        } else {
            auto& lw = model.attn_layers[attn_idx];
            auto nd = ctx.load_bf16(pname("attn_norm.weight"));
            lw.attn_norm = upload_1d_bf16(device, cq, nd.data(), MC::n_embd);
            auto qn = ctx.load_bf16(pname("attn_q_norm.weight"));
            lw.attn_q_norm = upload_1d_bf16(device, cq, qn.data(), MC::head_dim);
            auto kn = ctx.load_bf16(pname("attn_k_norm.weight"));
            lw.attn_k_norm = upload_1d_bf16(device, cq, kn.data(), MC::head_dim);
            auto pn = ctx.load_bf16(pname("post_attention_norm.weight"));
            lw.post_attn_norm = upload_1d_bf16(device, cq, pn.data(), MC::n_embd);
            attn_idx++;
        }
    }

    printf("Global + norm weights loaded.\n");
    return true;
}

// ============================================================================
// Legacy all-at-once loader (loads ALL packed tensors into host vectors).
// ============================================================================

bool load_gguf_weights(
    const std::string& path, ModelBuffers& model,
    MeshDevice* device, MeshCommandQueue& cq)
{
    GGUFContext ctx;
    if (!open_gguf(path, model, device, cq, ctx)) return false;

    printf("Loading BFP8_B tiled GGUF weights to device DRAM...\n");

    // output (LM head) packed
    model.output_packed = ctx.load_packed("output.weight");
    printf("  output: [pre-packed BFP8_B, %.1f MB]\n",
           model.output_packed.size() * 4.0f / (1024 * 1024));

    int attn_idx = 0, ssm_idx = 0;
    for (int il = 0; il < MC::n_layers; il++) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "blk.%d.", il);
        auto pname = [&](const char* suffix) { return std::string(prefix) + suffix; };

        if (MC::is_recurrent(il)) {
            auto& lw = model.ssm_layers[ssm_idx];
            auto qkv_p   = ctx.load_packed(pname("attn_qkv.weight"));
            auto gate_p  = ctx.load_packed(pname("attn_gate.weight"));
            auto alpha_p = ctx.load_packed(pname("ssm_alpha.weight"));
            auto beta_p  = ctx.load_packed(pname("ssm_beta.weight"));
            lw.w_combined_packed.insert(lw.w_combined_packed.end(), qkv_p.begin(),   qkv_p.end());
            lw.w_combined_packed.insert(lw.w_combined_packed.end(), gate_p.begin(),  gate_p.end());
            lw.w_combined_packed.insert(lw.w_combined_packed.end(), alpha_p.begin(), alpha_p.end());
            lw.w_combined_packed.insert(lw.w_combined_packed.end(), beta_p.begin(),  beta_p.end());
            lw.ssm_out_packed  = ctx.load_packed(pname("ssm_out.weight"));
            lw.ffn_gate_packed = ctx.load_packed(pname("ffn_gate.weight"));
            lw.ffn_up_packed   = ctx.load_packed(pname("ffn_up.weight"));
            lw.ffn_down_packed = ctx.load_packed(pname("ffn_down.weight"));
            printf("  Layer %d: SSM (delta-net) [%d]\n", il, ssm_idx);
            ssm_idx++;
        } else {
            auto& lw = model.attn_layers[attn_idx];
            auto q_p = ctx.load_packed(pname("attn_q.weight"));
            auto k_p = ctx.load_packed(pname("attn_k.weight"));
            auto v_p = ctx.load_packed(pname("attn_v.weight"));
            lw.wqkv_packed.insert(lw.wqkv_packed.end(), q_p.begin(), q_p.end());
            lw.wqkv_packed.insert(lw.wqkv_packed.end(), k_p.begin(), k_p.end());
            lw.wqkv_packed.insert(lw.wqkv_packed.end(), v_p.begin(), v_p.end());
            lw.wo_packed = ctx.load_packed(pname("attn_output.weight"));
            lw.ffn_gate_packed = ctx.load_packed(pname("ffn_gate.weight"));
            lw.ffn_up_packed   = ctx.load_packed(pname("ffn_up.weight"));
            lw.ffn_down_packed = ctx.load_packed(pname("ffn_down.weight"));
            printf("  Layer %d: Attention [%d]\n", il, attn_idx);
            attn_idx++;
        }
    }

    // ctx goes out of scope here → munmap
    printf("All weights loaded to device DRAM.\n");
    return true;
}
