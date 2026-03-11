// SPDX-License-Identifier: Apache-2.0
//
// make_bfp8_gguf — convert a BF16 GGUF to Tenstorrent BFP8_B tiled GGUF.
//
// Large 2D weight matrices (tile-aligned) are converted to Tenstorrent's
// BFP8_B block-float format (32×32 tiles, 1 shared exponent per 16 values,
// 1088 bytes/tile). All other tensors (norms, embeddings, SSM scalars) are
// copied verbatim. All GGUF metadata (tokenizer, model config, etc.) is
// preserved exactly.
//
// The output uses custom GGML type 200 (GGML_TYPE_TT_BFP8B_TILED) for
// converted tensors. The engine's gguf_loader needs to handle this type.
//
// Usage:
//   make_bfp8_gguf <input.gguf> <output.gguf>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <chrono>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <blockfloat_common.hpp>
#include <tt_stl/span.hpp>

using namespace tt::constants;
using ::bfloat16;

// Custom GGML type for Tenstorrent BFP8_B tiled format (32×32 tiles).
// Must match the value used in gguf_loader.cpp when reading.
static constexpr uint32_t GGML_TYPE_TT_BFP8B_TILED = 200;

// Standard GGML scalar types
static constexpr uint32_t GGML_TYPE_F32  = 0;
static constexpr uint32_t GGML_TYPE_F16  = 1;
static constexpr uint32_t GGML_TYPE_BF16 = 30;

// GGUF value types
enum GGUFType : uint32_t {
    GGUF_TYPE_UINT8   = 0,  GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,  GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,  GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,  GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,  GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10, GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

// ============================================================================
// GGUF helpers
// ============================================================================

static std::string read_str(FILE* f) {
    uint64_t len;
    fread(&len, 8, 1, f);
    std::string s(len, '\0');
    fread(&s[0], 1, len, f);
    return s;
}

static void write_str(FILE* f, const std::string& s) {
    uint64_t len = s.size();
    fwrite(&len, 8, 1, f);
    fwrite(s.data(), 1, len, f);
}

// Advance file past one GGUF value of the given type
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
        case GGUF_TYPE_STRING: {
            uint64_t len; fread(&len, 8, 1, f);
            fseek(f, (long)len, SEEK_CUR);
            break;
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t arr_type; uint64_t arr_len;
            fread(&arr_type, 4, 1, f);
            fread(&arr_len, 8, 1, f);
            for (uint64_t i = 0; i < arr_len; i++) skip_value(f, arr_type);
            break;
        }
        default:
            fprintf(stderr, "Unknown GGUF value type: %u\n", vtype);
            exit(1);
    }
}

static size_t elem_size(uint32_t ggml_type) {
    switch (ggml_type) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:  return 2;
        case GGML_TYPE_BF16: return 2;
        default: return 0;
    }
}

// ============================================================================
// Tensor descriptors
// ============================================================================

struct TensorDesc {
    std::string name;
    uint32_t n_dims;
    uint64_t dims[4];     // GGUF order: dims[0]=innermost (K), dims[1]=M for 2D
    uint32_t in_type;     // original GGML type
    uint64_t in_offset;   // byte offset within input data section
    uint64_t in_bytes;    // raw byte size in input file

    // Output plan
    uint32_t out_type;    // GGML_TYPE_TT_BFP8B_TILED or same as in_type
    uint64_t out_offset;  // byte offset within output data section
    uint64_t out_bytes;   // byte size in output file
};

// BFP8_B packed size for an M×K matrix (tile-aligned)
static uint64_t bfp8_packed_size(uint32_t M, uint32_t K) {
    return (uint64_t)(M / TILE_HEIGHT) * (K / TILE_WIDTH) * BFLOAT8_B_TILE_HW;
}

// Decide whether to convert this tensor to BFP8_B
static bool should_convert(const TensorDesc& t) {
    // Only 2D matrices
    if (t.n_dims != 2) return false;
    // Only BF16 and F32 (source types we can reinterpret)
    if (t.in_type != GGML_TYPE_BF16 && t.in_type != GGML_TYPE_F32) return false;
    // Both dims must be tile-aligned (multiples of 32)
    if (t.dims[0] % TILE_WIDTH != 0 || t.dims[1] % TILE_HEIGHT != 0) return false;
    // Minimum one tile in each dim
    if (t.dims[0] < TILE_WIDTH || t.dims[1] < TILE_HEIGHT) return false;
    // Keep token embeddings as BF16 — used for fast CPU lookup
    if (t.name == "token_embd.weight") return false;
    return true;
}

// ============================================================================
// BFP8_B packing
// ============================================================================

// Pack a row-major BF16 matrix [M, K] to BFP8_B tiled format.
// M = dims[1], K = dims[0] for a 2D GGUF tensor.
static std::vector<uint32_t> pack_bf16_as_bfp8b(
    const uint16_t* src, uint32_t M, uint32_t K)
{
    const uint32_t TH = TILE_HEIGHT, TW = TILE_WIDTH;
    uint32_t Mt = M / TH, Kt = K / TW;
    uint32_t tile_elems = TH * TW;

    // Rearrange from row-major [M,K] to tile-major: each 32×32 block contiguous
    std::vector<bfloat16> tiled(Mt * Kt * tile_elems);
    const bfloat16* s = reinterpret_cast<const bfloat16*>(src);
    for (uint32_t mt = 0; mt < Mt; mt++) {
        for (uint32_t kt = 0; kt < Kt; kt++) {
            bfloat16* dst = tiled.data() + (mt * Kt + kt) * tile_elems;
            for (uint32_t r = 0; r < TH; r++)
                memcpy(dst + r * TW, s + (mt * TH + r) * K + kt * TW,
                       TW * sizeof(bfloat16));
        }
    }

    ttsl::Span<const bfloat16> span(tiled.data(), tiled.size());
    return pack_as_bfp_tiles<tt::DataFormat::Bfp8_b>(
        span, /*row_major_input=*/true, /*is_exp_a=*/false);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.gguf> <output.gguf>\n", argv[0]);
        fprintf(stderr, "\n");
        fprintf(stderr, "Converts large 2D weight matrices to Tenstorrent BFP8_B\n");
        fprintf(stderr, "tiled format (GGML type %u). All metadata and small\n", GGML_TYPE_TT_BFP8B_TILED);
        fprintf(stderr, "tensors (norms, embeddings, SSM scalars) are copied verbatim.\n");
        return 1;
    }

    const char* in_path  = argv[1];
    const char* out_path = argv[2];

    // =========================================================================
    // Step 1: Read input header, KV blob, tensor info
    // =========================================================================

    FILE* fin = fopen(in_path, "rb");
    if (!fin) { fprintf(stderr, "Cannot open %s: %m\n", in_path); return 1; }

    char magic[4];
    fread(magic, 1, 4, fin);
    if (memcmp(magic, "GGUF", 4) != 0) {
        fprintf(stderr, "Not a GGUF file: %s\n", in_path);
        return 1;
    }
    uint32_t version;
    uint64_t n_tensors, n_kv;
    fread(&version, 4, 1, fin);
    fread(&n_tensors, 8, 1, fin);
    fread(&n_kv, 8, 1, fin);
    printf("Input: GGUF v%u, %lu tensors, %lu KV pairs\n", version, n_tensors, n_kv);

    // Capture the entire KV metadata section as a verbatim blob
    long kv_blob_start = ftell(fin);
    for (uint64_t i = 0; i < n_kv; i++) {
        read_str(fin);                         // key
        uint32_t vtype; fread(&vtype, 4, 1, fin);
        skip_value(fin, vtype);
    }
    long kv_blob_end = ftell(fin);

    fseek(fin, kv_blob_start, SEEK_SET);
    std::vector<uint8_t> kv_blob((size_t)(kv_blob_end - kv_blob_start));
    fread(kv_blob.data(), 1, kv_blob.size(), fin);

    // Read tensor info
    std::vector<TensorDesc> tensors(n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        auto& t = tensors[i];
        t.name   = read_str(fin);
        fread(&t.n_dims, 4, 1, fin);
        memset(t.dims, 0, sizeof(t.dims));
        for (uint32_t d = 0; d < t.n_dims; d++)
            fread(&t.dims[d], 8, 1, fin);
        fread(&t.in_type, 4, 1, fin);
        fread(&t.in_offset, 8, 1, fin);

        uint64_t n_elems = 1;
        for (uint32_t d = 0; d < t.n_dims; d++) n_elems *= t.dims[d];
        size_t esz = elem_size(t.in_type);
        if (esz == 0) {
            fprintf(stderr, "Unsupported GGML type %u in tensor '%s'\n",
                    t.in_type, t.name.c_str());
            return 1;
        }
        t.in_bytes = n_elems * esz;
    }

    long header_end  = ftell(fin);
    long data_start  = ((header_end + 31) / 32) * 32; // 32-byte aligned

    // =========================================================================
    // Step 2: Plan output (types, sizes, offsets)
    // =========================================================================

    int n_converted = 0;
    uint64_t out_offset = 0;

    for (auto& t : tensors) {
        if (should_convert(t)) {
            // dims[0] = K (inner), dims[1] = M (outer) in GGUF ordering
            uint32_t M = (uint32_t)t.dims[1];
            uint32_t K = (uint32_t)t.dims[0];
            t.out_type  = GGML_TYPE_TT_BFP8B_TILED;
            t.out_bytes = bfp8_packed_size(M, K);
            n_converted++;
        } else {
            t.out_type  = t.in_type;
            t.out_bytes = t.in_bytes;
        }
        t.out_offset = out_offset;
        out_offset  += t.out_bytes;
        out_offset   = ((out_offset + 31) / 32) * 32; // pad between tensors
    }

    printf("Converting %d/%lu tensors to BFP8_B tiled (type %u)\n",
           n_converted, n_tensors, GGML_TYPE_TT_BFP8B_TILED);

    // =========================================================================
    // Step 3: Write output GGUF header, KV blob, and tensor info
    // =========================================================================

    FILE* fout = fopen(out_path, "wb");
    if (!fout) { fprintf(stderr, "Cannot create %s: %m\n", out_path); return 1; }

    // Header
    fwrite("GGUF", 1, 4, fout);
    fwrite(&version,   4, 1, fout);
    fwrite(&n_tensors, 8, 1, fout);
    fwrite(&n_kv,      8, 1, fout);

    // KV metadata (verbatim)
    fwrite(kv_blob.data(), 1, kv_blob.size(), fout);

    // Tensor info
    for (const auto& t : tensors) {
        write_str(fout, t.name);
        fwrite(&t.n_dims, 4, 1, fout);
        for (uint32_t d = 0; d < t.n_dims; d++)
            fwrite(&t.dims[d], 8, 1, fout);
        fwrite(&t.out_type,   4, 1, fout);
        fwrite(&t.out_offset, 8, 1, fout);
    }

    // Pad to 32-byte alignment for the data section
    long cur = ftell(fout);
    long aligned = ((cur + 31) / 32) * 32;
    if (aligned > cur) {
        std::vector<uint8_t> pad(aligned - cur, 0);
        fwrite(pad.data(), 1, pad.size(), fout);
    }
    long out_data_start = ftell(fout);

    // =========================================================================
    // Step 4: Convert and write tensor data
    // =========================================================================

    auto t_start = std::chrono::steady_clock::now();
    uint64_t bytes_written = 0;

    for (uint64_t i = 0; i < n_tensors; i++) {
        const auto& t = tensors[i];

        // Seek to the correct output position
        if (fseek(fout, out_data_start + (long)t.out_offset, SEEK_SET) != 0) {
            fprintf(stderr, "fseek failed for tensor '%s'\n", t.name.c_str());
            return 1;
        }

        if (t.out_type == t.in_type) {
            // Copy verbatim
            fseek(fin, data_start + (long)t.in_offset, SEEK_SET);
            std::vector<uint8_t> buf(t.in_bytes);
            fread(buf.data(), 1, t.in_bytes, fin);
            fwrite(buf.data(), 1, t.in_bytes, fout);
            printf("  [%2lu/%2lu] %-55s copy    %.1f MB\n",
                   i + 1, n_tensors, t.name.c_str(),
                   t.in_bytes / 1e6);
        } else {
            // Read source as BF16, pack to BFP8_B
            uint32_t M = (uint32_t)t.dims[1];
            uint32_t K = (uint32_t)t.dims[0];
            uint64_t n_elems = (uint64_t)M * K;

            fseek(fin, data_start + (long)t.in_offset, SEEK_SET);
            std::vector<uint16_t> bf16(n_elems);

            if (t.in_type == GGML_TYPE_BF16) {
                fread(bf16.data(), 2, n_elems, fin);
            } else {
                // F32 → BF16: truncate lower 16 bits
                std::vector<uint32_t> f32(n_elems);
                fread(f32.data(), 4, n_elems, fin);
                for (uint64_t j = 0; j < n_elems; j++)
                    bf16[j] = (uint16_t)(f32[j] >> 16);
            }

            auto t0 = std::chrono::steady_clock::now();
            auto packed = pack_bf16_as_bfp8b(bf16.data(), M, K);
            double pack_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();

            fwrite(packed.data(), sizeof(uint32_t), packed.size(), fout);

            printf("  [%2lu/%2lu] %-55s BFP8_B  %5ux%-5u  %.1f→%.1f MB  %.0f ms\n",
                   i + 1, n_tensors, t.name.c_str(),
                   M, K,
                   t.in_bytes / 1e6, t.out_bytes / 1e6,
                   pack_ms);
        }

        bytes_written += t.out_bytes;
        fflush(fout);
    }

    fclose(fin);
    fclose(fout);

    double total_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();

    printf("\nDone in %.1fs. Output: %s (%.1f GB)\n",
           total_s, out_path, bytes_written / 1e9);
    return 0;
}
