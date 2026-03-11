#pragma once

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstddef>
#include <tt-metalium/bfloat16.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
class MeshBuffer;
class MeshCommandQueue;
}

struct ModelBuffers;

// ============================================================================
// Streaming GGUF context: keeps the file mmap'd so tensors can be loaded
// on-demand without holding all ~9 GB in host RAM simultaneously.
// ============================================================================
struct GGUFTensorInfo {
    std::string name;
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;
    uint64_t offset;
    uint64_t size_bytes;
};

struct GGUFContext {
    void*  mapped    = nullptr;   // mmap base
    size_t file_size = 0;
    size_t data_start = 0;        // byte offset to tensor data section
    std::unordered_map<std::string, GGUFTensorInfo> tmap;

    bool is_open() const { return mapped != nullptr; }

    // Load a tensor by name into a host vector (allocates a new vector each call).
    std::vector<uint32_t>                    load_packed(const std::string& name) const;
    std::vector<bfloat16>                    load_bf16  (const std::string& name) const;
    std::vector<float>                       load_f32   (const std::string& name) const;

    // Load into a pre-allocated vector (resize without realloc if big enough),
    // avoiding page-fault overhead from repeated large malloc/free cycles.
    void load_packed_into  (const std::string& name, std::vector<uint32_t>& dst) const;
    void load_bf16_into    (const std::string& name, std::vector<bfloat16>& dst) const;
    // Load BF16 tensor as raw uint16 (avoids class overhead and extra copy).
    void load_bf16_raw_into(const std::string& name, std::vector<uint16_t>& dst) const;

    bool has(const std::string& name) const;

    void close();
    ~GGUFContext() { close(); }
    GGUFContext() = default;
    // Non-copyable
    GGUFContext(const GGUFContext&) = delete;
    GGUFContext& operator=(const GGUFContext&) = delete;
};

// Open the GGUF file, parse the header, and mmap the data section.
// Also loads global weights (token_embd, output_norm, output.weight) and all
// per-layer norm buffers — small tensors only, without allocating the large
// packed weight vectors. Returns true on success.
bool open_gguf(
    const std::string& path,
    ModelBuffers& model,
    tt::tt_metal::distributed::MeshDevice* device,
    tt::tt_metal::distributed::MeshCommandQueue& cq,
    GGUFContext& ctx);

// Legacy all-at-once loader (loads ALL tensors into host vectors at once).
// Kept for compatibility; prefer the streaming open_gguf + per-layer loads.
bool load_gguf_weights(
    const std::string& path,
    ModelBuffers& model,
    tt::tt_metal::distributed::MeshDevice* device,
    tt::tt_metal::distributed::MeshCommandQueue& cq);
