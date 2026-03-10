#pragma once

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>

namespace tt::tt_metal::distributed {
class MeshDevice;
class MeshBuffer;
class MeshCommandQueue;
}

struct ModelBuffers;

// Parse GGUF file and upload all weights to device DRAM as tiled BF16.
// If skip_large_weights=true, skip reading large matmul BF16 matrices
// (used when BFP8_B cache is available — only small norm/SSM params loaded).
// Returns true on success.
bool load_gguf_weights(
    const std::string& path,
    ModelBuffers& model,
    tt::tt_metal::distributed::MeshDevice* device,
    tt::tt_metal::distributed::MeshCommandQueue& cq,
    bool skip_large_weights = false);
