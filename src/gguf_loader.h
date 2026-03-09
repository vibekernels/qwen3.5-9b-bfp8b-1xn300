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
// Returns true on success.
bool load_gguf_weights(
    const std::string& path,
    ModelBuffers& model,
    tt::tt_metal::distributed::MeshDevice* device,
    tt::tt_metal::distributed::MeshCommandQueue& cq);
