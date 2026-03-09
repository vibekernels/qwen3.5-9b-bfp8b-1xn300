// SPDX-License-Identifier: Apache-2.0
// Test: load Qwen3.5-9B GGUF weights into device DRAM, verify sizes.

#include "gguf_loader.h"
#include "engine.h"
#include "model_config.h"

#include <cstdio>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/device.hpp>

using namespace tt::tt_metal::distributed;

int main(int argc, char** argv) {
    const char* model_path = nullptr;

    if (argc >= 2) {
        model_path = argv[1];
    } else {
        model_path = "models/Qwen3.5-9B-BF16.gguf";
    }

    try {
        printf("Opening device...\n");
        auto mesh_device = MeshDevice::create_unit_mesh(0);
        auto grid = mesh_device->compute_with_storage_grid_size();
        printf("Device opened. Compute grid: %ux%u\n", grid.x, grid.y);

        MeshCommandQueue& cq = mesh_device->mesh_command_queue();

        ModelBuffers model{};
        printf("Loading weights from %s...\n", model_path);

        if (!load_gguf_weights(model_path, model, mesh_device.get(), cq)) {
            fprintf(stderr, "Failed to load weights\n");
            mesh_device->close();
            return 1;
        }

        // Verify key data was loaded
        int ok = 1;
        if (model.tok_embd_host.empty()) { fprintf(stderr, "tok_embd not loaded\n"); ok = 0; }
        if (model.output_host.empty())   { fprintf(stderr, "output not loaded\n"); ok = 0; }
        if (!model.output_norm)          { fprintf(stderr, "output_norm not loaded\n"); ok = 0; }

        for (int i = 0; i < 8; i++) {
            if (!model.attn_layers[i].wqkv) {
                fprintf(stderr, "attn_layer[%d].wqkv not loaded\n", i); ok = 0;
            }
        }
        for (int i = 0; i < 24; i++) {
            if (!model.ssm_layers[i].w_combined) {
                fprintf(stderr, "ssm_layer[%d].w_combined not loaded\n", i); ok = 0;
            }
        }

        // Wait for all uploads to complete
        Finish(cq);

        printf("%s\n", ok ? "PASSED" : "FAILED");
        mesh_device->close();
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        fprintf(stderr, "Test failed: %s\n", e.what());
        return 1;
    }
}
