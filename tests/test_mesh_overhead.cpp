// Test: compare overhead of single device vs 1x2 MeshDevice for EnqueueWrite/Read
#include <cstdio>
#include <chrono>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/device.hpp>

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using Clock = std::chrono::high_resolution_clock;

static void bench_replicated(MeshDevice* mesh, const char* label, int iters) {
    auto& cq = mesh->mesh_command_queue();
    uint32_t tile_bytes = TILE_HEIGHT * TILE_WIDTH * sizeof(bfloat16);
    uint32_t num_tiles = 128;

    auto buf = MeshBuffer::create(
        ReplicatedBufferConfig{.size = num_tiles * tile_bytes},
        DeviceLocalBufferConfig{.page_size = tile_bytes, .buffer_type = BufferType::DRAM},
        mesh);

    size_t elems = num_tiles * TILE_HEIGHT * TILE_WIDTH;
    std::vector<bfloat16> host_data(elems, bfloat16(1.0f));
    std::vector<bfloat16> host_read(elems);

    // Warmup
    for (int i = 0; i < 5; i++) {
        EnqueueWriteMeshBuffer(cq, buf, host_data, false);
        EnqueueReadMeshBuffer(cq, host_read, buf, true);
    }

    // Benchmark write
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) {
        EnqueueWriteMeshBuffer(cq, buf, host_data, false);
    }
    Finish(cq);
    double write_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    // Benchmark read
    auto t1 = Clock::now();
    for (int i = 0; i < iters; i++) {
        EnqueueReadMeshBuffer(cq, host_read, buf, true);
    }
    double read_ms = std::chrono::duration<double, std::milli>(Clock::now() - t1).count();

    printf("[%s replicated] write=%.3f ms/call  read=%.3f ms/call  (%d iters)\n",
           label, write_ms / iters, read_ms / iters, iters);
}

static void bench_sharded(MeshDevice* mesh, const char* label, int iters) {
    auto& cq = mesh->mesh_command_queue();
    uint32_t tile_bytes = TILE_HEIGHT * TILE_WIDTH * sizeof(bfloat16);
    uint32_t num_tiles = 128;
    uint32_t num_devices = mesh->num_devices();

    // Sharded: each device gets num_tiles tiles
    // Global shape: num_devices * num_tiles rows of tiles, 1 col
    auto shard_shape = Shape2D{num_tiles * TILE_HEIGHT, TILE_WIDTH};
    auto global_shape = Shape2D{shard_shape.height() * mesh->num_rows(), shard_shape.width() * mesh->num_cols()};

    auto buf = MeshBuffer::create(
        distributed::ShardedBufferConfig{
            .global_size = num_devices * num_tiles * tile_bytes,
            .global_buffer_shape = global_shape,
            .shard_shape = shard_shape,
            .shard_orientation = ShardOrientation::ROW_MAJOR},
        DeviceLocalBufferConfig{.page_size = tile_bytes, .buffer_type = BufferType::DRAM},
        mesh);

    size_t total_elems = num_devices * num_tiles * TILE_HEIGHT * TILE_WIDTH;
    std::vector<bfloat16> host_data(total_elems, bfloat16(1.0f));
    std::vector<bfloat16> host_read(total_elems);

    // Warmup
    for (int i = 0; i < 5; i++) {
        EnqueueWriteMeshBuffer(cq, buf, host_data, false);
        EnqueueReadMeshBuffer(cq, host_read, buf, true);
    }

    // Benchmark write
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) {
        EnqueueWriteMeshBuffer(cq, buf, host_data, false);
    }
    Finish(cq);
    double write_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    // Benchmark read
    auto t1 = Clock::now();
    for (int i = 0; i < iters; i++) {
        EnqueueReadMeshBuffer(cq, host_read, buf, true);
    }
    double read_ms = std::chrono::duration<double, std::milli>(Clock::now() - t1).count();

    printf("[%s sharded]    write=%.3f ms/call  read=%.3f ms/call  (%d iters)\n",
           label, write_ms / iters, read_ms / iters, iters);
}

int main() {
    int iters = 200;

    printf("=== Test 1: Single device (unit mesh chip 0) ===\n");
    {
        auto mesh = MeshDevice::create_unit_mesh(0);
        bench_replicated(mesh.get(), "1x1", iters);
        mesh->close();
    }

    printf("\n=== Test 2: 1x2 MeshDevice (both chips, sharded buffer) ===\n");
    {
        auto mesh = MeshDevice::create(MeshDeviceConfig(MeshShape(1, 2)));
        bench_sharded(mesh.get(), "1x2", iters);
        mesh->close();
    }

    printf("\n=== Test 3: Two separate unit meshes (chip 0 only) ===\n");
    {
        auto meshes = MeshDevice::create_unit_meshes({0, 1});
        bench_replicated(meshes[0].get(), "unit0", iters);
        meshes[0]->close();
        meshes[1]->close();
    }

    printf("\nDone.\n");
    return 0;
}
