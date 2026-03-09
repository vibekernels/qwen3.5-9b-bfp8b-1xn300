// SPDX-License-Identifier: Apache-2.0
// Minimal test: open device, verify grid size, run eltwise exp, close.
// Closely follows the eltwise_sfpu programming example.

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/core_coord.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

int main() {
    try {
        auto mesh_device = MeshDevice::create_unit_mesh(0);
        auto grid = mesh_device->compute_with_storage_grid_size();
        printf("Device opened. Compute grid: %ux%u (%u cores)\n",
               grid.x, grid.y, grid.x * grid.y);

        MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        MeshCoordinateRange device_range(mesh_device->shape());

        constexpr uint32_t n_tiles = 4;
        constexpr uint32_t elements_per_tile = constants::TILE_WIDTH * constants::TILE_HEIGHT;
        constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;

        // Allocate DRAM buffers
        DeviceLocalBufferConfig dram_config{
            .page_size = tile_size_bytes, .buffer_type = BufferType::DRAM};
        ReplicatedBufferConfig buffer_config{.size = tile_size_bytes * n_tiles};

        auto src_buf = MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto dst_buf = MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

        // Circular buffers
        constexpr tt::tt_metal::CoreCoord core = {0, 0};
        constexpr uint32_t cb_in_index = tt::CBIndex::c_0;
        constexpr uint32_t cb_out_index = tt::CBIndex::c_16;

        CircularBufferConfig cb_in_config =
            CircularBufferConfig(2 * tile_size_bytes, {{cb_in_index, tt::DataFormat::Float16_b}})
                .set_page_size(cb_in_index, tile_size_bytes);
        CircularBufferConfig cb_out_config =
            CircularBufferConfig(2 * tile_size_bytes, {{cb_out_index, tt::DataFormat::Float16_b}})
                .set_page_size(cb_out_index, tile_size_bytes);

        Program program = CreateProgram();
        CreateCircularBuffer(program, core, cb_in_config);
        CreateCircularBuffer(program, core, cb_out_config);

        // Reader kernel
        std::vector<uint32_t> reader_ct_args;
        TensorAccessorArgs(*src_buf).append_to(reader_ct_args);
        auto reader_id = CreateKernel(program,
            "tt_metal/programming_examples/eltwise_sfpu/kernels/dataflow/read_tile.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = reader_ct_args});

        // Writer kernel
        std::vector<uint32_t> writer_ct_args;
        TensorAccessorArgs(*dst_buf).append_to(writer_ct_args);
        auto writer_id = CreateKernel(program,
            "tt_metal/programming_examples/eltwise_sfpu/kernels/dataflow/write_tile.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = writer_ct_args});

        // Compute kernel: exp (from eltwise_sfpu example)
        auto compute_id = CreateKernel(program,
            "tt_metal/programming_examples/eltwise_sfpu/kernels/compute/eltwise_sfpu.cpp",
            core,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .math_approx_mode = false});

        // Input data
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        std::vector<bfloat16> input(n_tiles * elements_per_tile);
        for (auto& v : input) v = bfloat16(dist(rng));

        EnqueueWriteMeshBuffer(cq, src_buf, input, false);

        // Runtime args
        SetRuntimeArgs(program, compute_id, core, {n_tiles});
        SetRuntimeArgs(program, reader_id, core, {src_buf->address(), n_tiles});
        SetRuntimeArgs(program, writer_id, core, {dst_buf->address(), n_tiles});

        // Execute
        MeshWorkload workload;
        workload.add_program(device_range, std::move(program));
        EnqueueMeshWorkload(cq, workload, false);
        Finish(cq);

        // Read result
        std::vector<bfloat16> result;
        EnqueueReadMeshBuffer(cq, result, dst_buf, true);

        // Verify: exp(x) on each element
        int errors = 0;
        float max_diff = 0;
        for (uint32_t i = 0; i < result.size(); i++) {
            float expected = std::exp(static_cast<float>(input[i]));
            float actual = static_cast<float>(result[i]);
            float diff = std::abs(expected - actual);
            // bf16 exp tolerance
            float ref_expected = static_cast<float>(bfloat16(expected));
            float ref_diff = std::abs(ref_expected - actual);
            if (ref_diff > max_diff) max_diff = ref_diff;
            if (ref_diff > 0.05f) errors++;
        }

        printf("Exp test: %zu elements, max_diff=%.6f, errors=%d\n",
               result.size(), max_diff, errors);
        printf("%s\n", errors == 0 ? "PASSED" : "FAILED");

        mesh_device->close();
        return errors == 0 ? 0 : 1;

    } catch (const std::exception& e) {
        fprintf(stderr, "Test failed: %s\n", e.what());
        return 1;
    }
}
