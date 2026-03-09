// SPDX-License-Identifier: Apache-2.0
// Test: single-core matmul using the built-in matmul_single_core example kernels.
// Validates C = A * B where A=[M,K], B=[K,N].
// Uses the exact same kernel files and API as the programming example.

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/core_coord.hpp>

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

int main() {
    try {
        auto mesh_device = MeshDevice::create_unit_mesh(0);
        MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        auto device_range = MeshCoordinateRange(mesh_device->shape());

        // C = A * B, where A=[M,K], B=[K,N], C=[M,N]
        // Use small sizes: M=32 (1 tile), K=64 (2 tiles), N=32 (1 tile)
        constexpr uint32_t M = 32;
        constexpr uint32_t K = 64;
        constexpr uint32_t N = 32;
        constexpr uint32_t Mt = M / TILE_HEIGHT;  // 1
        constexpr uint32_t Kt = K / TILE_WIDTH;   // 2
        constexpr uint32_t Nt = N / TILE_WIDTH;   // 1

        // Generate random data
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

        std::vector<bfloat16> A_vec(M * K);
        std::vector<bfloat16> B_vec(K * N);
        for (auto& v : A_vec) v = bfloat16(dist(rng));
        for (auto& v : B_vec) v = bfloat16(dist(rng));

        // CPU golden: C = A * B
        std::vector<float> C_golden(M * N, 0.0f);
        for (uint32_t i = 0; i < M; i++) {
            for (uint32_t j = 0; j < N; j++) {
                float sum = 0;
                for (uint32_t k = 0; k < K; k++) {
                    sum += static_cast<float>(A_vec[i * K + k]) * static_cast<float>(B_vec[k * N + j]);
                }
                C_golden[i * N + j] = sum;
            }
        }

        // Tilize
        auto A_tiled = tilize_nfaces(A_vec, M, K);
        auto B_tiled = tilize_nfaces(B_vec, K, N);

        // DRAM buffers
        uint32_t tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
        DeviceLocalBufferConfig dram_config{.page_size = tile_size, .buffer_type = BufferType::DRAM};

        uint32_t A_num_tiles = Mt * Kt;
        uint32_t B_num_tiles = Kt * Nt;
        uint32_t C_num_tiles = Mt * Nt;

        auto a_buf = MeshBuffer::create(ReplicatedBufferConfig{.size = A_num_tiles * tile_size}, dram_config, mesh_device.get());
        auto b_buf = MeshBuffer::create(ReplicatedBufferConfig{.size = B_num_tiles * tile_size}, dram_config, mesh_device.get());
        auto c_buf = MeshBuffer::create(ReplicatedBufferConfig{.size = C_num_tiles * tile_size}, dram_config, mesh_device.get());

        EnqueueWriteMeshBuffer(cq, a_buf, A_tiled, false);
        EnqueueWriteMeshBuffer(cq, b_buf, B_tiled, false);

        // Program
        Program program = CreateProgram();
        tt::tt_metal::CoreCoord core = {0, 0};
        tt::DataFormat df = tt::DataFormat::Float16_b;

        // Circular buffers: c_0 for A, c_1 for B, c_16 for output
        CreateCircularBuffer(program, core,
            CircularBufferConfig(2 * tile_size, {{CBIndex::c_0, df}}).set_page_size(CBIndex::c_0, tile_size));
        CreateCircularBuffer(program, core,
            CircularBufferConfig(2 * tile_size, {{CBIndex::c_1, df}}).set_page_size(CBIndex::c_1, tile_size));
        CreateCircularBuffer(program, core,
            CircularBufferConfig(2 * tile_size, {{CBIndex::c_16, df}}).set_page_size(CBIndex::c_16, tile_size));

        // Reader: compile-time args = TensorAccessorArgs for A and B
        std::vector<uint32_t> reader_ct_args;
        TensorAccessorArgs(*a_buf).append_to(reader_ct_args);
        TensorAccessorArgs(*b_buf).append_to(reader_ct_args);

        auto reader_id = CreateKernel(program,
            "tt_metal/programming_examples/matmul/matmul_single_core/kernels/dataflow/reader_single_core_mm.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = reader_ct_args});

        // Writer: compile-time args = TensorAccessorArgs for C
        std::vector<uint32_t> writer_ct_args;
        TensorAccessorArgs(*c_buf).append_to(writer_ct_args);

        auto writer_id = CreateKernel(program,
            "tt_metal/programming_examples/matmul/matmul_single_core/kernels/dataflow/writer_single_core_mm.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = writer_ct_args});

        // Compute: compile-time args = {Mt, Kt, Nt}
        CreateKernel(program,
            "tt_metal/programming_examples/matmul/matmul_single_core/kernels/compute/mm.cpp",
            core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = {Mt, Kt, Nt}});

        // Runtime args: reader gets {src0_addr, src1_addr, Mt, Kt, Nt}
        //               writer gets {dst_addr, Mt, Nt}
        SetRuntimeArgs(program, reader_id, core,
            {(uint32_t)a_buf->address(), (uint32_t)b_buf->address(), Mt, Kt, Nt});
        SetRuntimeArgs(program, writer_id, core,
            {(uint32_t)c_buf->address(), Mt, Nt});

        // Execute
        MeshWorkload workload;
        workload.add_program(device_range, std::move(program));
        EnqueueMeshWorkload(cq, workload, false);
        Finish(cq);

        // Read result
        std::vector<bfloat16> result_tiled(M * N, bfloat16(0.0f));
        EnqueueReadMeshBuffer(cq, result_tiled, c_buf, true);
        auto result = untilize_nfaces(result_tiled, M, N);

        // Compare
        float max_diff = 0;
        float sum_sq_gold = 0, sum_sq_diff = 0;
        for (uint32_t i = 0; i < M * N; i++) {
            float expected = C_golden[i];
            float actual = static_cast<float>(result[i]);
            float diff = std::abs(expected - actual);
            if (diff > max_diff) max_diff = diff;
            sum_sq_gold += expected * expected;
            sum_sq_diff += (expected - actual) * (expected - actual);
        }
        float pcc = 1.0f - sum_sq_diff / (sum_sq_gold + 1e-10f);

        printf("Matmul test: [%u,%u] x [%u,%u] = [%u,%u]\n", M, K, K, N, M, N);
        printf("max_diff=%.6f, PCC=%.6f\n", max_diff, pcc);
        printf("%s\n", pcc > 0.95f ? "PASSED" : "FAILED");

        mesh_device->close();
        return pcc > 0.95f ? 0 : 1;

    } catch (const std::exception& e) {
        fprintf(stderr, "Test failed: %s\n", e.what());
        return 1;
    }
}
