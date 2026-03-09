// Test raw DRAM bandwidth: 12 cores reading from 12 banks (DRAM-sharded)
// Uses custom reader kernel with TRID pipelining to measure achievable bandwidth.
#include <cstdio>
#include <chrono>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;
using Clock = std::chrono::high_resolution_clock;

static std::string kernel_path(const char* name) {
    return std::string("/home/ubuntu/qwen3.5-9b-bf16-1xn300d/tt_metal/tests/kernels/") + name;
}

int main() {
    auto meshes = MeshDevice::create_unit_meshes({0});
    auto mesh = meshes[0];
    auto& cq = mesh->mesh_command_queue();

    uint32_t num_banks = mesh->num_dram_channels();
    auto dram_workers = mesh->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_0);
    printf("DRAM banks: %u, cores: %zu\n", num_banks, dram_workers.size());

    // BFP8_B tile size (matches GEMV weight format)
    constexpr uint32_t tile_bytes = 1088;  // BFP8_B

    // Test different total sizes and block sizes
    uint32_t test_sizes_mb[] = {50, 100, 200, 500};
    uint32_t block_tiles[] = {16, 64, 128, 256};

    for (uint32_t size_mb : test_sizes_mb) {
        uint32_t total_tiles = (size_mb * 1000000ULL + tile_bytes - 1) / tile_bytes;
        // Round to multiple of num_banks
        uint32_t tiles_per_bank = (total_tiles + num_banks - 1) / num_banks;
        total_tiles = tiles_per_bank * num_banks;
        uint64_t total_bytes = (uint64_t)total_tiles * tile_bytes;
        uint32_t bytes_per_bank = tiles_per_bank * tile_bytes;

        printf("\n=== %u MB (%u tiles, %u tiles/bank, %u KB/bank) ===\n",
               size_mb, total_tiles, tiles_per_bank, bytes_per_bank / 1024);

        // Create DRAM-sharded buffer
        CoreRange dram_bank_range({0, 0}, {num_banks - 1, 0});
        auto dram_cores = corerange_to_cores(CoreRangeSet(dram_bank_range));
        BufferDistributionSpec shard_spec(
            tt::tt_metal::Shape({1, total_tiles}),
            tt::tt_metal::Shape({1, tiles_per_bank}),
            dram_cores);
        DeviceLocalBufferConfig dram_cfg{
            .page_size = tile_bytes,
            .buffer_type = BufferType::DRAM,
            .sharding_args = BufferShardingArgs(shard_spec)};
        Shape2D global_shape = {1, total_tiles};
        distributed::ShardedBufferConfig sharded_cfg{
            .global_size = (uint64_t)total_tiles * tile_bytes,
            .global_buffer_shape = global_shape,
            .shard_shape = global_shape,
            .shard_orientation = ShardOrientation::ROW_MAJOR,
        };
        auto buf = MeshBuffer::create(sharded_cfg, dram_cfg, mesh.get());

        // Fill with data
        std::vector<uint32_t> host_data(total_tiles * (tile_bytes / 4), 0x3F803F80);
        EnqueueWriteMeshBuffer(cq, buf, host_data, false);
        Finish(cq);

        for (uint32_t blk_tiles : block_tiles) {
            if (blk_tiles > tiles_per_bank) continue;

            // Ensure tiles_per_bank is a multiple of blk_tiles
            uint32_t adjusted_tiles = (tiles_per_bank / blk_tiles) * blk_tiles;
            if (adjusted_tiles == 0) continue;
            uint32_t adj_bytes = adjusted_tiles * tile_bytes;
            uint32_t blk_bytes = blk_tiles * tile_bytes;
            uint32_t cb_tiles = blk_tiles * 2;  // double-buffered

            // Build program
            MeshCoordinateRange dev_range(mesh->shape());
            Program program = CreateProgram();

            // Build non-rectangular CoreRangeSet
            std::vector<CoreRange> core_ranges;
            for (uint32_t b = 0; b < num_banks; b++) {
                core_ranges.push_back(CoreRange(dram_workers[b], dram_workers[b]));
            }
            CoreRangeSet all_cores(core_ranges);

            // CB for weight data
            CircularBufferConfig cb_cfg =
                CircularBufferConfig(cb_tiles * tile_bytes, {{CBIndex::c_0, tt::DataFormat::Bfp8_b}})
                    .set_page_size(CBIndex::c_0, tile_bytes);
            CreateCircularBuffer(program, all_cores, cb_cfg);

            // Reader kernel
            std::vector<uint32_t> ct_args = {adj_bytes, blk_bytes};
            auto reader_kid = CreateKernel(program,
                kernel_path("reader_dram_bw.cpp"), all_cores,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                   .noc = NOC::RISCV_1_default,
                                   .compile_args = ct_args});

            for (uint32_t b = 0; b < num_banks; b++) {
                SetRuntimeArgs(program, reader_kid, dram_workers[b],
                               {b, (uint32_t)buf->address()});
            }

            MeshWorkload workload;
            workload.add_program(dev_range, std::move(program));

            // Warmup
            EnqueueMeshWorkload(cq, workload, false);
            Finish(cq);

            // Time multiple iterations
            int iters = 20;
            auto t0 = Clock::now();
            for (int i = 0; i < iters; i++) {
                EnqueueMeshWorkload(cq, workload, false);
            }
            Finish(cq);
            double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

            double per_iter_ms = ms / iters;
            double total_read = (double)adj_bytes * num_banks * iters;
            double gb_s = total_read / ms / 1e6;
            printf("  BLOCK=%3u tiles: %.3f ms/iter, %.1f GB/s aggregate (%u KB/bank read)\n",
                   blk_tiles, per_iter_ms, gb_s, adj_bytes / 1024);
        }
    }

    // ============================================================
    // Test 2: Reader + compute drain (CB sync overhead, no matmul)
    // ============================================================
    printf("\n=== Test 2: Reader + compute drain (CB sync overhead) ===\n");
    {
        uint32_t size_mb = 200;
        uint32_t total_tiles = (size_mb * 1000000ULL + tile_bytes - 1) / tile_bytes;
        uint32_t tiles_per_bank = (total_tiles + num_banks - 1) / num_banks;
        total_tiles = tiles_per_bank * num_banks;
        uint32_t bytes_per_bank = tiles_per_bank * tile_bytes;

        // Reuse the 200MB sharded buffer from above
        CoreRange dram_bank_range2({0, 0}, {num_banks - 1, 0});
        auto dram_cores2 = corerange_to_cores(CoreRangeSet(dram_bank_range2));
        BufferDistributionSpec shard_spec2(
            tt::tt_metal::Shape({1, total_tiles}),
            tt::tt_metal::Shape({1, tiles_per_bank}),
            dram_cores2);
        DeviceLocalBufferConfig dram_cfg2{
            .page_size = tile_bytes,
            .buffer_type = BufferType::DRAM,
            .sharding_args = BufferShardingArgs(shard_spec2)};
        Shape2D global_shape2 = {1, total_tiles};
        distributed::ShardedBufferConfig sharded_cfg2{
            .global_size = (uint64_t)total_tiles * tile_bytes,
            .global_buffer_shape = global_shape2,
            .shard_shape = global_shape2,
            .shard_orientation = ShardOrientation::ROW_MAJOR,
        };
        auto buf2 = MeshBuffer::create(sharded_cfg2, dram_cfg2, mesh.get());
        std::vector<uint32_t> host_data2(total_tiles * (tile_bytes / 4), 0x3F803F80);
        EnqueueWriteMeshBuffer(cq, buf2, host_data2, false);
        Finish(cq);

        uint32_t blk_tiles_list2[] = {1, 4, 16, 64};
        for (uint32_t blk_tiles : blk_tiles_list2) {
            uint32_t adjusted_tiles = (tiles_per_bank / blk_tiles) * blk_tiles;
            if (adjusted_tiles == 0) continue;
            uint32_t adj_bytes = adjusted_tiles * tile_bytes;
            uint32_t blk_bytes = blk_tiles * tile_bytes;
            uint32_t cb_tiles = blk_tiles * 2;

            MeshCoordinateRange dev_range2(mesh->shape());
            Program program2 = CreateProgram();

            std::vector<CoreRange> core_ranges2;
            for (uint32_t b = 0; b < num_banks; b++) {
                core_ranges2.push_back(CoreRange(dram_workers[b], dram_workers[b]));
            }
            CoreRangeSet all_cores2(core_ranges2);

            CircularBufferConfig cb_cfg2 =
                CircularBufferConfig(cb_tiles * tile_bytes, {{CBIndex::c_0, tt::DataFormat::Bfp8_b}})
                    .set_page_size(CBIndex::c_0, tile_bytes);
            CreateCircularBuffer(program2, all_cores2, cb_cfg2);

            std::vector<uint32_t> ct_args2 = {adj_bytes, blk_bytes};
            auto reader_kid2 = CreateKernel(program2,
                kernel_path("reader_dram_bw.cpp"), all_cores2,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                   .noc = NOC::RISCV_1_default,
                                   .compile_args = ct_args2});

            // Compute kernel that drains tiles one at a time
            std::vector<uint32_t> compute_ct_args = {adjusted_tiles};
            CreateKernel(program2,
                kernel_path("compute_drain.cpp"), all_cores2,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_ct_args});

            for (uint32_t b = 0; b < num_banks; b++) {
                SetRuntimeArgs(program2, reader_kid2, dram_workers[b],
                               {b, (uint32_t)buf2->address()});
            }

            MeshWorkload workload2;
            workload2.add_program(dev_range2, std::move(program2));

            EnqueueMeshWorkload(cq, workload2, false);
            Finish(cq);

            int iters = 20;
            auto t0 = Clock::now();
            for (int i = 0; i < iters; i++) {
                EnqueueMeshWorkload(cq, workload2, false);
            }
            Finish(cq);
            double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

            double per_iter_ms = ms / iters;
            double total_read = (double)adj_bytes * num_banks * iters;
            double gb_s = total_read / ms / 1e6;
            printf("  BLOCK=%3u tiles (drain 1-by-1): %.3f ms/iter, %.1f GB/s\n",
                   blk_tiles, per_iter_ms, gb_s);
        }
    }

    // ============================================================
    // Test 3: Full GEMV pipeline (reader + matmul + writer)
    // ============================================================
    printf("\n=== Test 3: Full GEMV pipeline (reader + matmul + writer) ===\n");
    {
        // Test realistic GEMV dimensions
        struct GemvTest { uint32_t Mt; uint32_t Kt; const char* name; };
        GemvTest tests[] = {
            {386, 128, "SSM combined (12352x4096)"},
            {384, 128, "FFN gate/up (12288x4096)"},
            {128, 384, "FFN down (4096x12288)"},
            {128, 128, "outproj (4096x4096)"},
        };

        for (auto& t : tests) {
            uint32_t Mt = t.Mt, Kt = t.Kt;
            uint32_t Mt_per_bank = (Mt + num_banks - 1) / num_banks;
            uint32_t Mt_padded = Mt_per_bank * num_banks;
            uint32_t total_weight_tiles = Mt_padded * Kt;
            uint32_t tiles_per_bank = Mt_per_bank * Kt;
            uint64_t total_weight_bytes = (uint64_t)total_weight_tiles * tile_bytes;

            // Create DRAM-sharded weight buffer
            CoreRange dram_bank_range3({0, 0}, {num_banks - 1, 0});
            auto dram_cores3 = corerange_to_cores(CoreRangeSet(dram_bank_range3));
            BufferDistributionSpec shard_spec3(
                tt::tt_metal::Shape({1, total_weight_tiles}),
                tt::tt_metal::Shape({1, tiles_per_bank}),
                dram_cores3);
            DeviceLocalBufferConfig weight_dram_cfg{
                .page_size = tile_bytes,
                .buffer_type = BufferType::DRAM,
                .sharding_args = BufferShardingArgs(shard_spec3)};
            Shape2D weight_shape = {1, total_weight_tiles};
            distributed::ShardedBufferConfig weight_sharded_cfg{
                .global_size = total_weight_bytes,
                .global_buffer_shape = weight_shape,
                .shard_shape = weight_shape,
                .shard_orientation = ShardOrientation::ROW_MAJOR,
            };
            auto weight_buf = MeshBuffer::create(weight_sharded_cfg, weight_dram_cfg, mesh.get());

            // Create interleaved activation and output buffers (BF16)
            uint32_t bf16_tile_bytes = 2048;
            DeviceLocalBufferConfig bf16_dram_cfg{.page_size = bf16_tile_bytes, .buffer_type = BufferType::DRAM};
            auto act_buf = MeshBuffer::create(
                ReplicatedBufferConfig{.size = Kt * bf16_tile_bytes}, bf16_dram_cfg, mesh.get());
            auto out_buf = MeshBuffer::create(
                ReplicatedBufferConfig{.size = Mt_padded * bf16_tile_bytes}, bf16_dram_cfg, mesh.get());

            // Fill with dummy data
            std::vector<uint32_t> weight_data(total_weight_tiles * (tile_bytes / 4), 0x3F803F80);
            EnqueueWriteMeshBuffer(cq, weight_buf, weight_data, false);
            std::vector<bfloat16> act_data(Kt * TILE_HEIGHT * TILE_WIDTH, bfloat16(0.01f));
            EnqueueWriteMeshBuffer(cq, act_buf, act_data, false);
            Finish(cq);

            // Test different BLOCK sizes with full GEMV compute
            uint32_t block_sizes[] = {4, 16, 64, 128};
            for (uint32_t block : block_sizes) {
                if (block > Kt) continue;

                MeshCoordinateRange dev_range3(mesh->shape());
                Program program3 = CreateProgram();

                std::vector<CoreRange> core_ranges3;
                for (uint32_t b = 0; b < num_banks; b++) {
                    core_ranges3.push_back(CoreRange(dram_workers[b], dram_workers[b]));
                }
                CoreRangeSet all_cores3(core_ranges3);

                // Activation CB
                CircularBufferConfig cb_act_cfg =
                    CircularBufferConfig(Kt * bf16_tile_bytes, {{CBIndex::c_0, tt::DataFormat::Float16_b}})
                        .set_page_size(CBIndex::c_0, bf16_tile_bytes);
                CreateCircularBuffer(program3, all_cores3, cb_act_cfg);

                // Weight CB (double-buffered)
                uint32_t weight_cb_tiles = block * 2;
                CircularBufferConfig cb_weight_cfg =
                    CircularBufferConfig(weight_cb_tiles * tile_bytes, {{CBIndex::c_1, tt::DataFormat::Bfp8_b}})
                        .set_page_size(CBIndex::c_1, tile_bytes);
                CreateCircularBuffer(program3, all_cores3, cb_weight_cfg);

                // Output CB
                CircularBufferConfig cb_out_cfg =
                    CircularBufferConfig(2 * bf16_tile_bytes, {{CBIndex::c_16, tt::DataFormat::Float16_b}})
                        .set_page_size(CBIndex::c_16, bf16_tile_bytes);
                CreateCircularBuffer(program3, all_cores3, cb_out_cfg);

                // Reader kernel
                std::vector<uint32_t> reader_ct_args = {(uint32_t)CBIndex::c_0, (uint32_t)CBIndex::c_1, Kt, block};
                TensorAccessorArgs(*act_buf).append_to(reader_ct_args);
                auto reader_kid = CreateKernel(program3,
                    "/home/ubuntu/qwen3.5-9b-bf16-1xn300d/tt_metal/kernels/dataflow/reader_gemv_dram_sharded.cpp",
                    all_cores3,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                       .noc = NOC::RISCV_1_default,
                                       .compile_args = reader_ct_args});

                // Compute kernel
                std::vector<uint32_t> compute_ct_args = {Kt, block};
                auto compute_kid = CreateKernel(program3,
                    "/home/ubuntu/qwen3.5-9b-bf16-1xn300d/tt_metal/kernels/compute/gemv.cpp",
                    all_cores3,
                    ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_ct_args});

                // Writer kernel
                std::vector<uint32_t> writer_ct_args = {(uint32_t)CBIndex::c_16};
                TensorAccessorArgs(*out_buf).append_to(writer_ct_args);
                auto writer_kid = CreateKernel(program3,
                    "/home/ubuntu/qwen3.5-9b-bf16-1xn300d/tt_metal/kernels/dataflow/writer_gemv_multicore.cpp",
                    all_cores3,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                                       .noc = NOC::RISCV_0_default,
                                       .compile_args = writer_ct_args});

                for (uint32_t b = 0; b < num_banks; b++) {
                    uint32_t start_row = b * Mt_per_bank;
                    uint32_t mt_this = (start_row >= Mt) ? 0 : std::min(Mt_per_bank, Mt - start_row);
                    SetRuntimeArgs(program3, reader_kid, dram_workers[b],
                                   {(uint32_t)act_buf->address(), (uint32_t)weight_buf->address(), mt_this, b});
                    SetRuntimeArgs(program3, compute_kid, dram_workers[b], {mt_this});
                    SetRuntimeArgs(program3, writer_kid, dram_workers[b],
                                   {(uint32_t)out_buf->address(), mt_this, start_row});
                }

                MeshWorkload workload3;
                workload3.add_program(dev_range3, std::move(program3));

                // Warmup
                EnqueueMeshWorkload(cq, workload3, false);
                Finish(cq);

                // Time
                int iters = 20;
                auto t0 = Clock::now();
                for (int i = 0; i < iters; i++) {
                    EnqueueMeshWorkload(cq, workload3, false);
                }
                Finish(cq);
                double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

                double per_iter_ms = ms / iters;
                double gb_s = (double)total_weight_bytes / per_iter_ms / 1e6;
                printf("  %s: BLOCK=%3u → %.3f ms, %.1f GB/s\n",
                       t.name, block, per_iter_ms, gb_s);
            }
        }
    }

    // ============================================================
    // Test 4: Measure per-dispatch overhead with trace replay
    // ============================================================
    printf("\n=== Test 4: Per-dispatch overhead (trace replay) ===\n");
    {
        // Create a minimal GEMV (outproj 4096x4096) and measure with/without trace
        uint32_t Mt = 128, Kt = 128;
        uint32_t Mt_per_bank = (Mt + num_banks - 1) / num_banks;
        uint32_t Mt_padded = Mt_per_bank * num_banks;
        uint32_t total_weight_tiles = Mt_padded * Kt;
        uint32_t tiles_per_bank = Mt_per_bank * Kt;

        CoreRange dram_bank_range4({0, 0}, {num_banks - 1, 0});
        auto dram_cores4 = corerange_to_cores(CoreRangeSet(dram_bank_range4));
        BufferDistributionSpec shard_spec4(
            tt::tt_metal::Shape({1, total_weight_tiles}),
            tt::tt_metal::Shape({1, tiles_per_bank}),
            dram_cores4);
        DeviceLocalBufferConfig weight_dram_cfg4{
            .page_size = tile_bytes,
            .buffer_type = BufferType::DRAM,
            .sharding_args = BufferShardingArgs(shard_spec4)};
        auto weight_buf4 = MeshBuffer::create(
            distributed::ShardedBufferConfig{
                .global_size = (uint64_t)total_weight_tiles * tile_bytes,
                .global_buffer_shape = {1, total_weight_tiles},
                .shard_shape = {1, total_weight_tiles},
                .shard_orientation = ShardOrientation::ROW_MAJOR},
            weight_dram_cfg4, mesh.get());

        uint32_t bf16_tile_bytes = 2048;
        DeviceLocalBufferConfig bf16_dram_cfg4{.page_size = bf16_tile_bytes, .buffer_type = BufferType::DRAM};
        auto act_buf4 = MeshBuffer::create(
            ReplicatedBufferConfig{.size = Kt * bf16_tile_bytes}, bf16_dram_cfg4, mesh.get());
        auto out_buf4 = MeshBuffer::create(
            ReplicatedBufferConfig{.size = Mt_padded * bf16_tile_bytes}, bf16_dram_cfg4, mesh.get());

        auto build_gemv_program = [&]() -> Program {
            Program prog = CreateProgram();
            std::vector<CoreRange> cr;
            for (uint32_t b = 0; b < num_banks; b++)
                cr.push_back(CoreRange(dram_workers[b], dram_workers[b]));
            CoreRangeSet ac(cr);

            CircularBufferConfig cba =
                CircularBufferConfig(Kt * bf16_tile_bytes, {{CBIndex::c_0, tt::DataFormat::Float16_b}})
                    .set_page_size(CBIndex::c_0, bf16_tile_bytes);
            CreateCircularBuffer(prog, ac, cba);
            CircularBufferConfig cbw =
                CircularBufferConfig(32 * tile_bytes, {{CBIndex::c_1, tt::DataFormat::Bfp8_b}})
                    .set_page_size(CBIndex::c_1, tile_bytes);
            CreateCircularBuffer(prog, ac, cbw);
            CircularBufferConfig cbo =
                CircularBufferConfig(2 * bf16_tile_bytes, {{CBIndex::c_16, tt::DataFormat::Float16_b}})
                    .set_page_size(CBIndex::c_16, bf16_tile_bytes);
            CreateCircularBuffer(prog, ac, cbo);

            std::vector<uint32_t> rca = {(uint32_t)CBIndex::c_0, (uint32_t)CBIndex::c_1, Kt, 16u};
            TensorAccessorArgs(*act_buf4).append_to(rca);
            auto rk = CreateKernel(prog,
                "/home/ubuntu/qwen3.5-9b-bf16-1xn300d/tt_metal/kernels/dataflow/reader_gemv_dram_sharded.cpp", ac,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                   .noc = NOC::RISCV_1_default, .compile_args = rca});
            auto ck = CreateKernel(prog,
                "/home/ubuntu/qwen3.5-9b-bf16-1xn300d/tt_metal/kernels/compute/gemv.cpp", ac,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = {Kt, 16u}});
            std::vector<uint32_t> wca = {(uint32_t)CBIndex::c_16};
            TensorAccessorArgs(*out_buf4).append_to(wca);
            auto wk = CreateKernel(prog,
                "/home/ubuntu/qwen3.5-9b-bf16-1xn300d/tt_metal/kernels/dataflow/writer_gemv_multicore.cpp", ac,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                                   .noc = NOC::RISCV_0_default, .compile_args = wca});
            for (uint32_t b = 0; b < num_banks; b++) {
                uint32_t sr = b * Mt_per_bank;
                uint32_t mt = (sr >= Mt) ? 0 : std::min(Mt_per_bank, Mt - sr);
                SetRuntimeArgs(prog, rk, dram_workers[b],
                    {(uint32_t)act_buf4->address(), (uint32_t)weight_buf4->address(), mt, b});
                SetRuntimeArgs(prog, ck, dram_workers[b], {mt});
                SetRuntimeArgs(prog, wk, dram_workers[b],
                    {(uint32_t)out_buf4->address(), mt, sr});
            }
            return prog;
        };

        MeshCoordinateRange dev_range4(mesh->shape());

        // Test A: Dispatch via MeshWorkload (no trace)
        {
            MeshWorkload wl;
            wl.add_program(dev_range4, build_gemv_program());
            EnqueueMeshWorkload(cq, wl, false);
            Finish(cq);

            int iters = 100;
            auto t0 = Clock::now();
            for (int i = 0; i < iters; i++) {
                EnqueueMeshWorkload(cq, wl, false);
            }
            Finish(cq);
            double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            printf("  MeshWorkload dispatch: %.3f ms/iter (100 iters)\n", ms / iters);
        }

        // Test B: Dispatch via trace replay
        {
            MeshWorkload wl;
            wl.add_program(dev_range4, build_gemv_program());
            EnqueueMeshWorkload(cq, wl, false);
            Finish(cq);

            auto tid = mesh->begin_mesh_trace(0);
            EnqueueMeshWorkload(cq, wl, false);
            mesh->end_mesh_trace(0, tid);

            int iters = 100;
            auto t0 = Clock::now();
            for (int i = 0; i < iters; i++) {
                mesh->replay_mesh_trace(0, tid, false);
            }
            Finish(cq);
            double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            printf("  Trace replay: %.3f ms/iter (100 iters)\n", ms / iters);
            mesh->release_mesh_trace(tid);
        }

        // Test C: Replay + blocking read (simulates forward pass pattern)
        {
            MeshWorkload wl;
            wl.add_program(dev_range4, build_gemv_program());
            EnqueueMeshWorkload(cq, wl, false);
            Finish(cq);

            auto tid = mesh->begin_mesh_trace(0);
            EnqueueMeshWorkload(cq, wl, false);
            mesh->end_mesh_trace(0, tid);

            // Read buffer
            std::vector<bfloat16> host_read(Mt_padded * TILE_HEIGHT * TILE_WIDTH);

            // Warmup
            for (int i = 0; i < 5; i++) {
                mesh->replay_mesh_trace(0, tid, false);
                EnqueueReadMeshBuffer(cq, host_read, out_buf4, true);
            }

            // Measure: replay + blocking read (like forward pass)
            int iters = 100;
            auto t0 = Clock::now();
            for (int i = 0; i < iters; i++) {
                mesh->replay_mesh_trace(0, tid, false);
                EnqueueReadMeshBuffer(cq, host_read, out_buf4, true);
            }
            double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            printf("  Replay + blocking read: %.3f ms/iter\n", ms / iters);

            // Measure: replay only (pipelined, no read)
            auto t1 = Clock::now();
            for (int i = 0; i < iters; i++) {
                mesh->replay_mesh_trace(0, tid, false);
            }
            Finish(cq);
            double ms2 = std::chrono::duration<double, std::milli>(Clock::now() - t1).count();
            printf("  Replay only (pipelined): %.3f ms/iter\n", ms2 / iters);

            // Measure: Finish per iteration (like blocking but no read)
            auto t2 = Clock::now();
            for (int i = 0; i < iters; i++) {
                mesh->replay_mesh_trace(0, tid, false);
                Finish(cq);
            }
            double ms3 = std::chrono::duration<double, std::milli>(Clock::now() - t2).count();
            printf("  Replay + Finish: %.3f ms/iter\n", ms3 / iters);

            // Measure: Read overhead only (no replay, just read same buffer)
            auto t3 = Clock::now();
            for (int i = 0; i < iters; i++) {
                EnqueueReadMeshBuffer(cq, host_read, out_buf4, true);
            }
            double ms4 = std::chrono::duration<double, std::milli>(Clock::now() - t3).count();
            printf("  Read only (no replay): %.3f ms/iter\n", ms4 / iters);

            mesh->release_mesh_trace(tid);
        }

        // Test D: Simulate forward pass pattern (replay norm + replay ffn + blocking read)
        {
            // Build 2 traces: "norm" trace and "ffn" trace
            MeshWorkload norm_wl, ffn_wl;
            norm_wl.add_program(dev_range4, build_gemv_program());
            ffn_wl.add_program(dev_range4, build_gemv_program());
            EnqueueMeshWorkload(cq, norm_wl, false);
            Finish(cq);
            EnqueueMeshWorkload(cq, ffn_wl, false);
            Finish(cq);

            auto norm_tid = mesh->begin_mesh_trace(0);
            EnqueueMeshWorkload(cq, norm_wl, false);
            mesh->end_mesh_trace(0, norm_tid);

            auto ffn_tid = mesh->begin_mesh_trace(0);
            EnqueueMeshWorkload(cq, ffn_wl, false);
            mesh->end_mesh_trace(0, ffn_tid);

            std::vector<bfloat16> host_read(Mt_padded * TILE_HEIGHT * TILE_WIDTH);

            // Simulate forward pass: replay_norm → read → replay_ffn → (next iter: replay_norm → read → ...)
            int iters = 100;
            auto t0 = Clock::now();
            for (int i = 0; i < iters; i++) {
                mesh->replay_mesh_trace(0, norm_tid, false);
                EnqueueReadMeshBuffer(cq, host_read, out_buf4, true);
                // Simulate host work (2ms)
                auto tw = Clock::now();
                while (std::chrono::duration<double, std::milli>(Clock::now() - tw).count() < 2.0) {}
                // Write back result
                EnqueueWriteMeshBuffer(cq, act_buf4, host_read, false);
                mesh->replay_mesh_trace(0, ffn_tid, false);
            }
            Finish(cq);
            double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            printf("  Simulated forward (norm+read+2ms_host+write+ffn): %.3f ms/iter\n", ms / iters);

            mesh->release_mesh_trace(norm_tid);
            mesh->release_mesh_trace(ffn_tid);
        }

        // Test E: Multiple GEMV programs in single trace (like FFN chain)
        {
            // Create 4 GEMV workloads (outproj + gate + up + down in one trace)
            std::vector<MeshWorkload> wls(4);
            for (int p = 0; p < 4; p++) {
                wls[p].add_program(dev_range4, build_gemv_program());
                EnqueueMeshWorkload(cq, wls[p], false);
                Finish(cq);
            }

            auto tid = mesh->begin_mesh_trace(0);
            for (int p = 0; p < 4; p++) {
                EnqueueMeshWorkload(cq, wls[p], false);
            }
            mesh->end_mesh_trace(0, tid);

            int iters = 100;
            auto t0 = Clock::now();
            for (int i = 0; i < iters; i++) {
                mesh->replay_mesh_trace(0, tid, false);
            }
            Finish(cq);
            double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            printf("  4-GEMV trace replay: %.3f ms/iter (vs %.3f ms for 4×single)\n",
                   ms / iters, 4 * 0.176);
            mesh->release_mesh_trace(tid);
        }

        // Test F: Realistic SSM layer sim with correct matrix sizes
        {
            // SSM combined: 12352 × 4096 (Mt=386, Kt=128)
            // Create realistic buffers
            uint32_t ssm_Mt = 396; // padded to 12 banks
            uint32_t ssm_Kt = 128;
            uint32_t ssm_weight_tiles = ssm_Mt * ssm_Kt;
            uint32_t ssm_tpb = ssm_weight_tiles / num_banks;

            CoreRange dbr({0, 0}, {num_banks - 1, 0});
            auto dc = corerange_to_cores(CoreRangeSet(dbr));
            auto ssm_weight = MeshBuffer::create(
                distributed::ShardedBufferConfig{
                    .global_size = (uint64_t)ssm_weight_tiles * tile_bytes,
                    .global_buffer_shape = {1, ssm_weight_tiles},
                    .shard_shape = {1, ssm_weight_tiles},
                    .shard_orientation = ShardOrientation::ROW_MAJOR},
                DeviceLocalBufferConfig{.page_size = tile_bytes, .buffer_type = BufferType::DRAM,
                    .sharding_args = BufferShardingArgs(BufferDistributionSpec(
                        tt::tt_metal::Shape({1, ssm_weight_tiles}),
                        tt::tt_metal::Shape({1, ssm_tpb}), dc))},
                mesh.get());
            auto ssm_act = MeshBuffer::create(
                ReplicatedBufferConfig{.size = ssm_Kt * bf16_tile_bytes},
                bf16_dram_cfg4, mesh.get());
            auto ssm_out = MeshBuffer::create(
                ReplicatedBufferConfig{.size = ssm_Mt * bf16_tile_bytes},
                bf16_dram_cfg4, mesh.get());

            // Build SSM GEMV program
            auto build_ssm_gemv = [&]() -> Program {
                Program prog = CreateProgram();
                std::vector<CoreRange> cr;
                for (uint32_t b = 0; b < num_banks; b++)
                    cr.push_back(CoreRange(dram_workers[b], dram_workers[b]));
                CoreRangeSet ac(cr);

                CreateCircularBuffer(prog, ac,
                    CircularBufferConfig(ssm_Kt * bf16_tile_bytes, {{CBIndex::c_0, tt::DataFormat::Float16_b}})
                        .set_page_size(CBIndex::c_0, bf16_tile_bytes));
                CreateCircularBuffer(prog, ac,
                    CircularBufferConfig(32 * tile_bytes, {{CBIndex::c_1, tt::DataFormat::Bfp8_b}})
                        .set_page_size(CBIndex::c_1, tile_bytes));
                CreateCircularBuffer(prog, ac,
                    CircularBufferConfig(2 * bf16_tile_bytes, {{CBIndex::c_16, tt::DataFormat::Float16_b}})
                        .set_page_size(CBIndex::c_16, bf16_tile_bytes));

                std::vector<uint32_t> rca = {(uint32_t)CBIndex::c_0, (uint32_t)CBIndex::c_1, ssm_Kt, 16u};
                TensorAccessorArgs(*ssm_act).append_to(rca);
                auto rk = CreateKernel(prog,
                    "/home/ubuntu/qwen3.5-9b-bf16-1xn300d/tt_metal/kernels/dataflow/reader_gemv_dram_sharded.cpp", ac,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                       .noc = NOC::RISCV_1_default, .compile_args = rca});
                auto ck = CreateKernel(prog,
                    "/home/ubuntu/qwen3.5-9b-bf16-1xn300d/tt_metal/kernels/compute/gemv.cpp", ac,
                    ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = {ssm_Kt, 16u}});
                std::vector<uint32_t> wca = {(uint32_t)CBIndex::c_16};
                TensorAccessorArgs(*ssm_out).append_to(wca);
                auto wk = CreateKernel(prog,
                    "/home/ubuntu/qwen3.5-9b-bf16-1xn300d/tt_metal/kernels/dataflow/writer_gemv_multicore.cpp", ac,
                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                                       .noc = NOC::RISCV_0_default, .compile_args = wca});

                uint32_t ssm_Mt_per_bank = ssm_Mt / num_banks;
                for (uint32_t b = 0; b < num_banks; b++) {
                    SetRuntimeArgs(prog, rk, dram_workers[b],
                        {(uint32_t)ssm_act->address(), (uint32_t)ssm_weight->address(), ssm_Mt_per_bank, b});
                    SetRuntimeArgs(prog, ck, dram_workers[b], {ssm_Mt_per_bank});
                    SetRuntimeArgs(prog, wk, dram_workers[b],
                        {(uint32_t)ssm_out->address(), ssm_Mt_per_bank, b * ssm_Mt_per_bank});
                }
                return prog;
            };

            MeshWorkload ssm_wl;
            ssm_wl.add_program(dev_range4, build_ssm_gemv());
            EnqueueMeshWorkload(cq, ssm_wl, false);
            Finish(cq);

            // Measure single SSM GEMV with blocking read (realistic pattern)
            auto ssm_tid = mesh->begin_mesh_trace(0);
            EnqueueMeshWorkload(cq, ssm_wl, false);
            mesh->end_mesh_trace(0, ssm_tid);

            std::vector<bfloat16> ssm_host_read(ssm_Mt * TILE_HEIGHT * TILE_WIDTH);

            // Warmup
            mesh->replay_mesh_trace(0, ssm_tid, false);
            EnqueueReadMeshBuffer(cq, ssm_host_read, ssm_out, true);

            int iters = 50;
            // Pipelined (no read)
            auto t0 = Clock::now();
            for (int i = 0; i < iters; i++) {
                mesh->replay_mesh_trace(0, ssm_tid, false);
            }
            Finish(cq);
            double ms0 = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

            // With blocking read
            auto t1 = Clock::now();
            for (int i = 0; i < iters; i++) {
                mesh->replay_mesh_trace(0, ssm_tid, false);
                EnqueueReadMeshBuffer(cq, ssm_host_read, ssm_out, true);
            }
            double ms1 = std::chrono::duration<double, std::milli>(Clock::now() - t1).count();

            // Simulated SSM layer (with 2.1ms host work)
            auto t2 = Clock::now();
            for (int i = 0; i < iters; i++) {
                mesh->replay_mesh_trace(0, ssm_tid, false);
                EnqueueReadMeshBuffer(cq, ssm_host_read, ssm_out, true);
                auto tw = Clock::now();
                while (std::chrono::duration<double, std::milli>(Clock::now() - tw).count() < 2.1) {}
                // Simulate FFN chain (4 GEMVs worth of work)
                mesh->replay_mesh_trace(0, ssm_tid, false);  // reuse as proxy
            }
            Finish(cq);
            double ms2 = std::chrono::duration<double, std::milli>(Clock::now() - t2).count();

            printf("  SSM GEMV (12352x4096) pipelined: %.3f ms\n", ms0 / iters);
            printf("  SSM GEMV (12352x4096) + blocking read: %.3f ms\n", ms1 / iters);
            printf("  SSM layer sim (GEMV+read+2.1ms_host+GEMV): %.3f ms\n", ms2 / iters);

            mesh->release_mesh_trace(ssm_tid);
        }
    }

    mesh->close();
    printf("\nDone.\n");
    return 0;
}
