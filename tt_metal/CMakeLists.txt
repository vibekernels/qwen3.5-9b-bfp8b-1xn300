cmake_minimum_required(VERSION 3.20)
project(qwen-tt-metal CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# tt-metal build directory
set(TT_METAL_BUILD "/home/ubuntu/tt-metal/build_Release" CACHE PATH "tt-metal build directory")

# Use the exported CMake package from lib/cmake/tt-metalium
set(tt-metalium_DIR "${TT_METAL_BUILD}/lib/cmake/tt-metalium")

# Dependencies shipped by the tt-metal build
list(PREPEND CMAKE_PREFIX_PATH
    "${TT_METAL_BUILD}/lib/cmake"
    "${TT_METAL_BUILD}/share/cmake"
    "${TT_METAL_BUILD}"
)

find_package(TT-Metalium REQUIRED CONFIG)

# Shared source files (CPU-side, reused from original project)
set(SHARED_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/tokenizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/download.cpp
)

# Engine library
add_library(qwen_engine STATIC host/engine.cpp host/gguf_loader.cpp ../src/tokenizer.cpp)
target_link_libraries(qwen_engine PUBLIC TT::Metalium)
target_include_directories(qwen_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/host ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_options(qwen_engine PRIVATE -march=native -ffast-math)
target_compile_definitions(qwen_engine PRIVATE KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels")

# Additional include paths from tt-metal build
target_include_directories(qwen_engine PUBLIC ${TT_METAL_BUILD}/include)

# Test: forward pass (single token through all layers)
add_executable(test_forward tests/test_forward.cpp)
target_link_libraries(test_forward PRIVATE qwen_engine)

# Test: basic matmul
add_executable(test_matmul tests/test_matmul.cpp)
target_link_libraries(test_matmul PRIVATE TT::Metalium)

# Mesh overhead benchmark
add_executable(test_mesh_overhead tests/test_mesh_overhead.cpp)
target_link_libraries(test_mesh_overhead PRIVATE TT::Metalium)

# DRAM bandwidth benchmark (on-device kernel reader)
add_executable(test_dram_bw tests/test_dram_bw.cpp)
target_link_libraries(test_dram_bw PRIVATE TT::Metalium)

# Integration test suite (requires model file + device)
add_executable(test_inference tests/test_inference.cpp ../src/download.cpp)
target_link_libraries(test_inference PRIVATE qwen_engine)
target_include_directories(test_inference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party)

# CTest support: `make test` runs the integration suite
enable_testing()
add_test(NAME inference
    COMMAND ${CMAKE_COMMAND} -E env
        TT_METAL_RUNTIME_ROOT=/home/ubuntu/tt-metal
        MODEL_PATH=/home/ubuntu/qwen3.5-9b-bf16-1x5090/models/Qwen3.5-9B-BF16.gguf
        QUIET=1
        $<TARGET_FILE:test_inference>
)
set_tests_properties(inference PROPERTIES TIMEOUT 900)
