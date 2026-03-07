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

# Test: device validation
add_executable(test_device tests/test_device.cpp)
target_link_libraries(test_device PRIVATE TT::Metalium)

# Test: basic matmul
add_executable(test_matmul tests/test_matmul.cpp)
target_link_libraries(test_matmul PRIVATE TT::Metalium)

# Test: weight loading
add_executable(test_load_weights tests/test_load_weights.cpp host/gguf_loader.cpp)
target_link_libraries(test_load_weights PRIVATE TT::Metalium)
target_include_directories(test_load_weights PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/host)

# Engine library
add_library(qwen_engine STATIC host/engine.cpp host/gguf_loader.cpp ../src/tokenizer.cpp)
target_link_libraries(qwen_engine PUBLIC TT::Metalium)
target_include_directories(qwen_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/host ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_options(qwen_engine PRIVATE -march=native -ffast-math)

# Link ttnn C++ library for device-side ops (matmul, rmsnorm, etc.)
target_link_libraries(qwen_engine PUBLIC ${TT_METAL_BUILD}/lib/_ttnncpp.so)
target_include_directories(qwen_engine PUBLIC ${TT_METAL_BUILD}/include)

# Test: forward pass (single token through all layers)
add_executable(test_forward tests/test_forward.cpp)
target_link_libraries(test_forward PRIVATE qwen_engine)

# Bandwidth benchmark
add_executable(test_bandwidth /tmp/test_bandwidth2.cpp)
target_link_libraries(test_bandwidth PRIVATE TT::Metalium ${TT_METAL_BUILD}/lib/_ttnncpp.so)
target_include_directories(test_bandwidth PRIVATE ${TT_METAL_BUILD}/include)
