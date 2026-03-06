#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CUDA_CHECK(err) do { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %d at %s:%d: %s\n", \
                err_, __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(1); \
    } \
} while (0)

// Allocate GPU memory and return pointer
template <typename T>
T* cuda_alloc(size_t count) {
    T* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

// Copy host -> device
template <typename T>
void cuda_upload(T* dst, const T* src, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice));
}

// Copy device -> host
template <typename T>
void cuda_download(T* dst, const T* src, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost));
}

// Ceiling division
inline int cdiv(int a, int b) { return (a + b - 1) / b; }
