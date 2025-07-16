#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda_runtime.h>
#include <iostream>

inline void checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " - " << cudaGetErrorString(err) << std::endl;
        exit(err);
    }
}

#define CHECK_CUDA(call) checkCudaError((call), __FILE__, __LINE__)

inline void allocateDeviceMemory(float** d_ptr, size_t size) {
    CHECK_CUDA(cudaMalloc((void**)d_ptr, size));
}

inline void copyToDevice(float* d_ptr, const float* h_ptr, size_t size) {
    CHECK_CUDA(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));
}

inline void copyToHost(float* h_ptr, const float* d_ptr, size_t size) {
    CHECK_CUDA(cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost));
}

inline void freeDeviceMemory(float* d_ptr) {
    CHECK_CUDA(cudaFree(d_ptr));
}

#endif // UTILS_CUH