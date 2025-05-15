#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include <iostream>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_CHECK(call) do { \
  cudaError_t err = call; \
  if(err != cudaSuccess) { \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                << " code=" << err << " \"" << cudaGetErrorString(err) << "\"\n"; \
      exit(1); \
  } \
} while(0)

// Kernel for vector addition.
__global__ void vectorAddKernel(float* a, const float* b, int n);

// Function to launch the kernel.
void vectorAdd(float* a, const float* b, int n, cudaStream_t stream);

#endif // COMMON_H
