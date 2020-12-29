#pragma once

#include <stdio.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "concat.h"
#include "cuda_fp16.h"

#ifndef GET_THREADS
#define GET_THREADS 1024
#endif

#ifndef GET_BLOCKS
#define GET_BLOCKS(x) \
  (((x) - 1) / (GET_THREADS) + 1)
#endif

#ifndef CUDA_CHECK

#define CUDA_CHECK(code)                                                  \
  {                                                                       \
    cudaError_t status = (code);                                         \
    if ((status) != cudaSuccess) {                                          \
      fprintf(stderr, "CUDA error in file: %s, line: %d, %s\n", __FILE__, \
              __LINE__, cudaGetErrorString((status)));                      \
      exit((status));                                                       \
    }                                                                     \
  }
#endif

template <typename T>
__global__ void Concat(const size_t size, const int w1, const int w2, const T* input_1, const T* input_2, T* output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    int n = pos / (w1 + w2);
    int m = pos % (w1 + w2);
    output[pos] = m >= w1 ? input_2[n * w2 + m - w1] : input_1[n * w1 + m];
  }
  return;
}

template <typename T>
__global__ void Concat(const size_t size, const int w1, const int w2, const int w3,
                       const T* input_1, const T* input_2, const T* input_3, T* output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    int n = pos / (w1 + w2 + w3);
    int m = pos % (w1 + w2 + w3);
    output[pos] = m < w1 ? input_1[n * w1 + m] :
                    m < w1 + w2 ? input_2[n * w2 + m - w1] :
                      input_3[n * w3 + m - w1 - w2];
  }
  return;
}

template <typename T>
__global__ __launch_bounds__(2) void Concat(const size_t size, const int w1, const int w2, const int w3, const int w4,
                       const T* input_1, const T* input_2, const T* input_3, const T* input_4, T* output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    int n = pos / (w1 + w2 + w3 + w4);
    int m = pos % (w1 + w2 + w3 + w4);
    output[pos] = m < w1 ? input_1[n * w1 + m] :
                    m < w1 + w2 ? input_2[n * w2 + m - w1]:
                      m < w1 + w2 + w3 ? input_3[n * w3 + m - w1 - w2]:
                        input_4[n * w4 + m - w1 - w2 - w3];
  }
  return;
}

template <typename T>
void ConcatKernel(const size_t size, const int w1, const int w2, const T* input_1, const T* input_2, T* output,
                 cudaStream_t cuda_stream) {
  Concat<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, w1, w2, input_1, input_2, output);
  return;
}

template <typename T>
void ConcatKernel(const size_t size, const int w1, const int w2, const int w3,
                  const T* input_1, const T* input_2, const T* input_3, T* output,
                  cudaStream_t cuda_stream) {
  Concat<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, w1, w2, w3, input_1, input_2, input_3, output);
  return;
}

template <typename T>
void ConcatKernel(const size_t size, const int w1, const int w2, const int w3, const int w4,
                  const T* input_1, const T* input_2, const T* input_3, const T* input_4, T* output,
                  cudaStream_t cuda_stream) {
  Concat<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, w1, w2, w3, w4, input_1,
                                                            input_2, input_3, input_4, output);
  return;
}

template void ConcatKernel(const size_t size, const int w1, const int w2, const float* input_1, const float* input_2,
                           float* output, cudaStream_t cuda_stream);
template void ConcatKernel(const size_t size, const int w1, const int w2, const int* input_1, const int* input_2,
                           int* output, cudaStream_t cuda_stream);
template void ConcatKernel(const size_t size, const int w1, const int w2, const half* input_1, const half* input_2,
                           half* output, cudaStream_t cuda_stream);

template void ConcatKernel(const size_t size, const int w1, const int w2, const int w3,
                           const float* input_1, const float* input_2, const float* input_3,
                           float* output, cudaStream_t cuda_stream);
template void ConcatKernel(const size_t size, const int w1, const int w2, const int w3,
                           const int* input_1, const int* input_2, const int* input_3,
                           int* output, cudaStream_t cuda_stream);
template void ConcatKernel(const size_t size, const int w1, const int w2, const int w3,
                           const half* input_1, const half* input_2, const half* input_3,
                           half* output, cudaStream_t cuda_stream);

template void ConcatKernel(const size_t size, const int w1, const int w2, const int w3, const int w4,
                           const float* input_1, const float* input_2, const float* input_3, const float* input_4,
                           float* output, cudaStream_t cuda_stream);
template void ConcatKernel(const size_t size, const int w1, const int w2, const int w3, const int w4,
                           const int* input_1, const int* input_2, const int* input_3, const int* input_4,
                           int* output, cudaStream_t cuda_stream);
template void ConcatKernel(const size_t size, const int w1, const int w2, const int w3, const int w4,
                           const half* input_1, const half* input_2, const half* input_3, const half* input_4,
                           half* output, cudaStream_t cuda_stream);
