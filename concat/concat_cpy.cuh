#pragma once

#include <stdio.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "concat.h"

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
void ConcatKernelCpy(const size_t size, const int w1, const int w2, const T* input_1, const T* input_2, T* output) {
  int item_size = sizeof(T);
  int n_outer = size / (w1 + w2);

  const T* in1 = input_1;
  const T* in2 = input_2;
  T* out = output;
  for (int i = 0; i < n_outer; ++i) {
    CUDA_CHECK(cudaMemcpy(out     , in1, item_size * w1,  cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(out + w1, in2, item_size * w2,  cudaMemcpyDeviceToDevice));
    out += w1 + w2;
    in1 += w1;
    in2 += w2;
  }
  return;
}

template <typename T>
void ConcatKernelCpy(const size_t size, const int w1, const int w2, const int w3,
                  const T* input_1, const T* input_2, const T* input_3, T* output) {
  int item_size = sizeof(T);
  int n_outer = size / (w1 + w2 + w3);

  const T* in1 = input_1;
  const T* in2 = input_2;
  const T* in3 = input_3;
  T* out = output;
  for (int i = 0; i < n_outer; ++i) {
    CUDA_CHECK(cudaMemcpy(out          , in1, item_size * w1,  cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(out + w1     , in2, item_size * w2,  cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(out + w1 + w2, in3, item_size * w3,  cudaMemcpyDeviceToDevice));
    out += w1 + w2 + w3;
    in1 += w1;
    in2 += w2;
    in3 += w3;
  }
  return;
}

template <typename T>
void ConcatKernelCpy(const size_t size, const int w1, const int w2, const int w3, const int w4,
                  const T* input_1, const T* input_2, const T* input_3, const T* input_4, T* output) {
  int item_size = sizeof(T);
  int n_outer = size / (w1 + w2 + w3 + w4);

  const T* in1 = input_1;
  const T* in2 = input_2;
  const T* in3 = input_3;
  const T* in4 = input_4;
  T* out = output;
  for (int i = 0; i < n_outer; ++i) {
    CUDA_CHECK(cudaMemcpy(out               , in1, item_size * w1,  cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(out + w1          , in2, item_size * w2,  cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(out + w1 + w2     , in3, item_size * w3,  cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(out + w1 + w2 + w3, in4, item_size * w4,  cudaMemcpyDeviceToDevice));
    out += w1 + w2 + w3 + w4;
    in1 += w1;
    in2 += w2;
    in3 += w3;
    in4 += w4;
  }
  return;
}