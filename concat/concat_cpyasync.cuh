#pragma once

#include <stdio.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "concat_ms.cuh"
#include <iostream>

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

static const int base = 2;


template <typename T>
void ConcatKernelCpyAsync(const size_t size, const int w1, const int w2, 
                          const T* input_1, const T* input_2, T* output,
                          cudaStream_t stream1, cudaStream_t stream2) {
  int item_size = sizeof(T);
  int n_outer = size / (w1 + w2);

  int n_outer_cpy = n_outer / base;
  int n_outer_krn = n_outer - n_outer_cpy;

  const T* in1 = input_1;
  const T* in2 = input_2;
  T* out = output;
  for (int i = 0; i < n_outer_cpy; ++i) {
    CUDA_CHECK(cudaMemcpyAsync(out     , in1, item_size * w1,  cudaMemcpyDeviceToDevice, stream1));
    CUDA_CHECK(cudaMemcpyAsync(out + w1, in2, item_size * w2,  cudaMemcpyDeviceToDevice, stream1));
    out += w1 + w2;
    in1 += w1;
    in2 += w2;
  }

  const T* in1_krn = input_1 + n_outer_cpy * w1;
  const T* in2_krn = input_2 + n_outer_cpy * w2;
  T* out_krn = output + n_outer_cpy * (w1 + w2);
  ConcatKernel(n_outer_krn * (w1 + w2), w1, w2, in1_krn, in2_krn, out_krn, stream2);
  return;
}

template <typename T>
void ConcatKernelCpyAsync(const size_t size, const int w1, const int w2, const int w3,
                          const T* input_1, const T* input_2, const T* input_3, T* output,
                          cudaStream_t stream1, cudaStream_t stream2) {
  int item_size = sizeof(T);
  int n_outer = size / (w1 + w2 + w3);

  int n_outer_cpy = n_outer / base;
  int n_outer_krn = n_outer - n_outer_cpy;

  const T* in1 = input_1;
  const T* in2 = input_2;
  const T* in3 = input_3;
  T* out = output;
  for (int i = 0; i < n_outer_cpy; ++i) {
    CUDA_CHECK(cudaMemcpyAsync(out          , in1, item_size * w1,  cudaMemcpyDeviceToDevice, stream1));
    CUDA_CHECK(cudaMemcpyAsync(out + w1     , in2, item_size * w2,  cudaMemcpyDeviceToDevice, stream1));
    CUDA_CHECK(cudaMemcpyAsync(out + w1 + w2, in3, item_size * w3,  cudaMemcpyDeviceToDevice, stream1));
    out += w1 + w2 + w3;
    in1 += w1;
    in2 += w2;
    in3 += w3;
  }

  const T* in1_krn = input_1 + n_outer_cpy * w1;
  const T* in2_krn = input_2 + n_outer_cpy * w2;
  const T* in3_krn = input_3 + n_outer_cpy * w3;
  T* out_krn = output + n_outer_cpy * (w1 + w2 + w3);
  ConcatKernel(n_outer_krn * (w1 + w2 + w3), w1, w2, w3, in1_krn, in2_krn, in3_krn, out_krn, stream2);
  return;
}

template <typename T>
void ConcatKernelCpyAsync(const size_t size, const int w1, const int w2, const int w3, const int w4,
                          const T* input_1, const T* input_2, const T* input_3, const T* input_4, T* output,
                          cudaStream_t stream1, cudaStream_t stream2) {
  int item_size = sizeof(T);
  int n_outer = size / (w1 + w2 + w3 + w4);

  int n_outer_cpy = n_outer / base;
  int n_outer_krn = n_outer - n_outer_cpy;

  const T* in1 = input_1;
  const T* in2 = input_2;
  const T* in3 = input_3;
  const T* in4 = input_4;
  T* out = output;
  for (int i = 0; i < n_outer_cpy; ++i) {
    CUDA_CHECK(cudaMemcpyAsync(out               , in1, item_size * w1,  cudaMemcpyDeviceToDevice, stream1));
    CUDA_CHECK(cudaMemcpyAsync(out + w1          , in2, item_size * w2,  cudaMemcpyDeviceToDevice, stream1));
    CUDA_CHECK(cudaMemcpyAsync(out + w1 + w2     , in3, item_size * w3,  cudaMemcpyDeviceToDevice, stream1));
    CUDA_CHECK(cudaMemcpyAsync(out + w1 + w2 + w3, in4, item_size * w4,  cudaMemcpyDeviceToDevice, stream1));
    out += w1 + w2 + w3 + w4;
    in1 += w1;
    in2 += w2;
    in3 += w3;
    in4 += w4;
  }
  const T* in1_krn = input_1 + n_outer_cpy * w1;
  const T* in2_krn = input_2 + n_outer_cpy * w2;
  const T* in3_krn = input_3 + n_outer_cpy * w3;
  const T* in4_krn = input_4 + n_outer_cpy * w4;
  T* out_krn = output + n_outer_cpy * (w1 + w2 + w3 + w4);

  ConcatKernel(n_outer_krn * (w1 + w2 + w3 + w4), w1, w2, w3, w4, in1_krn, in2_krn, in3_krn, in4_krn, out_krn, stream2);

  return;
}