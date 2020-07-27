#pragma once

#include <random>
#include <stdio.h>
#include <cublas_v2.h>

static const int64_t n_rounds = 1;

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#ifndef BLOCK_SIZE_L
#define BLOCK_SIZE_L 64
#endif

#ifndef CEIL_DIV
#define CEIL_DIV(x, y) (((x) - 1) / (y) + 1)
#endif

template <typename scalar_t>
inline void malloc_and_init(scalar_t** data, int64_t length) {
  int64_t seed = 3;
  std::default_random_engine gen(seed);
  std::normal_distribution<scalar_t> distribut(0, 1);
  *data = (scalar_t *)malloc(sizeof(scalar_t) * length);
  // const float tmp = length;
  for (int64_t i = 0; i < length; ++i) {
    // (*data)[i] = distribut(gen);
    // (*data)[i] = (i*i - 100*i + 7) / tmp;
    (*data)[i] = 1;
  }
}

void conv_1x1_im2col_test();

void conv_NxN_im2col_with_batch_test();

void convCuDNN(
  const int64_t BATCH_SIZE, const int64_t Ci, const int64_t Hi, const int64_t Wi, const float* input,
  const int64_t pad_h, const int64_t pad_w, 
  const int64_t stride_h, const int64_t stride_w,
  const int64_t dilation_h, const int64_t dilation_w,
  const int64_t Co, const int64_t Hk, const int64_t Wk, const float* kernel,
  const int64_t Ho, const int64_t Wo, float* output,
  float * time_ptr);


void gemmCublas(
  const int64_t M, const int64_t N, const int64_t K, 
  const float* A, const float* B, float* C,
  const int64_t batch_size, cublasHandle_t& handle
);