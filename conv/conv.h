#pragma once

#include <random>
#include <stdio.h>

static const int n_rounds = 1;

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
  int seed = 3;
  std::default_random_engine gen(seed);
  std::normal_distribution<scalar_t> distribut(0, 1);
  *data = (scalar_t *)malloc(sizeof(scalar_t) * length);
  // const float tmp = length;
  for (int64_t i = 0; i < length; ++i) {
    (*data)[i] = distribut(gen);
    // (*data)[i] = (i*i - 100*i + 7) / tmp;
    // (*data)[i] = i;
  }
}

void conv_1x1_im2col_test();

void conv_NxN_im2col_with_batch_test();

void convCuDNN(
  const int BATCH_SIZE, const int Ci, const int Hi, const int Wi, const float* input,
  const int pad_h, const int pad_w, 
  const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w,
  const int Co, const int Hk, const int Wk, const float* kernel,
  const int Ho, const int Wo, float* output,
  float * time_ptr);

