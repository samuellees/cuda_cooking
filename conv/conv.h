#pragma once

#include <random>
#include <stdio.h>

static const int n_rounds = 10;
// static const int BATCH_SIZE = 32;
// static const int Ci = 64;
// static const int Hi = 128;
// static const int Wi = 128;
// static const int Co = 128;

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
    // (*data)[i] = 1;
  }
}

void conv1x1_test();

// void conv3x3(const int N, const int Ci, const int Hi, const int Wi, const float* input, 
//               const int Co, const int Kh, const int Kw, const float* kernel, 
//               const int Ho, const int Wo, float *output);

// void conv5x5(const int N, const int Ci, const int Hi, const int Wi, const float* input, 
//               const int Co, const int Kh, const int Kw, const float* kernel, 
//               const int Ho, const int Wo, float *output);


void convCuDNN(
  const int BATCH_SIZE, const int Ci, const int Hi, const int Wi, const float* input,
  const int Co, const int Hk, const int Wk, const float* kernel,
  const int Ho, const int Wo, float* output,
  float * time_ptr);

