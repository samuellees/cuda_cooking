#pragma once
#include <random>
#include <iostream>
#include <vector>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#ifndef CEIL_DIV
#define CEIL_DIV(x, y) (((x) - 1) / (y) + 1)
#endif

typedef struct {
  int n_row;    // number of rows
  int n_col;    // number of columns
  int ld;       // leading dimensions
  float* data;  // pointer to data
} Matrix;

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

void gemm(const Matrix A, const Matrix B, Matrix C, std::vector<float>& flops_info);
void gemm_ref(const Matrix A, const Matrix B, Matrix C, std::vector<float>& flops_info);