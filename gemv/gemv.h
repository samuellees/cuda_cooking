#pragma once
#include <random>
#include <iostream>

struct Matrix {
  int n_row;
  int n_col;
  float* data;  // pointer to data
};

struct Vector {
  int length;       
  float* data;  // pointer to data
};


template <typename scalar_t>
inline void malloc_and_init(scalar_t** data, int length) {
  int seed = 3;
  std::default_random_engine gen(seed);
  std::normal_distribution<scalar_t> distribut(0, 1);
  *data = (scalar_t *)malloc(sizeof(scalar_t) * length);
  for (int i = 0; i < length; ++i) {
    (*data)[i] = distribut(gen);
  }
}

inline Matrix transpose(Matrix X) {
  Matrix X_trans;
  X_trans.n_col = X.n_row;
  X_trans.n_row = X.n_col;
  malloc_and_init(&X_trans.data, X_trans.n_row * X_trans.n_col);
  for (int i = 0; i < X.n_row; ++i) {
    for (int j = 0; j < X.n_col; ++j) {
      X_trans.data[j * X_trans.n_col + i] = X.data[i * X.n_col + j];
    }
  }
  return X_trans;
}

void gemv(const Matrix X, const Vector Y, Vector Z);

// reference implementation on CPU
void gemv_ref(const Matrix X, const Vector Y, Vector Z);