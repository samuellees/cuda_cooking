#include "gemv.h"

void gemv_ref(const Matrix A, const Vector X, Vector Y) {
  const int n_row = A.n_row;
  const int n_col = A.n_col;
  const float * a = A.data;
  const float * x = X.data;
  float * y = Y.data;

  for (int i = 0; i < n_row; ++i) {
    y[i] = 0;
    for (int j = 0; j < n_col; ++j) {
      y[i] += a[i * n_col + j] * x[j];
    }
  }
}