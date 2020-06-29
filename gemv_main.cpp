#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include "gemv.h"
#include "assert.h"

void print_vector(Vector& vec) {
  for (int i = 0; i < vec.length; ++i) {
      std::cout << vec.data[i] << ", ";
  }
  std::cout << std::endl;
}

int main() {
  Matrix A = {65536, 1024, NULL};
  Vector X = {1024, NULL};
  Vector Y = {65536, NULL};
  Vector Y_ref = {65536, NULL};
  malloc_and_init(&A.data, A.n_col * A.n_row);
  malloc_and_init(&X.data, X.length);
  malloc_and_init(&Y.data, Y.length);
  malloc_and_init(&Y_ref.data, Y_ref.length);
  gemv(A, X, Y);
  gemv_ref(A, X, Y_ref);

  bool error = false;
  for (int i = 0; i < A.n_row; ++i) {
    error = error && (abs(Y.data[i] - Y_ref.data[i]) > 1);
  }
  std::cout << "error: " << error << std::endl;

  free(A.data);
  free(X.data);
  free(Y.data);
  free(Y_ref.data);
  return 0;
}