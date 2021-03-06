#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include "gemv.h"
#include "assert.h"

template<typename scalar_t>
void print_data(const scalar_t * data, int length) {
  for (int i = 0; i < length; ++i) {
      std::cout << data[i] << ", ";
  }
  std::cout << std::endl;
}

int main() {
  Matrix A = {16384, 16384, NULL};
  Vector X = {16384, NULL};
  Vector Y = {16384, NULL};
  Vector Y_ref = {16384, NULL};
  std::cout << "prepare data..." << std::endl;
  malloc_and_init(&A.data, A.n_col * A.n_row);
  malloc_and_init(&X.data, X.length);
  malloc_and_init(&Y.data, Y.length);
  malloc_and_init(&Y_ref.data, Y_ref.length);
  std::cout << "computing..." << std::endl;
  gemv(A, X, Y);
  gemv_ref(A, X, Y_ref);

  std::cout << "check correctness..." << std::endl;
  bool error = false;
  #pragma unroll 64
  for (int i = 0; i < A.n_row; ++i) {
    error = error || (std::abs(Y.data[i] - Y_ref.data[i]) > 1);
  }
  std::cout << "error: " << error << std::endl;

  // print_data(Y.data, 20);
  // print_data(Y_ref.data, 20);

  free(A.data);
  free(X.data);
  free(Y.data);
  free(Y_ref.data);
  return 0;
}