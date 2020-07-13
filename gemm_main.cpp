#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include "gemm.h"

template<typename scalar_t>
void print_data(const scalar_t * data, int length) {
  for (int i = 0; i < length; ++i) {
      std::cout << data[i] << ", ";
  }
  std::cout << std::endl;
}

void print_matrix(Matrix mat) {
  for (int i = 0; i < mat.n_row; ++i) {
    for (int j = 0; j < mat.n_col; ++j) {
      printf("%.2f, ", mat.data[i * mat.n_col + j]);
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main() {
  const int MSIZE = 3180;
  Matrix A = {MSIZE, MSIZE, MSIZE, NULL};
  Matrix B = {MSIZE, MSIZE, MSIZE, NULL};
  Matrix C = {MSIZE, MSIZE, MSIZE, NULL};
  Matrix C_ref = {MSIZE, MSIZE, MSIZE, NULL};
  std::cout << "prepare data..." << std::endl;
  malloc_and_init(&A.data, A.n_row * A.n_col);
  malloc_and_init(&B.data, B.n_row * B.n_col);
  malloc_and_init(&C.data, C.n_row * C.n_col);
  malloc_and_init(&C_ref.data, C_ref.n_row * C_ref.n_col);
  std::cout << "computing..." << std::endl;
  std::vector<float> flops_info;
  gemm_ref(A, B, C_ref, flops_info);
  gemm(A, B, C, flops_info);

  std::cout << "check correctness..." << std::endl;
  bool error = false;
  #pragma unroll 64
  for (int i = 0; i < C.n_row * C.n_col; ++i) {
    error = error || (std::abs(C.data[i] - C_ref.data[i]) > 1e-5);
  }
  std::cout << "error: " << error << std::endl;

  free(A.data);
  free(B.data);
  free(C.data);
  free(C_ref.data);
  return 0;
}