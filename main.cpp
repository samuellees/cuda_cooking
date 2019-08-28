#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "gemm.h"

void malloc_and_init_matrix(Matrix *mat, float value) {
  mat->data = (float *)malloc(sizeof(float) * mat->n_col * mat->n_row);
  for (int i = 0; i < mat->n_col * mat->n_row; ++i) {
    mat->data[i] = value;
  }
}

void print_matrix(Matrix mat) {
  for (int i = 0; i < mat.n_row; ++i) {
    for (int j = 0; j < mat.n_col; ++j) {
      std::cout << mat.data[i * mat.n_col + j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main() {
  Matrix A = {6, 5, NULL};
  Matrix B = {5, 3, NULL};
  Matrix C = {6, 3, NULL};
  malloc_and_init_matrix(&A, 1);
  malloc_and_init_matrix(&B, 2);
  malloc_and_init_matrix(&C, 0);
  MatMul(A, B, C);
  print_matrix(A);
  print_matrix(B);
  print_matrix(C);
  free(A.data);
  free(B.data);
  free(C.data);
  return 0;
}