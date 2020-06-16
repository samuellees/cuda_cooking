#pragma once

#include "cuda_runtime.h"
#include "gemm.h"

#ifndef CUDA_CHECK
#define CUDA_CHECK(code)                                                  \
  {                                                                       \
    if ((code) != cudaSuccess) {                                          \
      fprintf(stderr, "CUDA error in file: %s, line: %d, %s\n", __FILE__, \
              __LINE__, cudaGetErrorString((code)));                      \
      exit((code));                                                       \
    }                                                                     \
  }
#endif

__global__ void matmul_kernel_naive(const Matrix A, const Matrix B, Matrix C) {
  const int row_C = blockIdx.y * blockDim.y + threadIdx.y;
  const int col_C = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_C >= C.n_row || col_C >= C.n_col) {
    return;
  }
  int offset_A = row_C * A.n_col; // =>  row_A * A.n_col + 0
  int offset_B = col_C;           // =>      0 * B.n_col + col_B
  int offset_C = row_C * C.n_col + col_C;
  C.data[offset_C] = 0;
  for (int k = 0; k < A.n_col; ++k) {
    C.data[offset_C] += A.data[offset_A] * B.data[offset_B];
    offset_A += 1;
    offset_B += B.n_col;
  }
}