#include "cuda_runtime.h"
#include "gemm.h"

#define BLOCK_SIZE 4

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

__global__ void matmul_kernel(const Matrix A, const Matrix B, Matrix C) {
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


void MatMul(const Matrix A, const Matrix B, Matrix C) {
  // copy A, B and C to device
  Matrix d_A;
  Matrix d_B;
  Matrix d_C;
  d_A.n_col = A.n_col;
  d_B.n_col = B.n_col;
  d_C.n_col = C.n_col;
  d_A.n_row = A.n_row;
  d_B.n_row = B.n_row;
  d_C.n_row = C.n_row;
  int size_A = d_A.n_col * d_A.n_row;
  int size_B = d_B.n_col * d_B.n_row;
  int size_C = d_C.n_col * d_C.n_row;
  CUDA_CHECK(cudaMalloc(&d_A.data, size_A * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B.data, size_B * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C.data, size_C * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A.data, A.data, size_A * sizeof(float),  cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B.data, B.data, size_B * sizeof(float),  cudaMemcpyHostToDevice));
  // invoke kernel
  dim3 dims_block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dims_grid(C.n_col / dims_block.x + 1, C.n_row / dims_block.y + 1);
  matmul_kernel<<<dims_grid, dims_block>>>(d_A, d_B, d_C);
  // copy data from device to host
  CUDA_CHECK(cudaMemcpy(C.data, d_C.data, size_C * sizeof(float),  cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_A.data));
  CUDA_CHECK(cudaFree(d_B.data));
  CUDA_CHECK(cudaFree(d_C.data));
  return;
}