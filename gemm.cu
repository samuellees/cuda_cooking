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

__device__ Matrix get_submatrix(const Matrix mat, int block_row, int block_col) {
  Matrix submat;
  submat.n_row = BLOCK_SIZE;
  submat.n_col = BLOCK_SIZE;
  submat.ld = mat.ld;
  submat.data = mat.data + block_row * submat.n_row * submat.ld + block_col * submat.n_col;
  return submat;
}

__global__ void matmul_kernel(const Matrix A, const Matrix B, Matrix C) {
  const int row_sub = threadIdx.y;
  const int col_sub = threadIdx.x;
  const int block_row_C = blockIdx.y;
  const int block_col_C = blockIdx.x;
  const int row_C = block_row_C * blockDim.y + threadIdx.y;
  const int col_C = block_col_C * blockDim.x + threadIdx.x;
  if (row_C >= C.n_row || col_C >= C.n_col) {
    return;
  }

  // should be in square shape
  __shared__ float shared_data_subA[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float shared_data_subB[BLOCK_SIZE][BLOCK_SIZE];

  float value = 0;
  for (int k = 0; k < A.n_col / BLOCK_SIZE; ++k) {
    // get submatrix of A and B
    Matrix subA = get_submatrix(A, block_row_C, k);
    Matrix subB = get_submatrix(B, k, block_col_C);

    // each thread loads one element of subA and subB from 
    // global memory to shared_memory.
    shared_data_subA[row_sub][col_sub] = subA.data[row_sub * subA.ld + col_sub];
    shared_data_subB[row_sub][col_sub] = subB.data[row_sub * subB.ld + col_sub];
    __syncthreads();
    // accumulate product into value
    for (int i = 0; i < BLOCK_SIZE; ++i) {
      value += shared_data_subA[row_sub][i] * shared_data_subB[i][col_sub];
    }
    __syncthreads();
  }
  // write value into global memory
  C.data[row_C * C.ld + col_C] = value;
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
  d_A.ld = A.ld;
  d_B.ld = B.ld;
  d_C.ld = C.ld;
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