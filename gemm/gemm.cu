#include "cuda_runtime.h"
#include "gemm.h"
#include "kernel_naive.cuh"
#include "kernel_shared_memory.cuh"

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

  cudaEvent_t start, mid, stop; 
  CUDA_CHECK(cudaEventCreate(&start)); 
  CUDA_CHECK(cudaEventCreate(&mid)); 
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, 0));
  matmul_kernel_naive<<<dims_grid, dims_block>>>(d_A, d_B, d_C);
  CUDA_CHECK(cudaEventRecord(mid, 0)); 
  matmul_kernel_shared_memory<<<dims_grid, dims_block>>>(d_A, d_B, d_C);
  CUDA_CHECK(cudaEventRecord(stop, 0)); 
  CUDA_CHECK(cudaEventSynchronize(stop)); 
  float elapsedTime_1, elapsedTime_2;
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_1, start, mid));
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_2, mid, stop));
  printf("Time of naive: %f\n", elapsedTime_1);
  printf("Time of shared_memory: %f\n", elapsedTime_2);
  // copy data from device to host
  CUDA_CHECK(cudaMemcpy(C.data, d_C.data, size_C * sizeof(float),  cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_A.data));
  CUDA_CHECK(cudaFree(d_B.data));
  CUDA_CHECK(cudaFree(d_C.data));
  CUDA_CHECK(cudaEventDestroy(start)); 
  CUDA_CHECK(cudaEventDestroy(stop));
  return;
}