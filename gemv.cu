#include "cuda_runtime.h"
#include "stdio.h"
#include "gemv.h"

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

static const int threadsPerBlock = 256;
static const int blocksPerGrid = 64;

__global__ void kernel_naive(const Matrix A, const Vector X, Vector Y) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // const int cid = threadIdx.x;
  if (tid > A.n_row) {
    return;
  }
  // __shared__ float cache[threadsPerBlock];
  for (int i = tid; i < A.n_row; i += threadsPerBlock * blocksPerGrid) {
    float temp = 0;
    for (int j = 0; j < A.n_col; j++) {
      temp += A.data[i * A.n_col + j] * X.data[j];
    }
    Y.data[i] = temp;
  }

  // cache[cid] = temp;
  // __syncthreads();

  // int idx = blockDim.x / 2;
  // while (idx > 0) {
  //   if (cid < idx) {
  //     cache[cid] += cache[cid + idx];
  //   }
  //   __syncthreads();
  //   idx /= 2;
  // }

  // if (cid == 0)
	// 	C.data[blockIdx.x] = cache[0];

}

__global__ void kernel_coalesce(const Matrix A_trans, const Vector X, Vector Y) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid > A_trans.n_col) {
    return;
  }

  for (int i = tid; i < A_trans.n_col; i += threadsPerBlock * blocksPerGrid) {
    float temp = 0;
    for (int j = 0; j < A_trans.n_row; j++) {
      temp += A_trans.data[i + j * A_trans.n_col] * X.data[j];
    }
    Y.data[i] = temp;
  }
}


void gemv(const Matrix A, const Vector X, Vector Y) {
  Matrix A_trans = transpose(A);
  Matrix d_A;
  Matrix d_A_trans;
  Vector d_X;
  Vector d_Y;
  d_A.n_col = A.n_col;
  d_A.n_row = A.n_row;
  d_A_trans.n_col = A_trans.n_col;
  d_A_trans.n_row = A_trans.n_row;
  d_X.length = X.length;
  d_Y.length = Y.length;
  int size_A = d_A.n_col * d_A.n_row;
  int size_X = d_X.length;
  int size_Y = d_Y.length;
  CUDA_CHECK(cudaMalloc(&d_A.data, size_A * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_A_trans.data, size_A * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_X.data, size_X * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_Y.data, size_Y * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A.data, A.data, size_A * sizeof(float),  cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_A_trans.data, A_trans.data, size_A * sizeof(float),  cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_X.data, X.data, size_X * sizeof(float),  cudaMemcpyHostToDevice));
  // invoke kernel
  dim3 dims_block(threadsPerBlock);
  dim3 dims_grid(blocksPerGrid);

  cudaEvent_t start_naive, stop_naive; 
  cudaEvent_t start_coalesce, stop_coalesce; 
  float elapsedTime_naive, elapsedTime_coalesce;
  CUDA_CHECK(cudaEventCreate(&start_naive)); 
  CUDA_CHECK(cudaEventCreate(&start_coalesce)); 
  CUDA_CHECK(cudaEventCreate(&stop_naive));
  CUDA_CHECK(cudaEventCreate(&stop_coalesce));
  // naive
  CUDA_CHECK(cudaEventRecord(start_naive, 0));
  kernel_naive<<<dims_grid, dims_block>>>(d_A, d_X, d_Y);
  CUDA_CHECK(cudaEventRecord(stop_naive, 0)); 
  CUDA_CHECK(cudaEventSynchronize(stop_naive)); 
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_naive, start_naive, stop_naive));
  CUDA_CHECK(cudaEventDestroy(start_naive)); 
  CUDA_CHECK(cudaEventDestroy(stop_naive));
  printf("Time of naive: %f\n", elapsedTime_naive);
  // coalesce
  CUDA_CHECK(cudaEventRecord(start_coalesce, 0));
  kernel_coalesce<<<dims_grid, dims_block>>>(d_A_trans, d_X, d_Y);
  CUDA_CHECK(cudaEventRecord(stop_coalesce, 0)); 
  CUDA_CHECK(cudaEventSynchronize(stop_coalesce)); 
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_coalesce, start_coalesce, stop_coalesce));
  CUDA_CHECK(cudaEventDestroy(start_coalesce)); 
  CUDA_CHECK(cudaEventDestroy(stop_coalesce));
  printf("Time of coalesce: %f\n", elapsedTime_coalesce);
  
  // copy data from device to host
  CUDA_CHECK(cudaMemcpy(Y.data, d_Y.data, size_Y * sizeof(float),  cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_A.data));
  CUDA_CHECK(cudaFree(d_X.data));
  CUDA_CHECK(cudaFree(d_Y.data));
  free(A_trans.data);
  return;
}