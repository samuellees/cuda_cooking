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

__global__ void gemv_naive(const Matrix A, const Vector X, Vector Y) {
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


void gemv(const Matrix A, const Vector X, Vector Y) {
  Matrix d_A;
  Vector d_X;
  Vector d_Y;
  d_A.n_col = A.n_col;
  d_A.n_row = A.n_row;
  d_X.length = X.length;
  d_Y.length = Y.length;
  int size_A = d_A.n_col * d_A.n_row;
  int size_X = d_X.length;
  int size_Y = d_Y.length;
  CUDA_CHECK(cudaMalloc(&d_A.data, size_A * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_X.data, size_X * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_Y.data, size_Y * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A.data, A.data, size_A * sizeof(float),  cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_X.data, X.data, size_X * sizeof(float),  cudaMemcpyHostToDevice));
  // invoke kernel
  dim3 dims_block(threadsPerBlock);
  dim3 dims_grid(blocksPerGrid);

  cudaEvent_t start, mid, stop; 
  CUDA_CHECK(cudaEventCreate(&start)); 
  CUDA_CHECK(cudaEventCreate(&mid)); 
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, 0));
  gemv_naive<<<dims_grid, dims_block>>>(d_A, d_X, d_Y);
  CUDA_CHECK(cudaEventRecord(mid, 0)); 
  // matmul_kernel_shared_memory<<<dims_grid, dims_block>>>(d_A, d_B, d_C);
  CUDA_CHECK(cudaEventRecord(stop, 0)); 
  CUDA_CHECK(cudaEventSynchronize(stop)); 
  float elapsedTime_1, elapsedTime_2;
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_1, start, mid));
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_2, mid, stop));
  printf("Time of naive: %f\n", elapsedTime_1);
  // printf("Time of shared_memory: %f\n", elapsedTime_2);
  
  // copy data from device to host
  CUDA_CHECK(cudaMemcpy(Y.data, d_Y.data, size_Y * sizeof(float),  cudaMemcpyDeviceToHost));
  // for (int i = 1; i < blocksPerGrid; ++i) {
  //   C.data[0] += C.data[i];
  // }
  CUDA_CHECK(cudaFree(d_A.data));
  CUDA_CHECK(cudaFree(d_X.data));
  CUDA_CHECK(cudaFree(d_Y.data));
  CUDA_CHECK(cudaEventDestroy(start)); 
  CUDA_CHECK(cudaEventDestroy(stop));
  return;
}