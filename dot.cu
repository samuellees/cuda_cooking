#include "cuda_runtime.h"
#include "stdio.h"
#include "dot.h"

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
static const int blocksPerGrid = 256;

__global__ void dot_naive(const Vector A, const Vector B, Vector C) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int cid = threadIdx.x;
  if (tid > A.length) {
    return;
  }
  __shared__ float cache[threadsPerBlock];

  float temp = 0;
  for (int idx = tid; idx < A.length; idx += blockDim.x * gridDim.x) {
    temp += A.data[idx] * B.data[idx];
  }

  cache[cid] = temp;
  __syncthreads();

  int idx = blockDim.x / 2;
  while (idx > 0) {
    if (cid < idx) {
      cache[cid] += cache[cid + idx];
    }
    __syncthreads();
    idx /= 2;
  }

  if (cid == 0)
		C.data[blockIdx.x] = cache[0];

}


void dot(const Vector A, const Vector B, Vector C) {
  Vector d_A;
  Vector d_B;
  Vector d_C;
  d_A.length = A.length;
  d_B.length = B.length;
  d_C.length = C.length;
  int size_A = d_A.length;
  int size_B = d_B.length;
  int size_C = d_C.length;
  CUDA_CHECK(cudaMalloc(&d_A.data, size_A * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B.data, size_B * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C.data, size_C * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A.data, A.data, size_A * sizeof(float),  cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B.data, B.data, size_B * sizeof(float),  cudaMemcpyHostToDevice));
  // invoke kernel
  dim3 dims_block(threadsPerBlock);
  dim3 dims_grid(blocksPerGrid);

  cudaEvent_t start, mid, stop; 
  CUDA_CHECK(cudaEventCreate(&start)); 
  CUDA_CHECK(cudaEventCreate(&mid)); 
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, 0));
  dot_naive<<<dims_grid, dims_block>>>(d_A, d_B, d_C);
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
  CUDA_CHECK(cudaMemcpy(C.data, d_C.data, size_C * sizeof(float),  cudaMemcpyDeviceToHost));
  for (int i = 1; i < blocksPerGrid; ++i) {
    C.data[0] += C.data[i];
  }
  CUDA_CHECK(cudaFree(d_A.data));
  CUDA_CHECK(cudaFree(d_B.data));
  CUDA_CHECK(cudaFree(d_C.data));
  CUDA_CHECK(cudaEventDestroy(start)); 
  CUDA_CHECK(cudaEventDestroy(stop));
  return;
}