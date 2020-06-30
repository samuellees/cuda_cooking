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

static const int threadsPerBlock = 128;
static const int blocksPerGrid = 32;

__global__ void kernel_naive(const Matrix A, const Vector X, Vector Y) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid > A.n_row) {
    return;
  }
  for (int i = tid; i < A.n_row; i += threadsPerBlock * blocksPerGrid) {
    float temp = 0;
    for (int j = 0; j < A.n_col; j++) {
      temp += A.data[i * A.n_col + j] * X.data[j];
    }
    Y.data[i] = temp;
  }
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

__constant__ float DATA_CONSTANT[16384];
__global__ void kernel_constant(const Matrix A_trans, Vector Y) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid > A_trans.n_col) {
    return;
  }

  for (int i = tid; i < A_trans.n_col; i += threadsPerBlock * blocksPerGrid) {
    float temp = 0;
    for (int j = 0; j < A_trans.n_row; j++) {
      temp += A_trans.data[i + j * A_trans.n_col] * DATA_CONSTANT[j];
    }
    Y.data[i] = temp;
  }
}

__global__ void kernel_shared(const Matrix A_trans, const Vector X, Vector Y) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int cid = threadIdx.x;
  __shared__ float cache[threadsPerBlock];
  const int cache_last = A_trans.n_row - A_trans.n_row % threadsPerBlock;
  // i: current col
  for (int i = tid; i < A_trans.n_row; i += threadsPerBlock * blocksPerGrid) {
    float temp = 0;
    // j: current row for load cache
    for (int j = cid; j < cache_last; j += threadsPerBlock) {
      __syncthreads();
      cache[cid] = X.data[j];
      __syncthreads();
      int begin = j - j % threadsPerBlock;
      // k: current row for calculate
      for (int k = begin; k < begin + threadsPerBlock; k++) {
          temp += A_trans.data[k * A_trans.n_col + i] * cache[k - begin];
      }
    }
    __syncthreads();
    if (cache_last + cid < A_trans.n_row) {
      cache[cid] = X.data[cache_last + cid];
    }
    __syncthreads();
    for (int k = cache_last; k < A_trans.n_row; k++) {
      temp += A_trans.data[k * A_trans.n_col + i] * cache[k - cache_last];
    }
    Y.data[i] = temp;
  }
}

__global__ void kernel_shuffle(const Matrix A_trans, const Vector X, Vector Y) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int laneId = threadIdx.x % 32;
  const int cache_last = A_trans.n_row - A_trans.n_row % 32;
  // i: current col
  for (int i = tid; i < A_trans.n_row; i += threadsPerBlock * blocksPerGrid) {
    float temp = 0;
    // j: current row for load cache
    float shuffle_val = 0;
    for (int j = laneId; j < cache_last; j += 32) {
      shuffle_val = X.data[j];
      int cache_offset = j / 32 * 32;
      // k: current row for calculate
      for (int k = 0; k < 32; k++) {
        temp += A_trans.data[(cache_offset + k) * A_trans.n_col + i]
          * __shfl_sync(0xffffffff, shuffle_val, k, 32);
      }
    }
    if (cache_last + laneId < A_trans.n_row) {
      shuffle_val = X.data[cache_last + laneId];
    }
    for (int k = cache_last; k < A_trans.n_row; k++) {
      temp += A_trans.data[k * A_trans.n_col + i] 
        * __shfl_sync(0xffffffff, shuffle_val, k - cache_last, 32);
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
  CUDA_CHECK(cudaMemcpy(d_A_trans.data, A_trans.data, size_A*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyToSymbol(DATA_CONSTANT, X.data, size_X*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_X.data, X.data, size_X * sizeof(float),  cudaMemcpyHostToDevice));
  // invoke kernel
  dim3 dims_block(threadsPerBlock);
  dim3 dims_grid(blocksPerGrid);

  // warm up
  kernel_naive<<<dims_grid, dims_block>>>(d_A, d_X, d_Y);

  // naive
  float elapsedTime_naive;
  cudaEvent_t start_naive, stop_naive; 
  CUDA_CHECK(cudaEventCreate(&start_naive)); 
  CUDA_CHECK(cudaEventCreate(&stop_naive));
  CUDA_CHECK(cudaEventRecord(start_naive, 0));
  kernel_naive<<<dims_grid, dims_block>>>(d_A, d_X, d_Y);
  CUDA_CHECK(cudaEventRecord(stop_naive, 0)); 
  CUDA_CHECK(cudaEventSynchronize(stop_naive)); 
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_naive, start_naive, stop_naive));
  CUDA_CHECK(cudaEventDestroy(start_naive)); 
  CUDA_CHECK(cudaEventDestroy(stop_naive));
  printf("Time of naive: %f\n", elapsedTime_naive);
  // coalesce
  float elapsedTime_coalesce;
  cudaEvent_t start_coalesce, stop_coalesce; 
  CUDA_CHECK(cudaEventCreate(&start_coalesce)); 
  CUDA_CHECK(cudaEventCreate(&stop_coalesce));
  CUDA_CHECK(cudaEventRecord(start_coalesce, 0));
  kernel_coalesce<<<dims_grid, dims_block>>>(d_A_trans, d_X, d_Y);
  CUDA_CHECK(cudaEventRecord(stop_coalesce, 0)); 
  CUDA_CHECK(cudaEventSynchronize(stop_coalesce)); 
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_coalesce, start_coalesce, stop_coalesce));
  CUDA_CHECK(cudaEventDestroy(start_coalesce)); 
  CUDA_CHECK(cudaEventDestroy(stop_coalesce));
  printf("Time of coalesce: %f\n", elapsedTime_coalesce);
  // constant
  float elapsedTime_constant;
  cudaEvent_t start_constant, stop_constant; 
  CUDA_CHECK(cudaEventCreate(&start_constant)); 
  CUDA_CHECK(cudaEventCreate(&stop_constant));
  CUDA_CHECK(cudaEventRecord(start_constant, 0));
  kernel_constant<<<dims_grid, dims_block>>>(d_A_trans, d_Y);
  CUDA_CHECK(cudaEventRecord(stop_constant, 0)); 
  CUDA_CHECK(cudaEventSynchronize(stop_constant)); 
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_constant, start_constant, stop_constant));
  CUDA_CHECK(cudaEventDestroy(start_constant)); 
  CUDA_CHECK(cudaEventDestroy(stop_constant));
  printf("Time of constant: %f\n", elapsedTime_constant);
  // shared
  float elapsedTime_shared;
  cudaEvent_t start_shared, stop_shared; 
  CUDA_CHECK(cudaEventCreate(&start_shared)); 
  CUDA_CHECK(cudaEventCreate(&stop_shared));
  CUDA_CHECK(cudaEventRecord(start_shared, 0));
  kernel_shared<<<dims_grid, dims_block>>>(d_A_trans, d_X, d_Y);
  CUDA_CHECK(cudaEventRecord(stop_shared, 0)); 
  CUDA_CHECK(cudaEventSynchronize(stop_shared)); 
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_shared, start_shared, stop_shared));
  CUDA_CHECK(cudaEventDestroy(start_shared)); 
  CUDA_CHECK(cudaEventDestroy(stop_shared));
  printf("Time of shared: %f\n", elapsedTime_shared);
  // shuffle
  float elapsedTime_shuffle;
  cudaEvent_t start_shuffle, stop_shuffle; 
  CUDA_CHECK(cudaEventCreate(&start_shuffle)); 
  CUDA_CHECK(cudaEventCreate(&stop_shuffle));
  CUDA_CHECK(cudaEventRecord(start_shuffle, 0));
  kernel_shuffle<<<dims_grid, dims_block>>>(d_A_trans, d_X, d_Y);
  CUDA_CHECK(cudaEventRecord(stop_shuffle, 0)); 
  CUDA_CHECK(cudaEventSynchronize(stop_shuffle)); 
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_shuffle, start_shuffle, stop_shuffle));
  CUDA_CHECK(cudaEventDestroy(start_shuffle)); 
  CUDA_CHECK(cudaEventDestroy(stop_shuffle));
  printf("Time of shuffle: %f\n", elapsedTime_shuffle);
  
  // copy data from device to host
  CUDA_CHECK(cudaMemcpy(Y.data, d_Y.data, size_Y * sizeof(float),  cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_A.data));
  CUDA_CHECK(cudaFree(d_X.data));
  CUDA_CHECK(cudaFree(d_Y.data));
  free(A_trans.data);
  return;
}