#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>

using namespace std;

__global__ void reduce_kernel(float* array, float* result) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int mask = 0xffffffff;

  float val = array[tid];
  // val += __shfl_down_sync(mask, val, 1);
  val += __shfl_sync(mask, val, 1);
  // for (int offset = 16; offset >= 1; offset /= 2) {
    // val += __shfl_up_sync(mask, val, offset);
  // }
  printf("tid = %d, val = %f.\n", tid, val);
  array[tid] = val;

}

template <int block_size>
__global__ void reduce_kernel(float* array, float* result, int size) {
  const int tid = threadIdx.x;
  __shared__ float sdata[1024];
  sdata[tid] = 0;

  for (int i = blockIdx.x * block_size + tid; i < size; i += block_size * gridDim.x) {
    sdata[tid] += array[i];
  }
  __synchronize();

  const unsigned int mask = 0xffffffff;
  float val = sdata[tid];
  for (int offset = 16; offset >= 1; offset /= 2) {
    val += __shfl_down_sync(mask, val, offset);
  }
  if (tid % 32 == 0) {
    sdata[tid / 32] = val;
  }
  __synchronize();

  if (tid < 32) {
    val = sdata[tid];
    for (int offset = 16; offset >= 1; offset /= 2) {
      val += __shfl_down_sync(mask, val, offset);
    }
    if (tid == 0) {
      result[blockIdx.x] = val;
    }
  }
}

void reduce_test() {
  const int array_length = 32;
  float* array  = new float[array_length];
  float* d_array = nullptr;
  for (int i = 0; i < array_length; ++i) {
    array[i] = i;
    // array[i] = 1;
  }
  cudaMalloc(&d_array, sizeof(float) * array_length);
  cudaMemcpy(d_array, array, sizeof(float) * array_length, cudaMemcpyHostToDevice);

  float* result = nullptr;
  reduce_kernel<<<1, 32>>>(d_array, result);
  cudaDeviceSynchronize();
}