#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "assert.h"
#include <cmath>

#include <cublas_v2.h>
#include "cuda_runtime.h"

const char* cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  // To suppress compiler warning.
  return "Unrecognized cublas error string";
}

#define CUDA_CHECK(code)                                                  \
  {                                                                       \
    cudaError_t status = (code);                                         \
    if ((status) != cudaSuccess) {                                          \
      fprintf(stderr, "CUDA error in file: %s, line: %d, %s\n", __FILE__, \
              __LINE__, cudaGetErrorString((status)));                      \
      exit((status));                                                       \
    }                                                                     \
  }

#define CUBLAS_CHECK(code)                                           \
  {                                                                       \
    cublasStatus_t status = (code);                                         \
    if ((status) != CUBLAS_STATUS_SUCCESS) {                                          \
      fprintf(stderr, "cublas error in file: %s, line: %d, %s\n", __FILE__, \
              __LINE__, cublasGetErrorString((status)));                      \
      exit((status));                                                       \
    }                                                                     \
  }

template<typename scalar_t>
void print_data(const scalar_t * data, int length) {
  for (int i = 0; i < length; ++i) {
      std::cout << data[i] << ", ";
  }
  std::cout << std::endl;
}

int main() {
  int size = 6;
  float *input = new float[size];
  float alpha = 2.0;
  float *input_dev = nullptr;
  float *alpha_dev = nullptr;

  for (int i = 0; i < size; ++i) input[i] = 2.1;
  
  CUDA_CHECK(cudaMalloc(&input_dev, sizeof(float) * size));
  CUDA_CHECK(cudaMemcpy(input_dev, input, sizeof(float) * size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&alpha_dev, sizeof(float) * 1));
  CUDA_CHECK(cudaMemcpy(alpha_dev, &alpha, sizeof(float) * 1, cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  CUBLAS_CHECK(cublasScalEx(handle, size, &alpha, CUDA_R_32F, input_dev, CUDA_R_32F, 1, CUDA_R_32F));
  // CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  // CUBLAS_CHECK(cublasScalEx(handle, size, alpha_dev, CUDA_R_32F, input_dev, CUDA_R_32F, 1, CUDA_R_32F));
  
  
  CUDA_CHECK(cudaMemcpy(input, input_dev, sizeof(float) * size, cudaMemcpyDeviceToHost));

  print_data(input, size);

  
  CUDA_CHECK(cudaFree(input_dev));
  CUDA_CHECK(cudaFree(alpha_dev));
  delete []input;

  return 0;
}