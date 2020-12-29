#include "cuda_runtime.h"
#include "gemm.h"
#include "gemm_kernels.cuh"
#include "coding.cuh"
#include "utils.cuh"
#include <cublas_v2.h>

#define WARMUP 1
// #define K1 1
// #define K2 1
// #define K3 1
// #define K4 1
// #define K5 1
// #define K6 1
#define K7 1
// #define K8 1

double getGFlops(double time_ms, int64_t m, int64_t n, int64_t k) {
  return 2 * m * k * n / (time_ms/1000) *1e-9;
}

const int n_rounds = 10;

void gemm(const Matrix A, const Matrix B, Matrix C, std::vector<float>& flops_info) {
  // copy A, B and C to device
  Matrix d_A = A;
  Matrix d_B = B;
  Matrix d_C = C;
  int64_t size_A = d_A.n_col * d_A.n_row;
  int64_t size_B = d_B.n_col * d_B.n_row;
  int64_t size_C = d_C.n_col * d_C.n_row;
  CUDA_CHECK(cudaMalloc(&d_A.data, size_A * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B.data, size_B * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C.data, size_C * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A.data, A.data, size_A * sizeof(float),  cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B.data, B.data, size_B * sizeof(float),  cudaMemcpyHostToDevice));
  
  // invoke kernel
  dim3 dims_block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dims_grid(CEIL_DIV(C.n_col, dims_block.x), CEIL_DIV(C.n_row, dims_block.y));

#ifdef WARMUP
  // warm up
  kernel_naive<<<dims_grid, dims_block>>>(
      d_A.n_row, d_B.n_col, d_A.n_col, d_A.data, d_B.data, d_C.data);
#endif

#ifdef K1
  // 1 naive
  float elapsedTime_naive;
  cudaEvent_t start_naive, stop_naive; 
  CUDA_CHECK(cudaEventCreate(&start_naive)); 
  CUDA_CHECK(cudaEventCreate(&stop_naive));
  CUDA_CHECK(cudaEventRecord(start_naive, 0));
  for (int i = 0; i < n_rounds; ++i) {
    kernel_naive<<<dims_grid, dims_block>>>(
        d_A.n_row, d_B.n_col, d_A.n_col, d_A.data, d_B.data, d_C.data);
  }
  CUDA_CHECK(cudaEventRecord(stop_naive, 0)); 
  CUDA_CHECK(cudaEventSynchronize(stop_naive)); 
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_naive, start_naive, stop_naive));
  CUDA_CHECK(cudaEventDestroy(start_naive)); 
  CUDA_CHECK(cudaEventDestroy(stop_naive));
  elapsedTime_naive = elapsedTime_naive / n_rounds;
  float flops_naive = getGFlops(elapsedTime_naive, A.n_row, B.n_col, A.n_col);
  printf("kernel %-20s: %8.2f ms, %8.2f GFlops, %6.2f%% of cublas.\n", 
            "naive", 
            elapsedTime_naive, flops_naive, 
            flops_naive / flops_info[0] * 100);
  flops_info.push_back(flops_naive);
#endif


#ifdef K2
  // 2 shared
  float elapsedTime_shared;
  cudaEvent_t start_shared, stop_shared; 
  CUDA_CHECK(cudaEventCreate(&start_shared)); 
  CUDA_CHECK(cudaEventCreate(&stop_shared));
  CUDA_CHECK(cudaEventRecord(start_shared, 0));
  for (int i = 0; i < n_rounds; ++i) {
    kernel_shared<<<dims_grid, dims_block>>>(
        d_A.n_row, d_B.n_col, d_A.n_col, d_A.data, d_B.data, d_C.data);
  }
  CUDA_CHECK(cudaEventRecord(stop_shared, 0)); 
  CUDA_CHECK(cudaEventSynchronize(stop_shared)); 
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_shared, start_shared, stop_shared));
  CUDA_CHECK(cudaEventDestroy(start_shared)); 
  CUDA_CHECK(cudaEventDestroy(stop_shared));
  elapsedTime_shared = elapsedTime_shared / n_rounds;
  float flops_shared = getGFlops(elapsedTime_shared, A.n_row, B.n_col, A.n_col);
  printf("kernel %-20s: %8.2f ms, %8.2f GFlops, %6.2f%% of cublas.\n", 
            "shared", 
            elapsedTime_shared, flops_shared, 
            flops_shared / flops_info[0] * 100);
  flops_info.push_back(flops_shared);
#endif


#ifdef K3
  // 3 shared_4workloads
  float elapsedTime_shared_4w;
  dim3 dims_block_shared_4w(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dims_grid_shared_4w(CEIL_DIV(C.n_col, BLOCK_SIZE*2), CEIL_DIV(C.n_row, BLOCK_SIZE*2));
  cudaEvent_t start_shared_4w, stop_shared_4w; 
  CUDA_CHECK(cudaEventCreate(&start_shared_4w)); 
  CUDA_CHECK(cudaEventCreate(&stop_shared_4w));
  CUDA_CHECK(cudaEventRecord(start_shared_4w, 0));
  for (int i = 0; i < n_rounds; ++i) {
    kernel_shared_4w<<<dims_grid_shared_4w, dims_block_shared_4w>>>(
        d_A.n_row, d_B.n_col, d_A.n_col, d_A.data, d_B.data, d_C.data);
  }
  CUDA_CHECK(cudaEventRecord(stop_shared_4w, 0)); 
  CUDA_CHECK(cudaEventSynchronize(stop_shared_4w)); 
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_shared_4w, start_shared_4w, stop_shared_4w));
  CUDA_CHECK(cudaEventDestroy(start_shared_4w)); 
  CUDA_CHECK(cudaEventDestroy(stop_shared_4w));
  elapsedTime_shared_4w = elapsedTime_shared_4w / n_rounds;
  float flops_shared_4w = getGFlops(elapsedTime_shared_4w, A.n_row, B.n_col, A.n_col);
  printf("kernel %-20s: %8.2f ms, %8.2f GFlops, %6.2f%% of cublas.\n", 
          "shared_4w", 
          elapsedTime_shared_4w, flops_shared_4w, 
          flops_shared_4w / flops_info[0] * 100);
  flops_info.push_back(flops_shared_4w);
#endif


  // padding vars
  float time_padding = 0;
  float time_unpadding = 0;
  float * d_padA = nullptr;
  float * d_padB = nullptr;
  float * d_padC = nullptr;
  int padM = 0;
  int padN = 0;
  int padK = 0;


#ifdef K4
  // 4 shared_4workloads_padding
  time_padding = 0;
  padM = CEIL_DIV(C.n_row, BLOCK_SIZE_L) * BLOCK_SIZE_L;
  padN = CEIL_DIV(C.n_col, BLOCK_SIZE_L) * BLOCK_SIZE_L;
  padK = CEIL_DIV(B.n_row, BLOCK_SIZE_L) * BLOCK_SIZE_L;
  CUDA_CHECK(cudaMalloc(&d_padA, padM * padK * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_padB, padK * padN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_padC, padM * padN * sizeof(float)));
  time_padding += padding(d_A.data, d_padA, A.n_row, A.n_col, padM, padK);
  time_padding += padding(d_B.data, d_padB, B.n_row, B.n_col, padK, padN);

  float elapsedTime_shared_4w_pad;
  dim3 dims_block_shared_4w_pad(BLOCK_SIZE_L/2, BLOCK_SIZE_L/2);
  dim3 dims_grid_shared_4w_pad(CEIL_DIV(C.n_col, BLOCK_SIZE_L), CEIL_DIV(C.n_row, BLOCK_SIZE_L));
  cudaEvent_t start_shared_4w_pad, stop_shared_4w_pad; 
  CUDA_CHECK(cudaEventCreate(&start_shared_4w_pad)); 
  CUDA_CHECK(cudaEventCreate(&stop_shared_4w_pad));
  CUDA_CHECK(cudaEventRecord(start_shared_4w_pad, 0));
  for (int i = 0; i < n_rounds; ++i) {
    kernel_shared_4w_pad<<<dims_grid_shared_4w_pad, dims_block_shared_4w_pad>>>(
      padM, padN, padK, d_padA, d_padB, d_padC);
  }
  CUDA_CHECK(cudaEventRecord(stop_shared_4w_pad, 0)); 
  CUDA_CHECK(cudaEventSynchronize(stop_shared_4w_pad)); 
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_shared_4w_pad, start_shared_4w_pad, stop_shared_4w_pad));
  CUDA_CHECK(cudaEventDestroy(start_shared_4w_pad)); 
  CUDA_CHECK(cudaEventDestroy(stop_shared_4w_pad));
  time_unpadding = unpadding(d_C.data, d_padC, d_C.n_row, d_C.n_col, padM, padN);
  CUDA_CHECK(cudaFree(d_padA));
  CUDA_CHECK(cudaFree(d_padB));
  CUDA_CHECK(cudaFree(d_padC));
  elapsedTime_shared_4w_pad = elapsedTime_shared_4w_pad / n_rounds + time_padding+time_unpadding;
  float flops_shared_4w_pad = getGFlops(elapsedTime_shared_4w_pad, A.n_row, B.n_col, A.n_col);
  printf("kernel %-20s: %8.2f ms, %8.2f GFlops, %6.2f%% of cublas.\n", 
        "shared_4w_pad", 
        elapsedTime_shared_4w_pad, flops_shared_4w_pad, 
        flops_shared_4w_pad / flops_info[0] * 100);
  flops_info.push_back(flops_shared_4w_pad);
#endif


#ifdef K5
  // 5 shared_8workloads_padding
  time_padding = 0;
  padM = CEIL_DIV(C.n_row, BLOCK_SIZE) * BLOCK_SIZE;
  padN = CEIL_DIV(C.n_col, BLOCK_SIZE) * BLOCK_SIZE;
  padK = CEIL_DIV(B.n_row, BLOCK_SIZE) * BLOCK_SIZE;
  CUDA_CHECK(cudaMalloc(&d_padA, padM * padK * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_padB, padK * padN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_padC, padM * padN * sizeof(float)));
  time_padding += padding(d_A.data, d_padA, A.n_row, A.n_col, padM, padK);
  time_padding += padding(d_B.data, d_padB, B.n_row, B.n_col, padK, padN);

  float elapsedTime_shared_8w_pad;
  dim3 dims_block_shared_8w_pad(BLOCK_SIZE, BLOCK_SIZE/WORK_PERTHREAD);
  dim3 dims_grid_shared_8w_pad(CEIL_DIV(C.n_col, BLOCK_SIZE), CEIL_DIV(C.n_row, BLOCK_SIZE));
  cudaEvent_t start_shared_8w_pad, stop_shared_8w_pad; 
  CUDA_CHECK(cudaEventCreate(&start_shared_8w_pad)); 
  CUDA_CHECK(cudaEventCreate(&stop_shared_8w_pad));
  CUDA_CHECK(cudaEventRecord(start_shared_8w_pad, 0));
  for (int i = 0; i < n_rounds; ++i) {
    kernel_shared_8w_pad<<<dims_grid_shared_8w_pad, dims_block_shared_8w_pad>>>(
      padM, padN, padK, d_padA, d_padB, d_padC);
  }
  CUDA_CHECK(cudaEventRecord(stop_shared_8w_pad, 0)); 
  CUDA_CHECK(cudaEventSynchronize(stop_shared_8w_pad)); 
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_shared_8w_pad, start_shared_8w_pad, stop_shared_8w_pad));
  CUDA_CHECK(cudaEventDestroy(start_shared_8w_pad)); 
  CUDA_CHECK(cudaEventDestroy(stop_shared_8w_pad));
  time_unpadding = unpadding(d_C.data, d_padC, d_C.n_row, d_C.n_col, padM, padN);
  CUDA_CHECK(cudaFree(d_padA));
  CUDA_CHECK(cudaFree(d_padB));
  CUDA_CHECK(cudaFree(d_padC));
  elapsedTime_shared_8w_pad = elapsedTime_shared_8w_pad / n_rounds + time_padding+time_unpadding;
  float flops_shared_8w_pad = getGFlops(elapsedTime_shared_8w_pad, A.n_row, B.n_col, A.n_col);
  printf("kernel %-20s: %8.2f ms, %8.2f GFlops, %6.2f%% of cublas.\n", 
        "shared_8w_pad", 
        elapsedTime_shared_8w_pad, flops_shared_8w_pad, 
        flops_shared_8w_pad / flops_info[0] * 100);
  flops_info.push_back(flops_shared_8w_pad);
#endif


#ifdef K6
  // 6 shared_32workloads2D_padding
  time_padding = 0;
  padM = CEIL_DIV(C.n_row, BLOCK_SIZE_L) * BLOCK_SIZE_L;
  padN = CEIL_DIV(C.n_col, BLOCK_SIZE_L) * BLOCK_SIZE_L;
  padK = CEIL_DIV(B.n_row, BLOCK_SIZE_L) * BLOCK_SIZE_L;
  CUDA_CHECK(cudaMalloc(&d_padA, padM * padK * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_padB, padK * padN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_padC, padM * padN * sizeof(float)));
  time_padding += padding(d_A.data, d_padA, A.n_row, A.n_col, padM, padK);
  time_padding += padding(d_B.data, d_padB, B.n_row, B.n_col, padK, padN);

  float elapsedTime_shared_32w2d_pad;
  dim3 dims_block_shared_32w2d_pad(NTX, NTY);
  dim3 dims_grid_shared_32w2d_pad(CEIL_DIV(C.n_col, BLOCK_SIZE_L), CEIL_DIV(C.n_row, BLOCK_SIZE_L));
  cudaEvent_t start_shared_32w2d_pad, stop_shared_32w2d_pad; 
  CUDA_CHECK(cudaEventCreate(&start_shared_32w2d_pad)); 
  CUDA_CHECK(cudaEventCreate(&stop_shared_32w2d_pad));
  CUDA_CHECK(cudaEventRecord(start_shared_32w2d_pad, 0));
  for (int i = 0; i < n_rounds; ++i) {
    kernel_shared_32w2d_pad<<<dims_grid_shared_32w2d_pad, dims_block_shared_32w2d_pad>>>(
      padM, padN, padK, d_padA, d_padB, d_padC);
  }
  CUDA_CHECK(cudaEventRecord(stop_shared_32w2d_pad, 0)); 
  CUDA_CHECK(cudaEventSynchronize(stop_shared_32w2d_pad)); 
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_shared_32w2d_pad, start_shared_32w2d_pad, stop_shared_32w2d_pad));
  CUDA_CHECK(cudaEventDestroy(start_shared_32w2d_pad)); 
  CUDA_CHECK(cudaEventDestroy(stop_shared_32w2d_pad));
  time_unpadding = unpadding(d_C.data, d_padC, d_C.n_row, d_C.n_col, padM, padN);
  CUDA_CHECK(cudaFree(d_padA));
  CUDA_CHECK(cudaFree(d_padB));
  CUDA_CHECK(cudaFree(d_padC));
  elapsedTime_shared_32w2d_pad = elapsedTime_shared_32w2d_pad / n_rounds + time_padding+time_unpadding;
  float flops_shared_32w2d_pad = getGFlops(elapsedTime_shared_32w2d_pad, A.n_row, B.n_col, A.n_col);
  printf("kernel %-20s: %8.2f ms, %8.2f GFlops, %6.2f%% of cublas.\n", 
      "shared_32w2d_pad", 
      elapsedTime_shared_32w2d_pad, flops_shared_32w2d_pad, 
      flops_shared_32w2d_pad / flops_info[0] * 100);
  flops_info.push_back(flops_shared_32w2d_pad);
#endif


#ifdef K7
  // 7 shared_32workloads2D_padding_vec
  time_padding = 0;
  padM = CEIL_DIV(C.n_row, BLOCK_SIZE_L) * BLOCK_SIZE_L;
  padN = CEIL_DIV(C.n_col, BLOCK_SIZE_L) * BLOCK_SIZE_L;
  padK = CEIL_DIV(B.n_row, BLOCK_SIZE_L) * BLOCK_SIZE_L;
  CUDA_CHECK(cudaMalloc(&d_padA, padM * padK * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_padB, padK * padN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_padC, padM * padN * sizeof(float)));
  time_padding += padding(d_A.data, d_padA, A.n_row, A.n_col, padM, padK);
  time_padding += padding(d_B.data, d_padB, B.n_row, B.n_col, padK, padN);
  float elapsedTime_shared_32w2d_pad_vec;
  dim3 dims_block_shared_32w2d_pad_vec(NTX, NTY);
  dim3 dims_grid_shared_32w2d_pad_vec(CEIL_DIV(C.n_col, BLOCK_SIZE_L), CEIL_DIV(C.n_row, BLOCK_SIZE_L));
  cudaEvent_t start_shared_32w2d_pad_vec, stop_shared_32w2d_pad_vec; 
    kernel_shared_32w2d_pad_vec<<<dims_grid_shared_32w2d_pad_vec, dims_block_shared_32w2d_pad_vec>>>(
      padM, padN, padK, (float4*)d_padA, (float4*)d_padB, d_padC);
  CUDA_CHECK(cudaEventCreate(&start_shared_32w2d_pad_vec)); 
  CUDA_CHECK(cudaEventCreate(&stop_shared_32w2d_pad_vec));
  CUDA_CHECK(cudaEventRecord(start_shared_32w2d_pad_vec, 0));
  for (int i = 0; i < n_rounds; ++i) {
    kernel_shared_32w2d_pad_vec<<<dims_grid_shared_32w2d_pad_vec, dims_block_shared_32w2d_pad_vec>>>(
      padM, padN, padK, (float4*)d_padA, (float4*)d_padB, d_padC);
  }
  CUDA_CHECK(cudaEventRecord(stop_shared_32w2d_pad_vec, 0)); 
  CUDA_CHECK(cudaEventSynchronize(stop_shared_32w2d_pad_vec)); 
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_shared_32w2d_pad_vec, start_shared_32w2d_pad_vec, stop_shared_32w2d_pad_vec));
  CUDA_CHECK(cudaEventDestroy(start_shared_32w2d_pad_vec)); 
  CUDA_CHECK(cudaEventDestroy(stop_shared_32w2d_pad_vec));
  time_unpadding = unpadding(d_C.data, d_padC, d_C.n_row, d_C.n_col, padM, padN);
  CUDA_CHECK(cudaFree(d_padA));
  CUDA_CHECK(cudaFree(d_padB));
  CUDA_CHECK(cudaFree(d_padC));
  elapsedTime_shared_32w2d_pad_vec = elapsedTime_shared_32w2d_pad_vec / n_rounds + time_padding+time_unpadding;
  // elapsedTime_shared_32w2d_pad_vec = elapsedTime_shared_32w2d_pad_vec / n_rounds;
  float flops_shared_32w2d_pad_vec = getGFlops(elapsedTime_shared_32w2d_pad_vec, A.n_row, B.n_col, A.n_col);
  printf("kernel %-20s: %8.2f ms, %8.2f GFlops, %6.2f%% of cublas.\n", 
    "shared_32w2d_pad_vec", 
    elapsedTime_shared_32w2d_pad_vec, flops_shared_32w2d_pad_vec, 
    flops_shared_32w2d_pad_vec / flops_info[0] * 100);
  flops_info.push_back(flops_shared_32w2d_pad_vec);
#endif


#ifdef K8
  // 8 shared_64workloads2D_padding_vec
  time_padding = 0;
  padM = CEIL_DIV(C.n_row, BLOCK_SIZE_L_MAX) * BLOCK_SIZE_L_MAX;
  padN = CEIL_DIV(C.n_col, BLOCK_SIZE_L_MAX) * BLOCK_SIZE_L_MAX;
  padK = CEIL_DIV(B.n_row, BLOCK_SIZE_L_MAX) * BLOCK_SIZE_L_MAX;
  CUDA_CHECK(cudaMalloc(&d_padA, padM * padK * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_padB, padK * padN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_padC, padM * padN * sizeof(float)));
  time_padding += padding(d_A.data, d_padA, A.n_row, A.n_col, padM, padK);
  time_padding += padding(d_B.data, d_padB, B.n_row, B.n_col, padK, padN);
  float elapsedTime_shared_64w2d_pad_vec;
  dim3 dims_block_shared_64w2d_pad_vec(NTX_MAX, NTY_MAX);
  dim3 dims_grid_shared_64w2d_pad_vec(CEIL_DIV(C.n_col, BLOCK_SIZE_L_MAX), CEIL_DIV(C.n_row, BLOCK_SIZE_L_MAX));
  cudaEvent_t start_shared_64w2d_pad_vec, stop_shared_64w2d_pad_vec; 
    kernel_shared_64w2d_pad_vec<<<dims_grid_shared_64w2d_pad_vec, dims_block_shared_64w2d_pad_vec>>>(
      padM, padN, padK, (float4*)d_padA, (float4*)d_padB, d_padC);
  CUDA_CHECK(cudaEventCreate(&start_shared_64w2d_pad_vec)); 
  CUDA_CHECK(cudaEventCreate(&stop_shared_64w2d_pad_vec));
  CUDA_CHECK(cudaEventRecord(start_shared_64w2d_pad_vec, 0));
  for (int i = 0; i < n_rounds; ++i) {
    kernel_shared_64w2d_pad_vec<<<dims_grid_shared_64w2d_pad_vec, dims_block_shared_64w2d_pad_vec>>>(
      padM, padN, padK, (float4*)d_padA, (float4*)d_padB, d_padC);
  }
  CUDA_CHECK(cudaEventRecord(stop_shared_64w2d_pad_vec, 0)); 
  CUDA_CHECK(cudaEventSynchronize(stop_shared_64w2d_pad_vec)); 
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_shared_64w2d_pad_vec, start_shared_64w2d_pad_vec, stop_shared_64w2d_pad_vec));
  CUDA_CHECK(cudaEventDestroy(start_shared_64w2d_pad_vec)); 
  CUDA_CHECK(cudaEventDestroy(stop_shared_64w2d_pad_vec));
  time_unpadding = unpadding(d_C.data, d_padC, d_C.n_row, d_C.n_col, padM, padN);
  CUDA_CHECK(cudaFree(d_padA));
  CUDA_CHECK(cudaFree(d_padB));
  CUDA_CHECK(cudaFree(d_padC));
  elapsedTime_shared_64w2d_pad_vec = elapsedTime_shared_64w2d_pad_vec / n_rounds + time_padding+time_unpadding;
  float flops_shared_64w2d_pad_vec = getGFlops(elapsedTime_shared_64w2d_pad_vec, A.n_row, B.n_col, A.n_col);
  printf("kernel %-20s: %8.2f ms, %8.2f GFlops, %6.2f%% of cublas.\n", 
    "shared_64w2d_pad_vec", 
    elapsedTime_shared_64w2d_pad_vec, flops_shared_64w2d_pad_vec, 
    flops_shared_64w2d_pad_vec / flops_info[0] * 100);
  flops_info.push_back(flops_shared_64w2d_pad_vec);
#endif


  // copy data from device to host
  CUDA_CHECK(cudaMemcpy(C.data, d_C.data, size_C * sizeof(float),  cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_A.data));
  CUDA_CHECK(cudaFree(d_B.data));
  CUDA_CHECK(cudaFree(d_C.data));
}


void gemm_ref(const Matrix A, const Matrix B, Matrix C, std::vector<float>& flops_info) {
  // copy A, B and C to device
  Matrix d_A = A;
  Matrix d_B = B;
  Matrix d_C = C;
  int64_t size_A = d_A.n_col * d_A.n_row;
  int64_t size_B = d_B.n_col * d_B.n_row;
  int64_t size_C = d_C.n_col * d_C.n_row;
  CUDA_CHECK(cudaMalloc(&d_A.data, size_A * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B.data, size_B * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C.data, size_C * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A.data, A.data, size_A * sizeof(float),  cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B.data, B.data, size_B * sizeof(float),  cudaMemcpyHostToDevice));

  float elapsedTime_cublas;
  cudaEvent_t start_cublas, stop_cublas; 
  CUDA_CHECK(cudaEventCreate(&start_cublas)); 
  CUDA_CHECK(cudaEventCreate(&stop_cublas));
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;
  const int M = A.n_row;
  const int K = A.n_col;
  const int N = B.n_col;
  const int m = N;
  const int k = K;
  const int n = M;
  const int lda = A.n_col;
  const int ldb = B.n_col;
  const int ldc = B.n_col;
  using scalar_t = float;
  const scalar_t *a = d_A.data;
  const scalar_t *b = d_B.data;
  scalar_t *c = d_C.data;
  scalar_t alpha = 1, beta = 0;
  // warm up
  cublasSgemm(handle, transb, transa, m, n, k,
                      &alpha, b, ldb, a, lda, &beta, c, ldc);
  CUDA_CHECK(cudaEventRecord(start_cublas, 0));
  for (int i = 0; i < n_rounds; ++i)
    cublasSgemm(handle, transb, transa, m, n, k,
                        &alpha, b, ldb, a, lda, &beta, c, ldc);
  CUDA_CHECK(cudaEventRecord(stop_cublas, 0)); 
  CUDA_CHECK(cudaEventSynchronize(stop_cublas)); 
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_cublas, start_cublas, stop_cublas));
  CUDA_CHECK(cudaEventDestroy(start_cublas)); 
  CUDA_CHECK(cudaEventDestroy(stop_cublas));
  elapsedTime_cublas = elapsedTime_cublas / n_rounds;
  float flops_cublas = getGFlops(elapsedTime_cublas, A.n_row, B.n_col, A.n_col);
  printf("kernel %-20s: %8.2f ms, %8.2f GFlops, %6.2f%% of cublas.\n", 
    "cublas", 
    elapsedTime_cublas, flops_cublas, 
    1 * 100.0);
  flops_info.push_back(flops_cublas);

  // copy data from device to host
  CUDA_CHECK(cudaMemcpy(C.data, d_C.data, size_C * sizeof(float),  cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_A.data));
  CUDA_CHECK(cudaFree(d_B.data));
  CUDA_CHECK(cudaMemset(d_C.data, 0, size_C * sizeof(float)));
  CUDA_CHECK(cudaFree(d_C.data));
  return;
}