#pragma once

#include <cudnn.h>
#include "cuda_runtime.h"
#include "conv.h"
#include "stdio.h"

#define CUDA_CHECK(code)                                                  \
  {                                                                       \
    cudaError_t status = (code);                                         \
    if ((status) != cudaSuccess) {                                          \
      fprintf(stderr, "CUDA error in file: %s, line: %d, %s\n", __FILE__, \
              __LINE__, cudaGetErrorString((status)));                      \
      exit((status));                                                       \
    }                                                                     \
  }

#define CUDNN_CHECK(code)                                            \
  {                                                                       \
    cudnnStatus_t condition = (code);                                         \
    if ((condition) != CUDNN_STATUS_SUCCESS) {                            \
      fprintf(stderr, "CUDNN error in file: %s, line: %d, %s\n", __FILE__, \
              __LINE__, cudnnGetErrorString((condition)));                \
      exit((condition));                                                  \
    }                                                                     \
  }

#define WPTX 4  // workloads per thread at X direction
#define WPTY 8  // workloads per thread at y direction
#define NTX (BLOCK_SIZE_L/WPTX) // works at x direction
#define NTY (BLOCK_SIZE_L/WPTY) // works at y direction
#define tidx (threadIdx.x)
#define tidy (threadIdx.y)
__global__ void kernel_conv_im2col_align(const int M, const int N, const int K, 
                        const float4 * kernel, const float4 * input, float *output) 
{
  // n_blocks per matrix in y direction 
  const int n_blocks = (M - 1) / BLOCK_SIZE_L + 1; 
  const float4* A = kernel;
  const float4* B = input + N * K / 4 * (blockIdx.y / n_blocks);
  float * C = output + M * N * (blockIdx.y / n_blocks);

  // each thread compute 32 elements of C, row_C means the #row of first element.
  const int row_C = (blockIdx.y % n_blocks)  * BLOCK_SIZE_L + tidy;
  const int col_C = blockIdx.x * BLOCK_SIZE_L + tidx;

  __shared__ float subA[BLOCK_SIZE_L][BLOCK_SIZE_L];
  __shared__ float subB[BLOCK_SIZE_L][BLOCK_SIZE_L];
  
  float regA[WPTY];
  float regB[WPTX];
  float accums[WPTY][WPTX];
  #pragma unroll
  for (int r = 0; r < WPTY; ++r) {
    for (int c = 0; c < WPTX; ++c) {
      accums[r][c] = 0;
    }
  }

  for (int bn = 0; bn < K; bn+=BLOCK_SIZE_L) {
    #pragma unroll
    for (int r = 0; r < WPTY; r++) {
      #pragma unroll
      for (int c = 0; c < WPTX/4; ++c) {
        float4 vecA = __ldg(&A[(row_C+r*NTY)*K/4 + (bn+(c*NTX+tidx)*4)/4]);
        float4 vecB = __ldg(&B[(bn+r*NTY+tidy)*N/4 + (blockIdx.x*BLOCK_SIZE_L+(c*NTX+tidx)*4)/4]);
        subA[r*NTY+tidy][(c*NTX+tidx)*4] = vecA.x;
        subA[r*NTY+tidy][(c*NTX+tidx)*4+1] = vecA.y;
        subA[r*NTY+tidy][(c*NTX+tidx)*4+2] = vecA.z;
        subA[r*NTY+tidy][(c*NTX+tidx)*4+3] = vecA.w;
        subB[r*NTY+tidy][(c*NTX+tidx)*4] = vecB.x;
        subB[r*NTY+tidy][(c*NTX+tidx)*4+1] = vecB.y;
        subB[r*NTY+tidy][(c*NTX+tidx)*4+2] = vecB.z;
        subB[r*NTY+tidy][(c*NTX+tidx)*4+3] = vecB.w;
      }
    }
    __syncthreads();
    // traversal on K dimension
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE_L; ++i) {
      // load into register
      #pragma unroll
      for (int r = 0; r < WPTY; r++) {
        regA[r] = subA[r*NTY+tidy][i];
      }
      #pragma unroll
      for (int c = 0; c < WPTX; c++) {
        regB[c] = subB[i][c*NTX+tidx];
      }
      // do computation
      #pragma unroll
      for (int r = 0; r < WPTY; ++r) {
        #pragma unroll
        for (int c = 0; c < WPTX; ++c) {
          accums[r][c] += regA[r] * regB[c];
        }
      }
    }
    __syncthreads();
  }
  // write value into global memory
  #pragma unroll
  for (int r = 0; r < WPTY; ++r) {
    #pragma unroll
    for (int c = 0; c < WPTX; ++c) {
      C[(row_C+r*NTY) * N + col_C+c*NTX] = accums[r][c];
    }
  }
}




// shared_4workloads
__global__ void kernel_shared_4w(const int M, const int N, const int K, const float * A, const float * B, float *C) {
  // each thread compute 4 elements of C, row_C means the #row of first element.
  const int row_C = blockIdx.y * BLOCK_SIZE * 2  + threadIdx.y;
  const int col_C = blockIdx.x * BLOCK_SIZE * 2  + threadIdx.x;

  __shared__ float subA[BLOCK_SIZE * 2][BLOCK_SIZE * 2];
  __shared__ float subB[BLOCK_SIZE * 2][BLOCK_SIZE * 2];
  
  float accums[2][2] = {{0, 0}, {0, 0}};
  for (int bn = 0; bn < K; bn += 2*BLOCK_SIZE) {
    // A00
    if (row_C < M && bn+threadIdx.x < K) {
      subA[threadIdx.y][threadIdx.x] = 
            A[(row_C)*K + (bn+threadIdx.x)];
    } else {
      subA[threadIdx.y][threadIdx.x] = 0;
    }
    // A01
    if (row_C < M && bn+threadIdx.x+BLOCK_SIZE < K) {
      subA[threadIdx.y][threadIdx.x+BLOCK_SIZE] = 
            A[(row_C)*K + (bn+threadIdx.x+BLOCK_SIZE)];
    } else {
      subA[threadIdx.y][threadIdx.x+BLOCK_SIZE] = 0;
    }
    // A10
    if (row_C+BLOCK_SIZE < M && bn+threadIdx.x < K) {
      subA[threadIdx.y+BLOCK_SIZE][threadIdx.x] = 
            A[(row_C+BLOCK_SIZE)*K + (bn+threadIdx.x)];
    } else {
      subA[threadIdx.y+BLOCK_SIZE][threadIdx.x] = 0;
    }
    // A11
    if (row_C+BLOCK_SIZE < M && bn+threadIdx.x+BLOCK_SIZE < K) {
      subA[threadIdx.y+BLOCK_SIZE][threadIdx.x+BLOCK_SIZE] = 
            A[(row_C+BLOCK_SIZE)*K + (bn+threadIdx.x+BLOCK_SIZE)];
    } else {
      subA[threadIdx.y+BLOCK_SIZE][threadIdx.x+BLOCK_SIZE] = 0;
    }
    // B00
    if (bn+threadIdx.y < K && col_C < N) {
      subB[threadIdx.y][threadIdx.x] = 
            B[(bn+threadIdx.y)*N + col_C];
    } else {
      subB[threadIdx.y][threadIdx.x] = 0;
    }
    // B01
    if (bn+threadIdx.y < K && col_C+BLOCK_SIZE < N) {
      subB[threadIdx.y][threadIdx.x+BLOCK_SIZE] = 
            B[(bn+threadIdx.y)*N + (col_C+BLOCK_SIZE)];
    } else {
      subB[threadIdx.y][threadIdx.x+BLOCK_SIZE] = 0;
    }
    // B10
    if (bn+threadIdx.y+BLOCK_SIZE < K && col_C < N) {
      subB[threadIdx.y+BLOCK_SIZE][threadIdx.x] = 
            B[(bn+threadIdx.y+BLOCK_SIZE)*N + (col_C)];
    } else {
      subB[threadIdx.y+BLOCK_SIZE][threadIdx.x] = 0;
    }
    // B11
    if (bn+threadIdx.y+BLOCK_SIZE < K && col_C+BLOCK_SIZE < N) {
      subB[threadIdx.y+BLOCK_SIZE][threadIdx.x+BLOCK_SIZE] = 
            B[(bn+threadIdx.y+BLOCK_SIZE)*N + (col_C+BLOCK_SIZE)];
    } else {
      subB[threadIdx.y+BLOCK_SIZE][threadIdx.x+BLOCK_SIZE] = 0;
    }
    __syncthreads();
    for (int i = 0; i < 2*BLOCK_SIZE; ++i) {
      float subA_y0i = subA[threadIdx.y][i];
      float subA_y1i = subA[threadIdx.y+BLOCK_SIZE][i];
      float subB_ix0 = subB[i][threadIdx.x];
      float subB_ix1 = subB[i][threadIdx.x+BLOCK_SIZE];
      accums[0][0] += subA_y0i * subB_ix0; 
      accums[0][1] += subA_y0i * subB_ix1; 
      accums[1][0] += subA_y1i * subB_ix0; 
      accums[1][1] += subA_y1i * subB_ix1; 
    }
    __syncthreads();
  }
  // write value into global memory
  // C00
  if (row_C < M && col_C < N)
    C[row_C * N + col_C] = accums[0][0];
  // C01
  if (row_C < M && col_C+BLOCK_SIZE < N)
    C[row_C * N + col_C+BLOCK_SIZE] = accums[0][1];
  // C10
  if (row_C+BLOCK_SIZE < M && col_C < N)
    C[(row_C+BLOCK_SIZE) * N + col_C] = accums[1][0];
  // C11
  if (row_C+BLOCK_SIZE < M && col_C+BLOCK_SIZE < N)
    C[(row_C+BLOCK_SIZE) * N + col_C+BLOCK_SIZE] = accums[1][1];
}