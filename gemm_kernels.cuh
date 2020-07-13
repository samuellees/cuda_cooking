#pragma once

#include "cuda_runtime.h"
#include "gemm.h"
#include "stdio.h"

// A(M*K) B(K*N) C(M*N)
#define KERNEL_PARAMS_LIST \
  const int M, const int N, const int K, const float * A, const float * B, float *C

#define CUDA_CHECK(code)                                                  \
  {                                                                       \
    if ((code) != cudaSuccess) {                                          \
      fprintf(stderr, "CUDA error in file: %s, line: %d, %s\n", __FILE__, \
              __LINE__, cudaGetErrorString((code)));                      \
      exit((code));                                                       \
    }                                                                     \
  }

// naive
__global__ void kernel_naive(KERNEL_PARAMS_LIST) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) {
    return;
  }
  const int offset_C = row * N + col;
  C[offset_C] = 0;
  for (int i = 0; i < K; ++i) {
    C[offset_C] += A[row * K + i] * B[i * N + col];
  }
}

// shared
__global__ void kernel_shared(KERNEL_PARAMS_LIST) {
  const int row_C = blockIdx.y * blockDim.y + threadIdx.y;
  const int col_C = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float subA[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float subB[BLOCK_SIZE][BLOCK_SIZE];

  float accum = 0;
  for (int bn = 0; bn < K; bn += BLOCK_SIZE) {
    if (row_C < M && bn+threadIdx.x < K) {
      subA[threadIdx.y][threadIdx.x] = 
            A[(row_C)*K + (bn+threadIdx.x)];
    } else {
      subA[threadIdx.y][threadIdx.x] = 0;
    }
    if (bn+threadIdx.y < K && col_C < N) {
      subB[threadIdx.y][threadIdx.x] = 
            B[(bn+threadIdx.y)*N + (col_C)];
    } else {
      subB[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();
    for (int i = 0; i < BLOCK_SIZE; ++i) {
      accum += subA[threadIdx.y][i] * subB[i][threadIdx.x]; 
    }
    __syncthreads();
  }
  // write value into global memory
  if (row_C < M && col_C < N)
    C[row_C * N + col_C] = accum;
}

// shared_4workloads
__global__ void kernel_shared_4w(KERNEL_PARAMS_LIST) {
  // each thread compute 4 elements of C, row_C means the #row of first element.
  const int row_C = blockIdx.y * blockDim.y + threadIdx.y;
  const int col_C = blockIdx.x * blockDim.x + threadIdx.x;

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

__global__ void kernel_padding(const float* A, float* padA,
                               int64_t M, int64_t N,
                               int64_t padM, int64_t padN) {
  const int row_A = blockIdx.y * blockDim.y + threadIdx.y;
  const int col_A = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_A < M && col_A < N) {
    padA[row_A * padN + col_A] = A[row_A * N + col_A];
  } else {
    padA[row_A * padN + col_A] = 0;
  }
}

__global__ void kernel_unpadding(float* A, const float* padA,
                                int64_t M, int64_t N,
                                int64_t padM, int64_t padN) {
  const int row_A = blockIdx.y * blockDim.y + threadIdx.y;
  const int col_A = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_A < M && col_A < N) {
    A[row_A * N + col_A] = padA[row_A * padN + col_A];
  }
}

// shared_4workloads_padding
#define BLOCK_SIZE_L 64   // large BLOCK
__global__ void kernel_shared_4w_pad(KERNEL_PARAMS_LIST) {
  // each thread compute 4 elements of C, row_C means the #row of first element.
  const int row_C = blockIdx.y * blockDim.y + threadIdx.y;
  const int col_C = blockIdx.x * blockDim.x + threadIdx.x;
  const int block_size_half = BLOCK_SIZE_L / 2;

  __shared__ float subA[BLOCK_SIZE_L][BLOCK_SIZE_L];
  __shared__ float subB[BLOCK_SIZE_L][BLOCK_SIZE_L];
  
  float accums[2][2] = {{0, 0}, {0, 0}};
  for (int bn = 0; bn < K; bn += BLOCK_SIZE_L) {
    // A00
    subA[threadIdx.y][threadIdx.x] = A[(row_C)*K + (bn+threadIdx.x)];
    // A01
    subA[threadIdx.y][threadIdx.x+block_size_half] = A[(row_C)*K + (bn+threadIdx.x+block_size_half)];
    // A10
    subA[threadIdx.y+block_size_half][threadIdx.x] = A[(row_C+block_size_half)*K + (bn+threadIdx.x)];
    // A11
    subA[threadIdx.y+block_size_half][threadIdx.x+block_size_half] = A[(row_C+block_size_half)*K + (bn+threadIdx.x+block_size_half)];
    // B00
    subB[threadIdx.y][threadIdx.x] = B[(bn+threadIdx.y)*N + col_C];
    // B01
    subB[threadIdx.y][threadIdx.x+block_size_half] = B[(bn+threadIdx.y)*N + (col_C+block_size_half)];
    // B10
      subB[threadIdx.y+block_size_half][threadIdx.x] = B[(bn+threadIdx.y+block_size_half)*N + (col_C)];
    // B11
    subB[threadIdx.y+block_size_half][threadIdx.x+block_size_half] = B[(bn+threadIdx.y+block_size_half)*N + (col_C+block_size_half)];
    __syncthreads();
    for (int i = 0; i < BLOCK_SIZE_L; ++i) {
      float subA_y0i = subA[threadIdx.y][i];
      float subA_y1i = subA[threadIdx.y+block_size_half][i];
      float subB_ix0 = subB[i][threadIdx.x];
      float subB_ix1 = subB[i][threadIdx.x+block_size_half];
      accums[0][0] += subA_y0i * subB_ix0; 
      accums[0][1] += subA_y0i * subB_ix1; 
      accums[1][0] += subA_y1i * subB_ix0; 
      accums[1][1] += subA_y1i * subB_ix1; 
    }
    __syncthreads();
  }
  // write value into global memory
  // C00
  C[row_C * N + col_C] = accums[0][0];
  // C01
  C[row_C * N + col_C+block_size_half] = accums[0][1];
  // C10
  C[(row_C+block_size_half) * N + col_C] = accums[1][0];
  // C11
  C[(row_C+block_size_half) * N + col_C+block_size_half] = accums[1][1];
}

// shared_8workloads_padding
#define WORK_PERTHREAD 8
#define N_WORKERS (BLOCK_SIZE/WORK_PERTHREAD)
__global__ void kernel_shared_8w_pad(KERNEL_PARAMS_LIST) {
  // each thread compute 4 elements of C, row_C means the #row of first element.
  const int row_C = blockIdx.y * blockDim.y + threadIdx.y;
  const int col_C = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float subA[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float subB[BLOCK_SIZE][BLOCK_SIZE];
  
  float accums[WORK_PERTHREAD];
  #pragma unroll
  for (int i = 0; i < WORK_PERTHREAD; ++i) accums[i] = 0;

  for (int bn = 0; bn < K; bn+=BLOCK_SIZE) {
    #pragma unroll
    for (int i = 0; i < WORK_PERTHREAD; i++) {
      subA[threadIdx.y+i*N_WORKERS][threadIdx.x] = 
            A[(row_C+i*N_WORKERS)*K + (bn+threadIdx.x)];
      subB[threadIdx.y+i*N_WORKERS][threadIdx.x] = 
            B[(bn+threadIdx.y+i*N_WORKERS)*N + col_C];
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE; ++i) {
      #pragma unroll
      for (int j = 0; j < WORK_PERTHREAD; ++j) {
        accums[j] += subA[threadIdx.y+N_WORKERS*j][i] * subB[i][threadIdx.x];
      }
    }
    __syncthreads();
  }
  // write value into global memory
  #pragma unroll
  for (int j = 0; j < WORK_PERTHREAD; ++j) {
    C[(row_C+j*N_WORKERS) * N + col_C] = accums[j];
  }
}

#define WPTX 4  // workloads per thread at X direction
#define WPTY 8  // workloads per thread at y direction
#define NTX (BLOCK_SIZE_L/WPTX) // works at x direction
#define NTY (BLOCK_SIZE_L/WPTY) // works at y direction
#define tidx (threadIdx.x)
#define tidy (threadIdx.y)
// shared_16workloads2D_padding
__global__ void kernel_shared_16w2d_pad(KERNEL_PARAMS_LIST) {
  // each thread compute 32 elements of C, row_C means the #row of first element.
  const int row_C = blockIdx.y * blockDim.y + tidy;
  const int col_C = blockIdx.x * blockDim.x + tidx;

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
    // load into shared memory
    #pragma unroll
    for (int r = 0; r < WPTY; r++) {
      #pragma unroll
      for (int c = 0; c < WPTX; ++c) {
        subA[r*NTY+tidy][c*NTX+tidx] = A[(row_C+r*NTY)*K + (bn+c*NTX+tidx)];
        subB[r*NTY+tidy][c*NTX+tidx] = B[(bn+r*NTY+tidy)*N + (col_C+c*NTX)];
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

// shared_16workloads2D_padding loading data using vec
__global__ void kernel_shared_16w2d_pad_vec(
                const int M, const int N, const int K, 
                const float4 * A, const float4 * B, float *C) 
{
  // each thread compute 32 elements of C, row_C means the #row of first element.
  const int row_C = blockIdx.y * blockDim.y + tidy;
  const int col_C = blockIdx.x * blockDim.x + tidx;

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
        float4 vecB = __ldg(&B[(bn+r*NTY+tidy)*N/4 + (blockIdx.x*blockDim.x+(c*NTX+tidx)*4)/4]);
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