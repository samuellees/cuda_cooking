// a demo of gemm


// grid_size(256, 256)   block_size(32, 32)
template<unsigned int block_size>
__global__ void gemm_kernel_v1(const float* A, const float* B, float* C, 
                            const int M, const int N, const int K) 
{
  // here M==N==K==8196
  const int col_C = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_C = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ float sharedA[block_size][block_size];
  __shared__ float sharedB[block_size][block_size];

  float tmp_result = 0;
  // loop reduce axis
  for (int k_offset = 0; k_offset < K; k_offset += block_size) {
    // load data into shared A and B
    int col_A = k_offset + threadIdx.x;
    int row_B = k_offset + threadIdx.y;
    sharedA[threadIdx.y][threadIdx.x] = A[row_C * K + col_A];
    sharedB[threadIdx.y][threadIdx.x] = B[row_B * N + col_C];
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < block_size; ++kk) {
      tmp_result += sharedA[threadIdx.y][kk] * sharedB[kk][threadIdx.x];
    }
    __syncthreads();
  }
  C[row_C * N + col_C] = tmp_result;
}


// grid_size(256, 256)   block_size(4, 8)
// template<unsigned int block_size>
__global__ void gemm_kernel_v2(const float* A, const float* B, float* C, 
                            const int M, const int N, const int K) 
{
  const int tile_size = 32;
  // here M==N==K==8196
  const int col_C = blockIdx.x * tile_size;
  const int row_C = blockIdx.y * tile_size;

  __shared__ float sharedA[tile_size][tile_size];
  __shared__ float sharedB[tile_size][tile_size];
  float regA[8];
  float regB[4];
  float regC[8][4];
  // init regC
  #pragma unroll
  for (int regC_row = 0; regC_row < 8; ++regC_row) {
    #pragma unroll
    for (int regC_col = 0; regC_col < 4; ++ regC_col) {
      regC[regC_row][regC_col] = 0;
    }
  }

  // loop reduce axis
  for (int k_offset = 0; k_offset < K; k_offset += tile_size) {
    // load data into shared A and B
    for (int sh_idx = 0; sh_idx < tile_size; ++sh_idx) {
      sharedA[sh_idx][threadIdx.y * blockDim.x + threadIdx.x] = A[(sh_idx + row_C) * K + threadIdx.y * blockDim.x + threadIdx.x  + k_offset];
      sharedB[sh_idx][threadIdx.y * blockDim.x + threadIdx.x] = B[(k_offset + sh_idx) * N + col_C + threadIdx.y * blockDim.x + threadIdx.x];
    }
    __syncthreads();
    // loop a tile
    for (int kk = 0; kk < tile_size; ++kk) {
      // load into reg A and B
      #pragma unroll
      for (int regA_idx = 0; regA_idx < 8; ++regA_idx) {
        regA[regA_idx] = sharedA[regA_idx * 4 + threadIdx.y][kk];
      }
      #pragma unroll
      for (int regB_idx = 0; regB_idx < 4; ++regB_idx) {
        regB[regB_idx] = sharedB[kk][regB_idx * 8 + threadIdx.x];
      }

      // calculate regC
      #pragma unroll
      for (int regC_row = 0; regC_row < 8; ++regC_row) {
        #pragma unroll
        for (int regC_col = 0; regC_col < 4; ++ regC_col) {
          regC[regC_row][regC_col] += regA[regC_row] * regB[regC_col];
        }
      }
    }
    __syncthreads();
  }
  // write C back
  #pragma unroll
  for (int regC_row = 0; regC_row < 8; ++regC_row) {
    #pragma unroll
    for (int regC_col = 0; regC_col < 4; ++ regC_col) {
      C[(row_C + regC_row * 4 + threadIdx.y) * N + (col_C + regC_col * 8 + threadIdx.x)] = regC[regC_row][regC_col];
    }
  }
}