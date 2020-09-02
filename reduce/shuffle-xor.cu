#include <stdio.h>
__global__ void warpReduce() {
  int laneId = threadIdx.x & 0x1f;
  // Seed starting value as inverse lane ID
  //int value = 31 - laneId;
  int value = 1;
  // Use XOR mode to perform butterfly reduction
  for (int i=16; i>=1; i/=2)
    value += __shfl_xor_sync(0xffffffff, value, i, 32);
  // "value" now contains the sum across all threads
  printf("Thread %d final value = %d\n", threadIdx.x, value);
}
int main() {
  warpReduce<<< 1, 32 >>>();
  cudaDeviceSynchronize();
  return 0;
}
