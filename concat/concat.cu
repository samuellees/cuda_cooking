#include <cstdint>
#include "concat_cpy.cuh"
#include "concat_ms.cuh"
#include "concat_cpyasync.cuh"
#include <iostream>

using namespace std;

void concat_test() {
  // input size
  const int64_t BATCH_SIZE = 100;
  const int64_t C1 = 100000;
  const int64_t C2 = 200000;
  const int64_t C3 = 300000;
  const int64_t C4 = 400000;

  const int64_t size1 = (C1 + C2) * BATCH_SIZE;
  const int64_t size2 = (C1 + C2 + C3) * BATCH_SIZE;
  const int64_t size3 = (C1 + C2 + C3 + C4) * BATCH_SIZE;

  using scalar_t = float;

  scalar_t* input1 = nullptr;
  scalar_t* input2 = nullptr;
  scalar_t* input3 = nullptr;
  scalar_t* input4 = nullptr;
  scalar_t* output1 = nullptr;
  scalar_t* output2 = nullptr;
  scalar_t* output3 = nullptr;

  int item_size = sizeof(scalar_t);
  cudaStream_t cuda_stream;
  cudaStream_t cuda_stream1;
  cudaStream_t cuda_stream2;
  CUDA_CHECK(cudaStreamCreate(&cuda_stream));
  CUDA_CHECK(cudaStreamCreate(&cuda_stream1));
  CUDA_CHECK(cudaStreamCreate(&cuda_stream2));
  
  CUDA_CHECK(cudaMalloc(&input1, BATCH_SIZE * C1 * item_size));
  CUDA_CHECK(cudaMalloc(&input2, BATCH_SIZE * C2 * item_size));
  CUDA_CHECK(cudaMalloc(&input3, BATCH_SIZE * C3 * item_size));
  CUDA_CHECK(cudaMalloc(&input4, BATCH_SIZE * C4 * item_size));
  CUDA_CHECK(cudaMalloc(&output1, size1 * item_size));
  CUDA_CHECK(cudaMalloc(&output2, size2 * item_size));
  CUDA_CHECK(cudaMalloc(&output3, size3 * item_size));
  ConcatKernelCpy(size1, C1, C2, input1, input2, output1);
  ConcatKernelCpy(size2, C1, C2, C3, input1, input2, input3, output2);
  ConcatKernelCpy(size3, C1, C2, C3, C4, input1, input2, input3, input4, output3);
  ConcatKernel(size1, C1, C2, input1, input2, output1, cuda_stream);
  ConcatKernel(size2, C1, C2, C3, input1, input2, input3, output2, cuda_stream);
  ConcatKernel(size3, C1, C2, C3, C4, input1, input2, input3, input4, output3, cuda_stream);
  ConcatKernelCpyAsync(size1, C1, C2, input1, input2, output1, cuda_stream1, cuda_stream2);
  ConcatKernelCpyAsync(size2, C1, C2, C3, 
                      input1, input2, input3, output2, cuda_stream1, cuda_stream2);
  ConcatKernelCpyAsync(size3, C1, C2, C3, C4, 
                      input1, input2, input3, input4, output3, cuda_stream1, cuda_stream2);
  CUDA_CHECK(cudaDeviceSynchronize()); 
  

  float elapsedTime_concat2_cpy = 0;
  cudaEvent_t start_concat2_cpy, stop_concat2_cpy;
  CUDA_CHECK(cudaEventCreate(&start_concat2_cpy)); 
  CUDA_CHECK(cudaEventCreate(&stop_concat2_cpy));
  CUDA_CHECK(cudaEventRecord(start_concat2_cpy, 0));
  for (int64_t i = 0; i < n_rounds; ++i) {
    // ConcatKernelCpy(size1, C1, C2, input1, input2, output1);
    // ConcatKernelCpy(size2, C1, C2, C3, input1, input2, input3, output2);
    ConcatKernelCpy(size3, C1, C2, C3, C4, input1, input2, input3, input4, output3);
  }
  CUDA_CHECK(cudaEventRecord(stop_concat2_cpy, 0)); 
  CUDA_CHECK(cudaEventSynchronize(stop_concat2_cpy)); 
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_concat2_cpy, start_concat2_cpy, stop_concat2_cpy));
  CUDA_CHECK(cudaEventDestroy(start_concat2_cpy)); 
  CUDA_CHECK(cudaEventDestroy(stop_concat2_cpy));
  cout << "concat2_cpy: " << elapsedTime_concat2_cpy << endl;


  float elapsedTime_concat2_krn = 0;
  cudaEvent_t start_concat2_krn, stop_concat2_krn;
  CUDA_CHECK(cudaEventCreate(&start_concat2_krn)); 
  CUDA_CHECK(cudaEventCreate(&stop_concat2_krn));
  CUDA_CHECK(cudaEventRecord(start_concat2_krn, 0));
  for (int64_t i = 0; i < n_rounds; ++i) {
    // ConcatKernel(size1, C1, C2, input1, input2, output1, cuda_stream);
    // ConcatKernel(size2, C1, C2, C3, input1, input2, input3, output2, cuda_stream);
    ConcatKernel(size3, C1, C2, C3, C4, input1, input2, input3, input4, output3, cuda_stream);
  }
  CUDA_CHECK(cudaEventRecord(stop_concat2_krn, 0)); 
  CUDA_CHECK(cudaEventSynchronize(stop_concat2_krn)); 
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_concat2_krn, start_concat2_krn, stop_concat2_krn));
  CUDA_CHECK(cudaEventDestroy(start_concat2_krn)); 
  CUDA_CHECK(cudaEventDestroy(stop_concat2_krn));
  cout << "concat2_krn: " << elapsedTime_concat2_krn << endl;


  float elapsedTime_concat2_cpyasync = 0;
  cudaEvent_t start_concat2_cpyasync, stop_concat2_cpyasync;
  CUDA_CHECK(cudaEventCreate(&start_concat2_cpyasync)); 
  CUDA_CHECK(cudaEventCreate(&stop_concat2_cpyasync));
  CUDA_CHECK(cudaEventRecord(start_concat2_cpyasync));
  for (int64_t i = 0; i < n_rounds; ++i) {
    // ConcatKernelCpyAsync(size1, C1, C2, input1, input2, output1, cuda_stream1, cuda_stream2);
    // ConcatKernelCpyAsync(size2, C1, C2, C3, 
    //                     input1, input2, input3, output2, cuda_stream1, cuda_stream2);
    ConcatKernelCpyAsync(size3, C1, C2, C3, C4, 
                        input1, input2, input3, input4, output3, cuda_stream1, cuda_stream2);
  }
  CUDA_CHECK(cudaStreamSynchronize(cuda_stream1));
  CUDA_CHECK(cudaStreamSynchronize(cuda_stream2));
  CUDA_CHECK(cudaEventRecord(stop_concat2_cpyasync)); 
  CUDA_CHECK(cudaEventSynchronize(stop_concat2_cpyasync)); 
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_concat2_cpyasync, start_concat2_cpyasync, stop_concat2_cpyasync));
  CUDA_CHECK(cudaEventDestroy(start_concat2_cpyasync)); 
  CUDA_CHECK(cudaEventDestroy(stop_concat2_cpyasync));
  cout << "concat2_cpyasync: " << elapsedTime_concat2_cpyasync << endl;


  CUDA_CHECK(cudaStreamDestroy(cuda_stream));
  CUDA_CHECK(cudaStreamDestroy(cuda_stream1));
  CUDA_CHECK(cudaStreamDestroy(cuda_stream2));
  CUDA_CHECK(cudaFree(input1));
  CUDA_CHECK(cudaFree(input2));
  CUDA_CHECK(cudaFree(input3));
  CUDA_CHECK(cudaFree(input4));
  CUDA_CHECK(cudaFree(output1));
  CUDA_CHECK(cudaFree(output2));
  CUDA_CHECK(cudaFree(output3));
}