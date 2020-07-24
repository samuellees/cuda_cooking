#include "conv_kernels.cuh"
#include <iostream>

void conv1x1_test() {
  // input size
  const int BATCH_SIZE = 64;
  const int Ci = 128;
  const int Hi = 128;
  const int Wi = 128;
  // kernel size
  const int Co = 128;
  const int Hk = 1; 
  const int Wk = 1;
  // output size
  const int Ho = Hi;
  const int Wo = Wi;
  // host data
  float* input = nullptr;
  float* kernel = nullptr;
  float* output = nullptr;
  const int size_input = BATCH_SIZE * Ci * Hi * Wi;
  const int size_kernel = Co * Ci * Hk * Wk;
  const int size_output = BATCH_SIZE * Co * Ho * Wo;
  malloc_and_init(&input, size_input);
  malloc_and_init(&kernel, size_kernel);
  malloc_and_init(&output, size_output);
  // device data
  float* d_input = nullptr;
  float* d_kernel = nullptr;
  float* d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, size_input * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_kernel, size_kernel * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, size_output * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, input, size_input * sizeof(float),  cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kernel, kernel, size_kernel * sizeof(float),  cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_output, output, size_output * sizeof(float),  cudaMemcpyHostToDevice));
  // prepare matrix size
  const int M = Co;
  const int K = Ci;
  const int N = Ho * Wo;


  // conv ref
  float* output_ref = nullptr;
  float* d_output_ref = nullptr;
  malloc_and_init(&output_ref, size_output);
  CUDA_CHECK(cudaMalloc(&d_output_ref, size_output * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_output_ref, output_ref, size_output * sizeof(float),  cudaMemcpyHostToDevice));
  float time_cudnn;
  convCuDNN(BATCH_SIZE, Ci, Hi, Wi, d_input, 
            Co, Hk, Wk, d_kernel, Ho, Wo, d_output_ref, &time_cudnn);
  time_cudnn = time_cudnn / n_rounds;
  printf("kernel %-20s: %8.2f ms, speedup=%.2f.\n", 
    "conv_cudnn", 
    time_cudnn, 
    1.0);
  

  // conv1x1
  float elapsedTime_conv_1x1;
  dim3 dims_block_conv_1x1(NTX, NTY);
  dim3 dims_grid_conv_1x1(CEIL_DIV(N, BLOCK_SIZE_L), BATCH_SIZE * CEIL_DIV(M, BLOCK_SIZE_L));
  // warm up
  kernel_conv_1x1<<<dims_grid_conv_1x1, dims_block_conv_1x1>>>(
    M, N, K, (float4*)d_kernel, (float4*)d_input, d_output);
  cudaEvent_t start_conv_1x1, stop_conv_1x1;
  CUDA_CHECK(cudaEventCreate(&start_conv_1x1)); 
  CUDA_CHECK(cudaEventCreate(&stop_conv_1x1));
  CUDA_CHECK(cudaEventRecord(start_conv_1x1, 0));
  for (int i = 0; i < n_rounds; ++i) {
    kernel_conv_1x1<<<dims_grid_conv_1x1, dims_block_conv_1x1>>>(
      M, N, K, (float4*)d_kernel, (float4*)d_input, d_output);
  }
  CUDA_CHECK(cudaEventRecord(stop_conv_1x1, 0)); 
  CUDA_CHECK(cudaEventSynchronize(stop_conv_1x1)); 
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_conv_1x1, start_conv_1x1, stop_conv_1x1));
  CUDA_CHECK(cudaEventDestroy(start_conv_1x1)); 
  CUDA_CHECK(cudaEventDestroy(stop_conv_1x1));
  elapsedTime_conv_1x1 = elapsedTime_conv_1x1 / n_rounds;
  printf("kernel %-20s: %8.2f ms, speedup=%.2f.\n", 
    "conv_1x1", 
    elapsedTime_conv_1x1, 
    elapsedTime_conv_1x1 / time_cudnn);


  // copy result to host
  CUDA_CHECK(cudaMemcpy(output_ref, d_output_ref, size_output * sizeof(float),  cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(output, d_output, size_output * sizeof(float),  cudaMemcpyDeviceToHost));
  // check
  std::cout << "check correctness..." << std::endl;
  bool error = false;
  #pragma unroll 64
  for (int i = 0; i < BATCH_SIZE * Co * Ho * Wo; ++i) {
    error = error || (std::abs(output[i] - output_ref[i]) > 1e-3);
  }
  std::cout << "error: " << error << std::endl;
  // free memory
  delete[] input;
  delete[] kernel;
  delete[] output;
  delete[] output_ref;
  cudaFree(d_input);
  cudaFree(d_kernel);
  cudaFree(d_output);
  cudaFree(d_output_ref);
}


void conv3x3_test() {
  // input size
  const int BATCH_SIZE = 64;
  const int Ci = 64;
  const int Hi = 64;
  const int Wi = 64;
  // kernel size
  const int Co = 64;
  const int Hk = 3;
  const int Wk = 3;
  // output size
  const int Ho = Hi;
  const int Wo = Wi;
  // host data
  float* input = nullptr;
  float* kernel = nullptr;
  float* output = nullptr;
  const int size_input = BATCH_SIZE * Ci * Hi * Wi;
  const int size_kernel = Co * Ci * Hk * Wk;
  const int size_output = BATCH_SIZE * Co * Ho * Wo;
  malloc_and_init(&input, size_input);
  malloc_and_init(&kernel, size_kernel);
  malloc_and_init(&output, size_output);
  // device data
  float* d_input = nullptr;
  float* d_kernel = nullptr;
  float* d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, size_input * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_kernel, size_kernel * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, size_output * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, input, size_input * sizeof(float),  cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kernel, kernel, size_kernel * sizeof(float),  cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_output, output, size_output * sizeof(float),  cudaMemcpyHostToDevice));
  // prepare matrix size
  const int M = Co;
  const int K = Ci;
  const int N = Ho * Wo;


  // conv ref
  float* output_ref = nullptr;
  float* d_output_ref = nullptr;
  malloc_and_init(&output_ref, size_output);
  CUDA_CHECK(cudaMalloc(&d_output_ref, size_output * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_output_ref, output_ref, size_output * sizeof(float),  cudaMemcpyHostToDevice));
  float time_cudnn;
  convCuDNN(BATCH_SIZE, Ci, Hi, Wi, d_input, 
            Co, Hk, Wk, d_kernel, Ho, Wo, d_output_ref, &time_cudnn);
  time_cudnn = time_cudnn / n_rounds;
  printf("kernel %-20s: %8.2f ms, speedup=%.2f.\n", 
    "conv_cudnn", 
    time_cudnn, 
    1.0);
  

  // conv1x1
  float elapsedTime_conv_1x1;
  dim3 dims_block_conv_1x1(NTX, NTY);
  dim3 dims_grid_conv_1x1(CEIL_DIV(N, BLOCK_SIZE_L), BATCH_SIZE * CEIL_DIV(M, BLOCK_SIZE_L));
  // warm up
  kernel_conv_1x1<<<dims_grid_conv_1x1, dims_block_conv_1x1>>>(
    M, N, K, (float4*)d_kernel, (float4*)d_input, d_output);
  cudaEvent_t start_conv_1x1, stop_conv_1x1;
  CUDA_CHECK(cudaEventCreate(&start_conv_1x1)); 
  CUDA_CHECK(cudaEventCreate(&stop_conv_1x1));
  CUDA_CHECK(cudaEventRecord(start_conv_1x1, 0));
  for (int i = 0; i < n_rounds; ++i) {
    kernel_conv_1x1<<<dims_grid_conv_1x1, dims_block_conv_1x1>>>(
      M, N, K, (float4*)d_kernel, (float4*)d_input, d_output);
  }
  CUDA_CHECK(cudaEventRecord(stop_conv_1x1, 0)); 
  CUDA_CHECK(cudaEventSynchronize(stop_conv_1x1)); 
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_conv_1x1, start_conv_1x1, stop_conv_1x1));
  CUDA_CHECK(cudaEventDestroy(start_conv_1x1)); 
  CUDA_CHECK(cudaEventDestroy(stop_conv_1x1));
  elapsedTime_conv_1x1 = elapsedTime_conv_1x1 / n_rounds;
  printf("kernel %-20s: %8.2f ms, speedup=%.2f.\n", 
    "conv_1x1", 
    elapsedTime_conv_1x1, 
    elapsedTime_conv_1x1 / time_cudnn);


  // copy result to host
  CUDA_CHECK(cudaMemcpy(output_ref, d_output_ref, size_output * sizeof(float),  cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(output, d_output, size_output * sizeof(float),  cudaMemcpyDeviceToHost));
  // check
  std::cout << "check correctness..." << std::endl;
  bool error = false;
  #pragma unroll 64
  for (int i = 0; i < BATCH_SIZE * Co * Ho * Wo; ++i) {
    error = error || (std::abs(output[i] - output_ref[i]) > 1e-3);
  }
  std::cout << "error: " << error << std::endl;
  // free memory
  delete[] input;
  delete[] kernel;
  delete[] output;
  delete[] output_ref;
  cudaFree(d_input);
  cudaFree(d_kernel);
  cudaFree(d_output);
  cudaFree(d_output_ref);
}


void convCuDNN(
  const int BATCH_SIZE, const int Ci, const int Hi, const int Wi, const float* input,
  const int Co, const int Hk, const int Wk, const float* kernel,
  const int Ho, const int Wo, float* output,
  float * time_ptr) {    
  //handle
  cudnnHandle_t handle;
  cudnnCreate(&handle);
  // tensor descriptor
  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t output_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
    input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    BATCH_SIZE, Ci, Hi, Wi));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
    output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
    BATCH_SIZE, Co, Ho, Wo));
  // kernel 
  cudnnFilterDescriptor_t kernel_desc;
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&kernel_desc));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(
    kernel_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    Co, Ci, Hk, Wk));
  // convolution descriptor
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnCreateConvolutionDescriptor(&conv_desc);
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
    0, 0, // zero-padding
    1, 1, // stride
    1, 1, // dilation
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  // algorithm
  cudnnConvolutionFwdAlgo_t algo;
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
    handle, input_desc, kernel_desc, conv_desc, output_desc, 
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
    0, &algo));
  // workspace size && allocate memory
  size_t workspace_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle,
    input_desc, kernel_desc, conv_desc, output_desc,
    algo, &workspace_size));
  void * workspace = nullptr;
  CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
  // convolution
  auto alpha = 1.0f, beta = 0.0f;
  // warm  up
  CUDNN_CHECK(cudnnConvolutionForward(handle,
    &alpha, input_desc, input,
    kernel_desc, kernel,
    conv_desc, algo,
    workspace, workspace_size,
    &beta, output_desc, output));
  cudaEvent_t start_conv_ref, stop_conv_ref;
  CUDA_CHECK(cudaEventCreate(&start_conv_ref)); 
  CUDA_CHECK(cudaEventCreate(&stop_conv_ref));
  CUDA_CHECK(cudaEventRecord(start_conv_ref, 0));
  for (int i = 0; i < n_rounds; ++i) {
    CUDNN_CHECK(cudnnConvolutionForward(handle,
      &alpha, input_desc, input,
      kernel_desc, kernel,
      conv_desc, algo,
      workspace, workspace_size,
      &beta, output_desc, output));
  }
  CUDA_CHECK(cudaEventRecord(stop_conv_ref, 0)); 
  CUDA_CHECK(cudaEventSynchronize(stop_conv_ref)); 
  CUDA_CHECK(cudaEventElapsedTime(time_ptr, start_conv_ref, stop_conv_ref));
  CUDA_CHECK(cudaEventDestroy(start_conv_ref)); 
  CUDA_CHECK(cudaEventDestroy(stop_conv_ref));
  // destroy
  cudaFree(workspace);
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyTensorDescriptor(output_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroyFilterDescriptor(kernel_desc);
  cudnnDestroy(handle);
} 