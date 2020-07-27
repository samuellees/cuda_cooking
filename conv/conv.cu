#include "conv_kernels.cuh"
#include "im2col.cuh"
#include "utils.cuh"
#include <iostream>

void conv_1x1_im2col_test() {
  // input size
  const int BATCH_SIZE = 32;
  const int Ci = 64;
  const int Hi = 64;
  const int Wi = 64;
  // kernel size
  const int Co = 64;
  const int Hk = 1; 
  const int Wk = 1;
  // padding, stride and dilation
  const int pad_h = 0;
  const int pad_w = 0;
  const int stride_h = 1;
  const int stride_w = 1;
  const int dilation_h = 1;
  const int dilation_w = 1;
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
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
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
  kernel_conv_im2col_align<<<dims_grid_conv_1x1, dims_block_conv_1x1>>>(
    M, N, K, (float4*)d_kernel, (float4*)d_input, d_output);
  cudaEvent_t start_conv_1x1, stop_conv_1x1;
  CUDA_CHECK(cudaEventCreate(&start_conv_1x1)); 
  CUDA_CHECK(cudaEventCreate(&stop_conv_1x1));
  CUDA_CHECK(cudaEventRecord(start_conv_1x1, 0));
  for (int i = 0; i < n_rounds; ++i) {
    kernel_conv_im2col_align<<<dims_grid_conv_1x1, dims_block_conv_1x1>>>(
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


void conv_NxN_im2col_with_batch_test() {
  // input size
  const int BATCH_SIZE = 1;
  const int Ci = 32;
  const int Hi = 32;
  const int Wi = 32;
  // kernel size
  const int Co = 32;
  const int Hk = 3;
  const int Wk = 3;
  // padding, stride and dilation
  const int pad_h = 0;
  const int pad_w = 0;
  const int stride_h = 2;
  const int stride_w = 2;
  const int dilation_h = 1;
  const int dilation_w = 1;
  // output size
  const int Ho = (Hi - ((Hk-1)*dilation_h+1) + 2*pad_h) / stride_h + 1;
  const int Wo = (Wi - ((Wk-1)*dilation_w+1) + 2*pad_w) / stride_w + 1;
  // prepare matrix size
  const int alignment = BLOCK_SIZE_L;
  // const int alignment = 8;
  const int M = Co;
  const int K = Ci * Hk * Wk;
  const int N = Ho * Wo;
  const int M_align = align(M, alignment);
  const int K_align = align(K, alignment);
  const int N_align = align(N, alignment);
  // column size
  const int Hc_align = K_align;
  const int Wc_align = N_align;
  // host data
  float* input = nullptr;
  float* kernel = nullptr;
  float* column = nullptr;
  float* column_align = nullptr;
  float* output = nullptr;
  float* output_align = nullptr;
  const int size_input = BATCH_SIZE * Ci * Hi * Wi;
  const int size_kernel = M * K;                    // (Co * Ci * Hk * Wk)
  const int size_kernel_align = M_align * K_align;  
  const int size_column = BATCH_SIZE * Hk * Wk * Ho * Wo * Ci;
  const int size_column_align = BATCH_SIZE * Hc_align * Wc_align;
  const int size_output = BATCH_SIZE * M * N; // = BATCH_SIZE * Co * Ho * Wo
  const int size_output_align = BATCH_SIZE * M_align * N_align;
  malloc_and_init(&input, size_input);
  malloc_and_init(&kernel, size_kernel);
  malloc_and_init(&column, size_column);
  malloc_and_init(&column_align, size_column_align);
  malloc_and_init(&output, size_output);
  malloc_and_init(&output_align, size_output_align);
  // device data
  float* d_input = nullptr;
  float* d_kernel = nullptr;
  float* d_kernel_align = nullptr;
  float* d_column = nullptr;
  float* d_column_align = nullptr;
  float* d_output = nullptr;
  float* d_output_align = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, size_input * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_kernel, size_kernel * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_kernel_align, size_kernel_align * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_column, size_column * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_column_align, size_column_align * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, size_output * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output_align, size_output_align * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, input, size_input * sizeof(float),  cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kernel, kernel, size_kernel * sizeof(float),  cudaMemcpyHostToDevice));

  // conv ref
  float* output_ref = nullptr;
  float* d_output_ref = nullptr;
  malloc_and_init(&output_ref, size_output);
  CUDA_CHECK(cudaMalloc(&d_output_ref, size_output * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_output_ref, output_ref, size_output * sizeof(float),  cudaMemcpyHostToDevice));
  float time_cudnn;
  convCuDNN(BATCH_SIZE, Ci, Hi, Wi, d_input,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
            Co, Hk, Wk, d_kernel, Ho, Wo, d_output_ref, &time_cudnn);
  time_cudnn = time_cudnn / n_rounds;
  printf("kernel %-20s: %8.2f ms, speedup=%.2f.\n", 
    "conv_cudnn", 
    time_cudnn, 
    1.0);
  

  // conv im2col
  float elapsedTime_conv;
  dim3 dims_block_conv(NTX, NTY);
  dim3 dims_grid_conv(CEIL_DIV(N, BLOCK_SIZE_L), BATCH_SIZE * CEIL_DIV(M, BLOCK_SIZE_L));
  // dim3 dims_block_conv(BLOCK_SIZE, BLOCK_SIZE);
  // dim3 dims_grid_conv(CEIL_DIV(N, BLOCK_SIZE_L), BATCH_SIZE * CEIL_DIV(M, BLOCK_SIZE_L));

  // im2col
  kernel_im2col_align<<<1024, 1024>>>(
    alignment,
    Ci, 
    Hi, Wi, 
    Ho, Wo,
    Hk, Wk, 
    stride_w, stride_h, 
    pad_w, pad_h, 
    dilation_w, dilation_h,
    d_input,
    d_column_align);
  // kernel_im2col<<<1024, 1024>>>(
  //   Ci, 
  //   Hi, Wi, 
  //   Ho, Wo,
  //   Hk, Wk, 
  //   stride_w, stride_h, 
  //   pad_w, pad_h, 
  //   dilation_w, dilation_h,
  //   d_input,
  //   d_column);
  
  // CUDA_CHECK(cudaDeviceSynchronize());
  // CUDA_CHECK(cudaMemcpy(column, d_column, size_column * sizeof(float),  cudaMemcpyDeviceToHost));
  // CUDA_CHECK(cudaMemcpy(column_align, d_column_align, size_column_align * sizeof(float),  cudaMemcpyDeviceToHost));
  // CUDA_CHECK(cudaDeviceSynchronize());

  // std::cout << "input1:" << std::endl;
  // print_matrix(input, Hi, Wi);

  // std::cout << "input2:" << std::endl;
  // print_matrix(input+Wi*Hi, Hi, Wi);

  // std::cout << "col:" << std::endl;
  // print_matrix(column, Ci*Hk*Wk, Wo*Ho);

  // std::cout << "col_align:" << std::endl;
  // print_matrix(column_align, Hc_align, Wc_align);

  // warm up
  kernel_conv_im2col_align<<<dims_grid_conv, dims_block_conv>>>(
    M_align, N_align, K_align, (float4*)d_kernel_align, (float4*)d_column_align, d_output_align);
  // align
  float time_padding = padding(d_kernel, d_kernel_align, M, K, M_align, K_align);
  cudaEvent_t start_conv, stop_conv;
  CUDA_CHECK(cudaEventCreate(&start_conv)); 
  CUDA_CHECK(cudaEventCreate(&stop_conv));
  CUDA_CHECK(cudaEventRecord(start_conv, 0));
  for (int i = 0; i < n_rounds; ++i) {
    kernel_conv_im2col_align<<<dims_grid_conv, dims_block_conv>>>(
      M_align, N_align, K_align, (float4*)d_kernel_align, (float4*)d_column_align, d_output_align);
  }
  CUDA_CHECK(cudaEventRecord(stop_conv, 0)); 
  CUDA_CHECK(cudaEventSynchronize(stop_conv)); 
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime_conv, start_conv, stop_conv));
  CUDA_CHECK(cudaEventDestroy(start_conv)); 
  CUDA_CHECK(cudaEventDestroy(stop_conv));
  float time_unpadding = unpadding(d_output, d_output_align, M, N, M_align, N_align);
  elapsedTime_conv = elapsedTime_conv / n_rounds + time_unpadding + time_padding;
  printf("kernel %-20s: %8.2f ms, speedup=%.2f.\n", 
    "conv_NxN_im2col", 
    elapsedTime_conv, 
    elapsedTime_conv / time_cudnn);


  // copy result to host
  CUDA_CHECK(cudaMemcpy(output_ref, d_output_ref, size_output * sizeof(float),  cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(output, d_output, size_output * sizeof(float),  cudaMemcpyDeviceToHost));

  // std::cout << "output_ref:" << std::endl;
  // print_matrix(output_ref, M, N);

  // std::cout << "output:" << std::endl;
  // print_matrix(output, M, N);

  // check
  std::cout << "check correctness..." << std::endl;
  bool error = false;
  #pragma unroll 64
  for (int i = 0; i < BATCH_SIZE * Co * Ho * Wo; ++i) {
    error = error || (std::abs(output[i] - output_ref[i]) > 1e-3);
  }
  std::cout << "error: " << error << std::endl;
  // free memory
  free(input);
  free(kernel);
  free(output);
  free(output_ref);
  cudaFree(d_input);
  cudaFree(d_kernel);
  cudaFree(d_column_align);
  cudaFree(d_output);
  cudaFree(d_output_align);
  cudaFree(d_output_ref);
  free(column);
  cudaFree(d_column);
  free(column_align);
}


void convCuDNN(
  const int BATCH_SIZE, const int Ci, const int Hi, const int Wi, const float* input,
  const int pad_h, const int pad_w, 
  const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w,
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
    pad_h, pad_w,           // padding
    stride_h, stride_w,     // stride
    dilation_h, dilation_w, // dilation
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