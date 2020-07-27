#pragma once

#include <stdio.h>

__global__ void kernel_im2col(
  const int C_img,                        // channel of image
  const int H_img, const int W_img,       // shape of image(input)
  const int H_out, const int W_out,       // shape of convolution output
  const int H_ker, const int W_ker,       // shape of kernel
  const int stride_w, const int stride_h,
  const int pad_w, const int pad_h, 
  const int dilation_w, const int dilation_h,
  const float* data_image,   // data ptr of image 
  float* data_column          // data ptr of column 
) {
  const int W_clm = H_out * W_out;         // size of output
  const int n_tasks = C_img * W_clm;  
  for (int tid=blockDim.x*blockIdx.x+threadIdx.x; tid<n_tasks; tid+=blockDim.x*gridDim.x) {
    int rowidx_out = tid / W_out % H_out;         // row index of output
    int colidx_out = tid % W_out;                 // col index of output

    int chaidx_img = tid / W_clm;                // channel idx of image
    int rowidx_img = rowidx_out * stride_h - pad_h; // row index of image
    int colidx_img = colidx_out * stride_w - pad_w; // col index of image

    int rowidx_clm = chaidx_img * H_ker * W_ker;    // row index of column
    int colidx_clm = tid % W_clm;                // row index of column

    const float* data_img = data_image + (chaidx_img*H_img + rowidx_img)*W_img + colidx_img;
    float* data_clm = data_column + rowidx_clm*W_clm + colidx_clm;

    // data_img += (chaidx_img*H_img + rowidx_img)*W_img + colidx_img;
    // data_clm += rowidx_clm*W_clm + colidx_clm;
    
    // process current kernel
    for (int i = 0; i < H_ker; ++i) {
      for (int j = 0; j < W_ker; ++j) {
        int ridx_img = rowidx_img + i * dilation_h;
        int cidx_img = colidx_img + j * dilation_w;
        // check out bound
        if (ridx_img < 0 || ridx_img >= H_img || 
            cidx_img < 0 || cidx_img >= W_img) {
          *data_clm = 0;
        } else {
          *data_clm = data_img[i * dilation_h * W_img + j * dilation_w];
        }
        data_clm += W_clm;
      } // for W_ker

    } // for H_ker

  } // for n_tasks

}


// 
__global__ void kernel_im2col_align(
  const int aligment,
  const int C_img,                              // channel of image
  const int H_img, const int W_img,             // shape of image(input)
  const int H_out, const int W_out,             // shape of convolution output
  const int H_ker, const int W_ker,             // shape of kernel
  const int stride_w, const int stride_h,
  const int pad_w, const int pad_h, 
  const int dilation_w, const int dilation_h,
  const float* data_image,    // data ptr of image 
  float* data_column           // data ptr of column 
) {
  const int H_clm_align = CEIL_DIV(C_img*H_ker*W_ker, aligment)*aligment;
  const int W_clm_align = CEIL_DIV(H_out*W_out, aligment)*aligment;
  const int C_img_align = H_clm_align / (H_ker * W_ker);
  const int n_tasks = C_img_align * W_clm_align;  
  for (int tid=blockDim.x*blockIdx.x+threadIdx.x; tid<n_tasks; tid+=blockDim.x*gridDim.x) {
    int rowidx_out = tid % W_clm_align / W_out;   // row index of output
    int colidx_out = tid % W_clm_align % W_out;   // col index of output

    int chaidx_img = tid / W_clm_align;                 // channel idx of image
    int rowidx_img = rowidx_out * stride_h - pad_h;     // row index of image
    int colidx_img = colidx_out * stride_w - pad_w;     // col index of image

    int rowidx_clm = chaidx_img * H_ker * W_ker;        // row index of column
    int colidx_clm = tid % W_clm_align;                 // col index of column

    // if (tid==64) {
    //   printf("rowidx_out=%d, colidx_out=%d, \n", rowidx_out, colidx_out);
    //   printf("chaidx_img=%d, rowidx_img=%d, colidx_img=%d\n", chaidx_img, rowidx_img, colidx_img);
    //   printf("rowidx_clm=%d, colidx_clm=%d\n", rowidx_clm, colidx_clm);
    // }

    const float* data_img = data_image + (chaidx_img*H_img + rowidx_img)*W_img + colidx_img;
    float* data_clm = data_column + rowidx_clm*W_clm_align + colidx_clm;

    // process current kernel
    for (int i = 0; i < H_ker; ++i) {
      for (int j = 0; j < W_ker; ++j) {
        int ridx_img = rowidx_img + i * dilation_h;
        int cidx_img = colidx_img + j * dilation_w;
        // check out bound
        if (ridx_img < 0 || ridx_img >= H_img || 
            cidx_img < 0 || cidx_img >= W_img || 
            chaidx_img >= C_img || 
            rowidx_out >= H_out || colidx_out >= W_out) {
          *data_clm = 0;
        } else {
          *data_clm = data_img[i * dilation_h * W_img + j * dilation_w];
        }
        data_clm += W_clm_align;
      } // for W_ker

    } // for H_ker

  } // for n_tasks

}