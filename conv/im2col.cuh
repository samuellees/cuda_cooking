#pragma once

#include <stdio.h>

__global__ void kernel_im2col(
  const int64_t C_img,                        // channel of image
  const int64_t H_img, const int64_t W_img,       // shape of image(input)
  const int64_t H_out, const int64_t W_out,       // shape of convolution output
  const int64_t H_ker, const int64_t W_ker,       // shape of kernel
  const int64_t stride_w, const int64_t stride_h,
  const int64_t pad_w, const int64_t pad_h, 
  const int64_t dilation_w, const int64_t dilation_h,
  const float* data_image,   // data ptr of image 
  float* data_column          // data ptr of column 
) {
  const int64_t W_clm = H_out * W_out;         // size of output
  const int64_t n_tasks = C_img * W_clm;  
  for (int64_t tid=blockDim.x*blockIdx.x+threadIdx.x; tid<n_tasks; tid+=blockDim.x*gridDim.x) {
    int64_t rowidx_out = tid / W_out % H_out;         // row index of output
    int64_t colidx_out = tid % W_out;                 // col index of output

    int64_t chaidx_img = tid / W_clm;                // channel idx of image
    int64_t rowidx_img = rowidx_out * stride_h - pad_h; // row index of image
    int64_t colidx_img = colidx_out * stride_w - pad_w; // col index of image

    int64_t rowidx_clm = chaidx_img * H_ker * W_ker;    // row index of column
    int64_t colidx_clm = tid % W_clm;                // row index of column

    const float* data_img = data_image + (chaidx_img*H_img + rowidx_img)*W_img + colidx_img;
    float* data_clm = data_column + rowidx_clm*W_clm + colidx_clm;
    
    // process current kernel
    for (int64_t i = 0; i < H_ker; ++i) {
      for (int64_t j = 0; j < W_ker; ++j) {
        int64_t ridx_img = rowidx_img + i * dilation_h;
        int64_t cidx_img = colidx_img + j * dilation_w;
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


__global__ void kernel_im2col_align(
  const int64_t aligment,
  const int64_t C_img,                              // channel of image
  const int64_t H_img, const int64_t W_img,             // shape of image(input)
  const int64_t H_out, const int64_t W_out,             // shape of convolution output
  const int64_t H_ker, const int64_t W_ker,             // shape of kernel
  const int64_t stride_w, const int64_t stride_h,
  const int64_t pad_w, const int64_t pad_h, 
  const int64_t dilation_w, const int64_t dilation_h,
  const float* data_image,    // data ptr of image 
  float* data_column           // data ptr of column 
) {
  const int64_t H_clm_align = CEIL_DIV(C_img*H_ker*W_ker, aligment)*aligment;
  const int64_t W_clm_align = CEIL_DIV(H_out*W_out, aligment)*aligment;
  const int64_t C_img_align = H_clm_align / (H_ker * W_ker);
  const int64_t n_tasks = C_img_align * W_clm_align;  
  for (int64_t tid=blockDim.x*blockIdx.x+threadIdx.x; tid<n_tasks; tid+=blockDim.x*gridDim.x) {
    int64_t rowidx_out = tid % W_clm_align / W_out;   // row index of output
    int64_t colidx_out = tid % W_clm_align % W_out;   // col index of output

    int64_t chaidx_img = tid / W_clm_align;                 // channel idx of image
    int64_t rowidx_img = rowidx_out * stride_h - pad_h;     // row index of image
    int64_t colidx_img = colidx_out * stride_w - pad_w;     // col index of image

    int64_t rowidx_clm = chaidx_img * H_ker * W_ker;        // row index of column
    int64_t colidx_clm = tid % W_clm_align;                 // col index of column

    // if (tid==64) {
    //   printf("rowidx_out=%d, colidx_out=%d, \n", rowidx_out, colidx_out);
    //   printf("chaidx_img=%d, rowidx_img=%d, colidx_img=%d\n", chaidx_img, rowidx_img, colidx_img);
    //   printf("rowidx_clm=%d, colidx_clm=%d\n", rowidx_clm, colidx_clm);
    // }

    const float* data_img = data_image + (chaidx_img*H_img + rowidx_img)*W_img + colidx_img;
    float* data_clm = data_column + rowidx_clm*W_clm_align + colidx_clm;

    // process current kernel
    for (int64_t i = 0; i < H_ker; ++i) {
      for (int64_t j = 0; j < W_ker; ++j) {
        int64_t ridx_img = rowidx_img + i * dilation_h;
        int64_t cidx_img = colidx_img + j * dilation_w;
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



__global__ void kernel_im2col_align_with_batch(
  const int64_t aligment,
  const int64_t batch_size,
  const int64_t C_img,                              // channel of image
  const int64_t H_img, const int64_t W_img,             // shape of image(input)
  const int64_t H_out, const int64_t W_out,             // shape of convolution output
  const int64_t H_ker, const int64_t W_ker,             // shape of kernel
  const int64_t stride_w, const int64_t stride_h,
  const int64_t pad_w, const int64_t pad_h, 
  const int64_t dilation_w, const int64_t dilation_h,
  const float* data_image,    // data ptr of image 
  float* data_column           // data ptr of column 
) {
  const int64_t H_clm_align = CEIL_DIV(C_img*H_ker*W_ker, aligment)*aligment;
  const int64_t W_clm_align = CEIL_DIV(H_out*W_out, aligment)*aligment;
  const int64_t C_img_align = H_clm_align / (H_ker * W_ker);
  const int64_t n_tasks = C_img_align * W_clm_align * batch_size;
  for (int64_t tid=blockDim.x*blockIdx.x+threadIdx.x; tid<n_tasks; tid+=blockDim.x*gridDim.x) {
    int64_t batch_id = tid / (C_img_align * W_clm_align);
    tid %= (C_img_align * W_clm_align);
    int64_t rowidx_out = tid % W_clm_align / W_out;   // row index of output
    int64_t colidx_out = tid % W_clm_align % W_out;   // col index of output

    int64_t chaidx_img = tid / W_clm_align;                 // channel idx of image
    int64_t rowidx_img = rowidx_out * stride_h - pad_h;     // row index of image
    int64_t colidx_img = colidx_out * stride_w - pad_w;     // col index of image

    int64_t rowidx_clm = chaidx_img * H_ker * W_ker;        // row index of column
    int64_t colidx_clm = tid % W_clm_align;                 // col index of column

    // if (tid==64) {
    //   printf("rowidx_out=%d, colidx_out=%d, \n", rowidx_out, colidx_out);
    //   printf("chaidx_img=%d, rowidx_img=%d, colidx_img=%d\n", chaidx_img, rowidx_img, colidx_img);
    //   printf("rowidx_clm=%d, colidx_clm=%d\n", rowidx_clm, colidx_clm);
    // }

    const float* data_img = data_image + batch_id*C_img*H_img*W_img + (chaidx_img*H_img + rowidx_img)*W_img + colidx_img;
    float* data_clm = data_column + batch_id*H_clm_align*W_clm_align + rowidx_clm*W_clm_align + colidx_clm;

    // process current kernel
    for (int64_t i = 0; i < H_ker; ++i) {
      for (int64_t j = 0; j < W_ker; ++j) {
        int64_t ridx_img = rowidx_img + i * dilation_h;
        int64_t cidx_img = colidx_img + j * dilation_w;
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