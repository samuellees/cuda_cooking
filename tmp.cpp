
__global__ void im2col_kernel(
    const int64_t n,
    const dt* data_img,
    const int64_t height,
    const int64_t width,
    const int64_t kernel_height,
    const int64_t kernel_width,
    const int64_t pad_height,
    const int64_t pad_width,
    const int64_t stride_height,
    const int64_t stride_width,
    const int64_t dilation_height,
    const int64_t dilation_width,
    const int64_t H_out,
    const int64_t W_out,
    dt* data_clm) {
  CUDA_KERNEL_LOOP(tid, n) {
    int64_t rowidx_out_at_clm = tid / W_out;
    int64_t rowidx_out = rowidx_out_at_clm % H_out;
    int64_t colidx_out = tid % W_out;

    int64_t chaidx_img = rowidx_out_at_clm / H_out;
    int64_t rowidx_img = rowidx_out * stride_height - pad_height;
    int64_t colidx_img = colidx_out * stride_width - pad_width;

    int64_t rowidx_clm = chaidx_img * kernel_height * kernel_width;

    data_img += (chaidx_img * height + rowidx_img) * width + colidx_img;
    data_clm += (rowidx_clm * H_out * W_out + rowidx_out*W_out + colidx_out;

    for (int64_t i = 0; i < kernel_height; ++i) {
      for (int64_t j = 0; j < kernel_width; ++j) {
        int64_t h = rowidx_img + i * dilation_height;
        int64_t w = colidx_img + j * dilation_width;
        *data_clm = (h >= 0 && w >= 0 && h < height && w < width)
            ? data_img[i * dilation_height * width + j * dilation_width]
            : ScalarConvert<int, dt>::to(0);
        data_clm += H_out * W_out;
      }
    }
  }
}