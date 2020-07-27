#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include "conv.h"

template<typename scalar_t>
void print_data(const scalar_t * data, int length) {
  for (int i = 0; i < length; ++i) {
      std::cout << data[i] << ", ";
  }
  std::cout << std::endl;
}

int main() {
  // conv_1x1_im2col_test();
  conv_NxN_im2col_with_batch_test();
  
  return 0;
}