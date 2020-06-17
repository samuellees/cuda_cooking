#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <random>
#include "dot.h"
#include "assert.h"

void malloc_and_init_vector(Vector& vec) {
  int seed = 3;
  std::default_random_engine gen(seed);
  std::normal_distribution<double> distribut(0,1);
  vec.data = (float *)malloc(sizeof(float) * vec.length);
  for (int i = 0; i < vec.length; ++i) {
    (vec.data)[i] = distribut(gen);
  }
}

void print_vector(Vector& vec) {
  for (int i = 0; i < vec.length; ++i) {
      std::cout << vec.data[i] << ", ";
  }
  std::cout << std::endl;
}

int main() {
  Vector X = {65536, NULL};
  Vector Y = {65536, NULL};
  Vector Z = {256, NULL};
  malloc_and_init_vector(X);
  malloc_and_init_vector(Y);
  malloc_and_init_vector(Z);
  dot(X, Y, Z);

  Vector Z_ref = {256, NULL};
  malloc_and_init_vector(Z_ref);
  dot_ref(X, Y, Z_ref);

  std::cout << "cpu result: " << Z.data[0] << std::endl;
  std::cout << "gpu result: " << Z_ref.data[0] << std::endl;

  free(X.data);
  free(Y.data);
  free(Z.data);
  free(Z_ref.data);
  return 0;
}