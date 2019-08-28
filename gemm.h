#pragma once
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int n_row;
  int n_col;
  float* data;
} Matrix;

void MatMul(const Matrix A, const Matrix B, Matrix C);