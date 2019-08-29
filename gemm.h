#pragma once
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int n_row;    // number of rows
  int n_col;    // number of columns
  int ld;       // leading dimensions
  float* data;  // pointer to data
} Matrix;

/**
 * @brief To simplify the logic, shared memory in GPU is in square shape(4x4),
 * so the shape of A, B and C should be multiples of 4.
 *
 * @param A
 * @param B
 * @param C
 */
void MatMul(const Matrix A, const Matrix B, Matrix C);