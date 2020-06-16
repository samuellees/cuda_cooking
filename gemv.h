#pragma once

struct Matrix {
  int n_row;    // number of rows
  int n_col;    // number of columns
  int ld;       // leading dimensions
  float* data;  // pointer to data
};

struct Vector {
  int length;       // leading dimensions
  float* data;  // pointer to data
};

/**
 * @brief GEMV
 *
 */
void gemv(const Matrix A, const Vector X, Vector Y);