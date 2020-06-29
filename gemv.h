#pragma once

struct Matrix {
  int n_row;
  int n_col;
  float* data;  // pointer to data
};

struct Vector {
  int length;       
  float* data;  // pointer to data
};

void gemv(const Matrix X, const Vector Y, Vector Z);

// reference implementation on CPU
void gemv_ref(const Matrix X, const Vector Y, Vector Z);