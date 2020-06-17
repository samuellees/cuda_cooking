#pragma once

struct Vector {
  int length;       
  float* data;  // pointer to data
};

void dot(const Vector X, const Vector Y, Vector Z);

// reference implementation on CPU
void dot_ref(const Vector X, const Vector Y, Vector Z);