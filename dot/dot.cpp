#include "dot.h"

void dot_ref(const Vector X, const Vector Y, Vector Z) {
  const float * x = X.data;
  const float * y = Y.data;
  float * z = Z.data;
  int len = X.length;

  z[0] = 0;
  for (int i = 0; i < len; ++i) {
    z[0] += x[i] * y[i];
  }
}