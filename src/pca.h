#pragma once
#include "utils.h"

struct PCA {
  typedef std::pair<Vector, floating_t> EigenPair;

  PCA(uint, uint);

  void fit(const RMatrix&);
  Matrix transform(const RMatrix&);

  EigenPair _power_iter(const RMatrix&);
  Matrix _to_covariance(Matrix);
  void _deflate(RMatrix, const EigenPair&); //A: Removes the component v from M

  Matrix _M; Vector _avg;
  uint _n, _max_iter, _size;
};

#include "pca.cpp"
