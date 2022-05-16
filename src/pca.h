#pragma once
#include "utils.h"

struct PCA {
  typedef std::pair<Vector, floating_t> EigenPair;

  PCA(uint, uint);

  void fit(const Ref<Matrix>&);
  Matrix transform(const Ref<Matrix>&);

  EigenPair _power_iter(const Ref<Matrix>&);
  Matrix _to_covariance(Matrix);
  void _deflate(Ref<Matrix>, const EigenPair&); //A: Removes the component v from M

  Matrix _M; Vector _avg;
  uint _n, _max_iter, _size;
};

#include "pca.cpp"
