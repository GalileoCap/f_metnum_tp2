#pragma once
#include "utils.h"

struct PCA {
  typedef std::pair<Vector, floating_t> EigenPair;

  PCA(uint, uint);

  void fit(Matrix);
  Matrix transform(const RMatrix&);

  EigenPair _power_iter(const RMatrix&);
  void _deflate(RMatrix, const EigenPair&); //A: Removes the component v from M

  Matrix _M;
  uint _n, _max_iter;
  bool _trained;
};

#include "pca.cpp"
