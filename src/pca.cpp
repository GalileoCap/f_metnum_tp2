#pragma once
#include <iostream>
#include "pca.h"

PCA::PCA(uint n, uint max_iter) : _n(n), _max_iter(max_iter), _size(0) {};

void PCA::fit(const RMatrix& X) { 
  //TODO: Warn if wrong dimensions for the chosen n
  Matrix M = _to_covariance(X);
  _M = Matrix(_n, M.cols());
  for (uint i = 0; i < _n; i++) { //A: Iterate for each component
    EigenPair vl(_power_iter(M)); 
    _deflate(M, vl);
    _M.row(i) = vl.first; //A: Save this component //TODO: Multiply by lambda?
  }
}

PCA::EigenPair PCA::_power_iter(const RMatrix& M) {
  std::pair<Vector, floating_t> res(Vector::Random(M.rows()), 0); //A: Start with a random vector and lambda = 1 //TODO: Check it's not zero
  Eigen::Ref<Vector> v = res.first; floating_t& lambda = res.second; //A: Rename
  for (uint i = 0; i < _max_iter && (M * v).isApprox(lambda * v); i++) { //A: Iterate until the result is close enough 
    v = (M * v).normalized(); 
    lambda = ((v.transpose() * (M * v)) / v.norm())(0); //A: Calculate eigenvalue
  }
  return res;
}

Matrix PCA::_to_covariance(Matrix X) {
  _avg = X.colwise().mean(); //A: Average of each columm
  _size = X.rows();

  X.rowwise() -= _avg.transpose(); //A: Set average to zero
  X /= pow(_size - 1, 0.5); 
  return X.transpose() * X; //A: X = Covariance matrix of X
}
void PCA::_deflate(RMatrix M, const EigenPair& vl) { //U: Removes the component v from M
  M -= vl.second * vl.first * vl.first.transpose();
}

Matrix PCA::transform(const RMatrix& X) { 
  //TODO: Warn if wrong dimensions or if not trained (_size == 0)
  Matrix res(X.rows(), _n);

  uint i = 0;
  for (auto x : X.rowwise()) //A: Transform each vector
    res.row(i++) = (_M * (x.transpose() - _avg) / pow(_size - 1, 0.5)).transpose();
  //TODO: Optimize

  return res;
}
