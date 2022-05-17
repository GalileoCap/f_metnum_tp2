#pragma once
#include "pca.h"

PCA::PCA(uint n, uint max_iter) : _n(n), _max_iter(max_iter), _size(0) {};

bool PCA::stopFit(uint i, uint rows, uint cols, floating_t lambda_prev, floating_t lambda) {
  if (_n == 0) { //A: Choose automatically 
    uint eigenvalues = rows <= cols ? rows : cols; //A: The covariance matrix has up to min(rows, cols) eigenvalues
    return (i == eigenvalues) || //A: There's no more eigenvalues to calc
          (lambda * (eigenvalues - i) < lambda_prev); //A: The next eigenvalues don't add up to the previous one
  } else return (i == _n);
}

uint PCA::fit(const Ref<Matrix>& X) { 
  //TODO: Warn if wrong dimensions for the chosen n
  Matrix M = _to_covariance(X);
  std::list<Vector> eigenvectors; floating_t lambda_prev = 0, lambda = 0;
  uint i = 0;
  do { //A: Iterate for each component
    lambda_prev = lambda;
    EigenPair vl(_power_iter(M)); 
    _deflate(M, vl);
    eigenvectors.push_back(vl.first); //A: Save this component //TODO: Multiply by lambda?
    lambda = vl.second;
    i++;
  } while (!stopFit(i, X.rows(), X.cols(), lambda_prev, lambda));
  _M = Matrix(i, M.cols());
  uint j = 0; for (const Ref<Vector>& v : eigenvectors) _M.row(j++) = v;

  return i;
}

PCA::EigenPair PCA::_power_iter(const Ref<Matrix>& M) {
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
void PCA::_deflate(Ref<Matrix> M, const EigenPair& vl) { //U: Removes the component v from M
  M -= vl.second * vl.first * vl.first.transpose();
}

Matrix PCA::transform(const Ref<Matrix>& X) { 
  //TODO: Warn if wrong dimensions or if not trained (_size == 0)
  Matrix res(X.rows(), _M.rows());

  uint i = 0;
  for (auto x : X.rowwise()) //A: Transform each vector
    res.row(i++) = (_M * (x.transpose() - _avg) / pow(_size - 1, 0.5)).transpose();
  //TODO: Optimize

  return res;
}
