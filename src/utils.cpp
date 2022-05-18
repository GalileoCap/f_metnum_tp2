#include "utils.h"

Vector row_norm_sqrd(const Ref<const Matrix>& X) {
  Vector res(X.rows());
  for (uint i = 0; i < X.rows(); i++)
    res(i) = X.row(i).squaredNorm();
  return res;
}

Matrix extend(Vector V, uint reps) {
  //TODO: reps > 1
  Matrix res(V.rows(), reps);
  for (uint i = 0; i < reps; i++)
    res.col(i) = V; 
  return res;
}

Matrix distances_sqrd(const Ref<const Matrix>& X, const Ref<const Matrix>& Y) {
  Vector XN = row_norm_sqrd(X),
         YN = row_norm_sqrd(Y);
  Matrix XX = extend(XN, Y.rows()),
         YY = extend(YN, X.rows());
  return -2 * X * Y.transpose() + XX + YY.transpose();
}
