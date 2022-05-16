#include "knn.h"

KNN::KNN(uint k) : _k(k) {}

void KNN::fit(const RMatrix& X, const RMatrix& Y) {
  _X = X; _Y = Y;
};

Vector KNN::predict(const RMatrix& X) const {
  Vector res(X.rows());

  uint i = 0;
  for (const auto& x : X.rowwise()) //A: Guess for each vector
    res(i) = predict_one(x);

  return res;
}

uint KNN::predict_one(const Eigen::Ref<const Vector>& x) const { 
  SortedDistances neighbors(_k);

  for (uint i = 0; i < _X.rows(); i++)
    neighbors.emplace_back(_Y(i), (x.transpose() - _X.row(i)).norm());

  return neighbors.majority();
}

KNN::SortedDistances::SortedDistances(uint k) : _k(k) {};

void KNN::SortedDistances::emplace_back(uint y, floating_t d) {
  uint i = 0; while (i < _v.size() && _v[i].second < d) i++;
  std::pair<uint, floating_t> prev(y, d), tmp; 
  for (uint j = i; j < _v.size(); j++) { //A: Swap from i until the end to keep it sorted
    tmp = prev;
    prev = _v[j];
    _v[j] = tmp;
  }
  if (_v.size() < _k) _v.push_back(prev);
}

uint KNN::SortedDistances::majority() const {
  std::map<uint, uint> counts;
  for (const std::pair<uint, floating_t> kv : _v) {
    uint label = kv.first;
    if (counts.count(label)) counts[label]++;
    else counts[label] = 1;
  }

  const std::pair<const uint, uint> *heaviest = nullptr;
  for (const std::pair<const uint, uint>& kv : counts)
    if (!heaviest || kv.second < heaviest->second) heaviest = &kv;

  return heaviest->first;
}
