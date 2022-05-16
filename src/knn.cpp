#include "knn.h"

KNN::KNN(uint k) : _k(k) {}

void KNN::fit(const Ref<Matrix>& X, const Ref<Labels>& Y) {
  _X = X; _Y = Y;
};

Labels KNN::predict(const Ref<Matrix>& X) const {
  Labels res(X.rows());

  uint i = 0;
  for (const auto& x : X.rowwise()) //A: Guess for each vector
    res(i) = predict_one(x);

  return res;
}

Label KNN::predict_one(const Ref<const Vector>& x) const { 
  SortedDistances neighbors(_k);

  for (uint i = 0; i < _X.rows(); i++)
    neighbors.emplace_back(_Y(i), (x.transpose() - _X.row(i)).norm());

  return neighbors.majority();
}

KNN::SortedDistances::SortedDistances(uint k) : _k(k) {};

void KNN::SortedDistances::emplace_back(Label y, floating_t d) {
  uint i = 0; while (i < _v.size() && _v[i].second < d) i++;
  std::pair<Label, floating_t> prev(y, d), tmp; 
  for (uint j = i; j < _v.size(); j++) { //A: Swap from i until the end to keep it sorted
    tmp = prev;
    prev = _v[j];
    _v[j] = tmp;
  }
  if (_v.size() < _k) _v.push_back(prev);
}

Label KNN::SortedDistances::majority() const {
  std::map<Label, uint> counts;
  for (const std::pair<Label, floating_t> kv : _v) {
    Label label = kv.first;
    if (counts.count(label)) counts[label]++;
    else counts[label] = 1;
  }

  const std::pair<const Label, uint> *heaviest = nullptr;
  for (const std::pair<const Label, uint>& kv : counts)
    if (!heaviest || kv.second < heaviest->second) heaviest = &kv;

  return heaviest->first;
}
