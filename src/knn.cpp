#include "knn.h"

KNN::KNN(uint k) : _k(k) {}

void KNN::fit(const Ref<Matrix>& X, const Ref<Labels>& Y) {
  _X = X; _Y = Y;
};

Labels KNN::predict(const Ref<Matrix>& X) const {
  Labels res(X.rows());

  Matrix distances = distances_sqrd(X, _X);
  for (uint x = 0; x < distances.rows(); x++)
    res(x) = predict_one(distances, x);

  return res;
}

Label KNN::predict_one(const Ref<Matrix>& distances, uint x) const { 
  SortedDistances neighbors(_k);

  for (uint _x = 0; _x < distances.cols(); _x++) {
    neighbors.emplace_back(_Y(_x), distances(x, _x));
  }

  return neighbors.majority();
}

KNN::SortedDistances::SortedDistances(uint k) : _k(k) {};

void KNN::SortedDistances::emplace_back(Label y, floating_t d) {
  _v.push(LabelDist (y, d));
}

Label KNN::SortedDistances::majority() {
  //TODO: Check _size >= k
  std::map<Label, uint> counts;
  for (uint i = 0; i < _k; i++) {
    Label label = _v.top().first; _v.pop();
    if (counts.count(label)) counts[label]++;
    else counts[label] = 1;
  }

  const std::pair<const Label, uint> *heaviest = nullptr;
  for (const std::pair<const Label, uint>& kv : counts)
    if (!heaviest || kv.second < heaviest->second) heaviest = &kv;

  return heaviest->first;
}

bool KNN::SortedDistances::Comp::operator()(const LabelDist& left, const LabelDist& right) {
  return left.second > right.second;
};
