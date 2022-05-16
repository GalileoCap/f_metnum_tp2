#pragma once
#include <vector>
#include <tuple>
#include <map>
#include "utils.h"

struct KNN {
  KNN(uint);

  void fit(const RMatrix&, const RMatrix&);
  Vector predict(const RMatrix&) const;
  uint predict_one(const Eigen::Ref<const Vector>&) const;

  struct SortedDistances {
    SortedDistances(uint);

    void emplace_back(uint, floating_t);
    uint majority() const;

    uint _k;
    std::vector<std::pair<uint, floating_t>> _v;
  };

  uint _k;
  Matrix _X, _Y;
};

#include "knn.cpp"
