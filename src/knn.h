#pragma once
#include <vector>
#include <tuple>
#include <map>
#include "utils.h"

struct KNN {
  KNN(uint);

  void fit(const RMatrix&, const RMatrix&);
  Vector guess(const RMatrix&) const;
  uint guess_one(Vector) const; //TODO: Pass-by-copy because of Eigen's shenanigans

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
