#pragma once
#include <vector>
#include <tuple>
#include <map>
#include "utils.h"

struct KNN {
  KNN(uint);

  void fit(const Ref<Matrix>&, const Ref<Labels>&);
  Labels predict(const Ref<Matrix>&) const;
  Label predict_one(const Ref<const Vector>&) const;

  struct SortedDistances {
    SortedDistances(uint);

    void emplace_back(Label, floating_t);
    Label majority() const;

    uint _k;
    std::vector<std::pair<Label, floating_t>> _v;
  };

  Label _k;
  Matrix _X; Labels _Y;
};

#include "knn.cpp"
