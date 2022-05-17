#pragma once
#include <tuple>
#include <queue>
#include <vector>
#include <map>

#include "utils.h"

struct KNN {
  typedef std::pair<Label, floating_t> LabelDist;

  KNN(uint);

  void fit(const Ref<Matrix>&, const Ref<Labels>&);
  Labels predict(const Ref<Matrix>&) const;
  Label predict_one(const Ref<const Vector>&) const;

  struct SortedDistances {
    SortedDistances(uint);

    void emplace_back(Label, floating_t);
    Label majority();

    struct Comp {
      bool operator()(const LabelDist&, const LabelDist&);
    };
    std::priority_queue<LabelDist, std::vector<LabelDist>, Comp> _v;
    uint _k;
  };

  Label _k;
  Matrix _X; Labels _Y;
};

#include "knn.cpp"
