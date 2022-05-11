#include "utils.h"

long get_time() { //U: Returns the current time in milliseconds
  auto start = std::chrono::system_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count();
}

SortedDistances::SortedDistances(uint _n) : n(_n) {}

void SortedDistances::push_back(uint label, floating_t d) {
  uint i = 0; while (i < v.size() && v[i].second < d) i++;
  std::pair<uint, floating_t> prev(label, d), tmp; 
  for (uint j = i; j < v.size(); j++) {
    tmp = prev;
    prev = v[j];
    v[j] = tmp;
  }
  if (v.size() < n) v.push_back(prev);
}

uint SortedDistances::consensus() const {
  std::map<uint, uint> counts;
  for (uint i = 0; i < v.size(); i++) {
    uint label = v[i].first;
    if (counts.count(label)) counts[label]++;
    else counts[label] = 1;
  }

  const std::pair<const uint, uint> *heaviest = nullptr;
  for (const std::pair<const uint, uint>& kv : counts)
    if (!heaviest || kv.second < heaviest->second) heaviest = &kv;

  return heaviest->first;
}
