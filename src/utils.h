#ifndef _UTILS_
#define _UTILS_

#include <Eigen/Core>

#include <algorithm>
#include <type_traits>
#include <cmath>
#include <list>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <chrono>
#include <iomanip>
#include <stdexcept>

typedef unsigned int uint;
typedef long double floating_t;
typedef long int time_t;
typedef Eigen::Matrix<floating_t, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Vector<floating_t, Eigen::Dynamic> Vector;
typedef std::pair<Vector, uint> Entry; //TODO: Rename

time_t get_time(); //U: Returns the current time in milliseconds

struct SortedDistances {
  SortedDistances(uint);

  void push_back(uint, floating_t);
  uint consensus() const;

  std::vector<std::pair<uint, floating_t>> v;
  uint n;
};

#include "utils.cpp"
#endif
