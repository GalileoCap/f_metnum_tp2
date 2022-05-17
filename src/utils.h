#pragma once

#include <Eigen/Core>

#include <cmath>
#include <iostream>
#include <list>

typedef unsigned int uint;
typedef unsigned long Label;
typedef long double floating_t;

typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;
typedef Eigen::VectorX<Label> Labels;
template<typename T> using Ref = Eigen::Ref<T, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>; //SEE: https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html#storage-orders
