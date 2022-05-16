#pragma once

#include <Eigen/Dense>

#include <stdexcept>
#include <iostream>

typedef unsigned int uint;
typedef long double floating_t;

typedef Eigen::MatrixXd Matrix;
typedef Eigen::Ref<Matrix, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> RMatrix; //SEE: https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html#storage-orders
typedef Eigen::VectorXd Vector;
