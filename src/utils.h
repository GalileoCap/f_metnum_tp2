#ifndef _UTILS_
#define _UTILS_

#include <Eigen/Core>

#include <math.h>
#include <vector>
#include <fstream>
//#include <iostream>
#include <stdio.h>
//#include <string.h>
#include <chrono>
#include <iomanip>
#include <stdexcept>

typedef unsigned int uint;
typedef long double floating_t;
typedef long int time_t;
typedef Eigen::Matrix<floating_t, Eigen::Dynamic, Eigen::Dynamic> Matrix;

time_t get_time(); //U: Returns the current time in milliseconds

#include "utils.cpp"
#endif
