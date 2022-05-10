#ifndef _SYSTEM_
#define _SYSTEM_

#include "utils.h"

struct System {
  System(char*); //U: Loads training data, and calc's avg and M

  void calc_avg(); //U: Calculates the average vector and normalizes X
  void calc_M();
  void calc_pca(uint, uint); //U: Calculates first n eigenvalues and eigenvectors using power iteration, reduces M and updates X to only use those

  void guess(char*, uint); //U: Makes a guess for an entire file of vectors
  uint guess(Matrix, uint) const; //U: Makes a guess for a single vector

  void read_vectors(char*, std::vector<std::pair<Matrix, uint>>&); //U: Reads a file of vectors and adds them to X
  void save_results(char*, bool) const;

  std::vector<std::pair<Matrix, uint>> X; //U: Data used to train the system TODO: Optimize for quickly getting nearest neighbors 
  Matrix M, //U: Covariance matrix 
         avg; //U: Average vector
  std::vector<uint> results; //U: Resulting values
  std::vector<time_t> times; //U: Profiling times
};

#include "system.cpp"
#endif
