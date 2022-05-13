#ifndef _SYSTEM_
#define _SYSTEM_

#include "utils.h"

struct System {
  System(char*); //U: Loads training data, and calc's avg and M

  void calc_avg(); //U: Calculates the average vector and normalizes X
  void calc_M();
  void calc_pca(uint, uint); //U: Calculates first n eigenvalues and eigenvectors using power iteration, reduces M and updates X to only use those

  void guess(char*, uint); //U: Makes a guess for an entire file of vectors
  void guess(Entry&, uint); //U: Makes a guess for a single vector

  void read_vectors(char*, std::list<Entry>&); //U: Reads a file of vectors and adds them to X
  void save_results(char*, bool) const;

  std::list<Entry>
    X, //U: Data used to train the system //TODO: Optimize for quickly getting nearest neighbors 
    Y; //U: Vectors to guess with their guessed result
  Matrix M; //U: Covariance matrix 
  Vector avg; //U: Average vector
  std::list<time_t> times; //U: Profiling times
};

#include "system.cpp"
#endif
