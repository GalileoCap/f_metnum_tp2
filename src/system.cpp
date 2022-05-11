#include "system.h"

System::System(char *pathIn) { //U: Loads training data
  read_vectors(pathIn, X);
  calc_avg();
  calc_M();
}

void System::calc_avg() { //U: Calculates the average vector and normalizes X
  avg = Matrix(X[0].first.rows(), 1); avg.setZero();
  for (uint i = 0; i < X.size(); i++) avg += X[i].first; //A: Add up each vector
  avg /= X.size(); //A: Divide by the size
  for (uint i = 0; i < X.size(); i++) //A: Update each vector
    X[i].first = (X[i].first - avg) / pow(X.size() - 1, 0.5);
}

void System::calc_M() {
  Matrix _X(X.size(), X[0].first.rows()); for (uint i = 0; i < X.size(); i++) _X.row(i) = (X[i].first).transpose(); //A: Matrix with each vector as a row
  M = _X.transpose() * _X;
}

void System::calc_pca(uint n, uint niter) { //U: Calculates first n eigenvalues and eigenvectors using power iteration, calculates and reduces M so that Mx = tc(x), and updates X with this new M; NOTE: Assumes at least one vector in X
  time_t start = get_time();
  //S: Power iteration
  Matrix _M(M); M = Matrix(n, M.cols()); //A: Copy and remove every component from the original
  for (uint i = 0; i < n; i++) { //A: Iterate for each eigenvector
    Matrix v = Matrix::Random(_M.rows(), 1); //A: Start with a random vector //TODO: Check it's not zero
    floating_t lambda = 1;
    for (uint j = 0; j < niter && (_M * v).isApprox(lambda * v); j++) { //A: Iterate until the result is close enough or reached max iters //TODO: Warn if reached max iters
      v = (_M * v).normalized(); 
      lambda = ((v.transpose() * (_M * v)) / v.norm())(0); //A: Calculate eigenvalue; NOTE: The result is a 1x1 matrix
    }
    _M -= lambda * v * v.transpose(); //A: Remove this component from _M
    M.row(i) = v.transpose(); //A: Add this component to M //TODO: Should I multiply by lambda?
  }

  for (uint i = 0; i < X.size(); i++) X[i].first = M * X[i].first; //A: Apply tc(x) for each training vector
  times.push_back(get_time() - start);
}

void System::guess(char *pathIn, uint k) { //U: Makes a guess for an entire file of vectors //TODO: Repeated code with the constructor
  std::vector<std::pair<Matrix, uint>> toGuess;
  read_vectors(pathIn, toGuess);

  for (uint i = 0; i < toGuess.size(); i++) {
    time_t start = get_time();
    results.push_back(guess(toGuess[i].first, k));
    times.push_back(get_time() - start);
  }
}

uint System::guess(Matrix x, uint k) const { //U: Makes a guess for a single vector
  x = M * (x - avg) / pow(X.size() - 1, 0.5); //A: x = tc(x)
  //S: Find k nearest vectors //TODO: Use chunks to speed up
  SortedDistances distances(k);
  for (uint i = 0; i < X.size(); i++)
    distances.push_back(X[i].second, (X[i].first - x).norm());
  return distances.consensus(); //A: Return the label with most consensus
}

void System::read_vectors(char *pathIn, std::vector<std::pair<Matrix, uint>>& X) { //U: Reads a file of vectors and adds them to X
  std::string line; char ignore;
  uint n = 0;
  std::ifstream fin(pathIn);
  if (!fin.is_open()) throw std::invalid_argument("invalid pathIn");

  //S: Read header line
  std::getline(fin, line); 
  for (const char &c : line) n += c == ','; //A: n = #cols - 1 = size of each vector
  
  //S: Read vectors
  while (std::getline(fin, line)) {
    Matrix x(n, 1); //A: Vector of 1xn
    uint label; //A: What number is shown in the image

    std::stringstream str(line);
    str >> label;
    for (uint i = 0; i < n; i++) str >> ignore >> x(i, 0); //A: Read each pixel while skipping over commas
    
    X.emplace_back(x, label);
  }
  fin.close();
}

void System::save_results(char *pathOut, bool usePCA) const { //U: Writes the times, eigenvectors, and the results to a file
  std::ofstream fout(pathOut);
  for (const time_t& t : times) fout << t << ' '; fout << '\n';
  if (usePCA) for (uint i = 0; i < M.rows(); i++) fout << M.row(i) << '\n'; //A: Print PCA's eigenvectors
  else fout << '\n'; //A: Empty line
  for (uint n : results) fout << n << ' ';
  fout.close();
}
