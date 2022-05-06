#include "system.h"

System::System(char *pathIn) { //U: Loads training data
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

  //S: Calculate avg and M
  avg = Matrix(X[0].first.rows(), 1); avg.setZero();
  for (uint i = 0; i < X.size(); i++) avg += X[i].first; //A: Add up each vector
  avg /= X.size(); //A: Divide by the size
  for (uint i = 0; i < X.size(); i++) //A: Update each vector
    X[i].first = (X[i].first - avg) / pow(X.size() - 1, 0.5);

  Matrix _X(X.size(), X[0].first.rows()); for (uint i = 0; i < X.size(); i++) _X.row(i) = (X[i].first).transpose(); //A: Matrix with each vector as a row
  M = _X.transpose() * _X;
}

void System::calc_pca(uint n, uint niter) { //U: Calculates first n eigenvalues and eigenvectors using power iteration, calculates and reduces M so that Mx = tc(x), and updates X with this new M; NOTE: Assumes at least one vector in X
  time_t start = get_time();
  //S: Power iteration
  Matrix _M(M); M = Matrix(n, M.cols()); //A: Copy and remove every component from the original
  for (uint i = 0; i < n; i++) { //A: Iterate for each eigenvector
    Matrix v = Matrix::Random(_M.rows(), 1); //A: Start with a random vector
    for (uint j = 0; j < niter; j++) v = (_M * v).normalized(); //A: Iterate niter times
    floating_t lambda = ((v.transpose() * (_M * v)) / v.norm())(0); //A: Calculate eigenvalue; NOTE: The result is a 1x1 matrix
    _M -= lambda * v * v.transpose(); //A: Remove this component from _M
    M.row(i) = v.transpose(); //A: Add this component to M //TODO: Should I multiply by lambda?
  }

  for (uint i = 0; i < X.size(); i++) X[i].first = M * X[i].first; //A: Apply tc(x) for each training vector
  times.push_back(get_time() - start);
}

void System::guess(char *pathIn, uint k) { //U: Makes a guess for an entire file of vectors //TODO: Repeated code with the constructor
  std::string line; char ignore;
  uint n = 0;
  std::ifstream fin(pathIn);
  if (!fin.is_open()) throw std::invalid_argument("invalid pathIn");

  //S: Read header line
  std::getline(fin, line); 
  for (const char &c : line) n += c == ','; //A: n = #cols = size of each vector
  n++;
  
  //S: Read vectors
  while (std::getline(fin, line)) {
    Matrix x(n, 1); //A: Vector of 1xn

    std::stringstream str(line);
    for (uint i = 0; i < n; i++) str >> x(i, 0) >> ignore; //A: Read each pixel while skipping over commas
    
    time_t start = get_time();
    results.push_back(guess(x, k));
    times.push_back(get_time() - start);
  }
  fin.close();
}

uint System::guess(Matrix x, uint k) const { //U: Makes a guess for a single vector
  x = M * (x - avg) / pow(X.size() - 1, 0.5); //A: x = tc(x)
  //S: Find k nearest vectors //TODO: Use chunks to speed up
  SortedDistances distances(k);
  for (uint i = 0; i < X.size(); i++)
    distances.push_back(X[i].second, (X[i].first - x).norm());
  return distances.consensus(); //A: Return the label with most consensus
}

void System::save_results(char *pathOut) const { //U: Writes the times and results to a file
  std::ofstream fout(pathOut);
  for (const time_t& t : times) fout << t << ' '; fout << '\n';
  for (uint n : results) fout << n << ' ';
  fout.close();
}
