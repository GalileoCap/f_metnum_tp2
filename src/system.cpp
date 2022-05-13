#include "system.h"

System::System(char *pathIn) { //U: Loads training data
  read_vectors(pathIn, X);
  calc_avg();
  calc_M();
}

void System::calc_avg() { //U: Calculates the average vector and normalizes X
  avg = Matrix(X.front().first.rows(), 1); avg.setZero();
  for (const Entry& x : X) avg += x.first; //A: Add up each vector
  avg /= X.size(); //A: Divide by the size
  for (Entry& x : X) //A: Update each vector
    x.first = (x.first - avg) / pow(X.size() - 1, 0.5);
}

void System::calc_M() {
  M = Matrix(X.size(), X.front().first.rows());
  uint i = 0; for (const Entry& x : X) M.row(i++) = x.first.transpose(); //A: Matrix with each vector as a row
  M = M.transpose() * M / (X.size() - 1);
}

void System::calc_pca(uint n, uint niter) { //U: Calculates first n eigenvalues and eigenvectors using power iteration, calculates and reduces M so that Mx = tc(x), and updates X with this new M; NOTE: Assumes at least one vector in X
  time_t start = get_time();
  //S: Power iteration
  Matrix _M(n, M.cols());
  for (uint i = 0; i < n; i++) { //A: Iterate for each eigenvector
    Matrix v = Vector::Random(M.rows()); //A: Start with a random vector //TODO: Check it's not zero
    floating_t lambda = 1;
    for (uint j = 0; j < niter && (M * v).isApprox(lambda * v); j++) { //A: Iterate until the result is close enough or reached max iters //TODO: Warn if reached max iters
      v = (M * v).normalized(); 
      lambda = ((v.transpose() * (M * v)) / v.norm())(0); //A: Calculate eigenvalue
    }
    M -= lambda * v * v.transpose(); //A: Remove this component from M
    _M.row(i) = v.transpose(); //A: Save this component //TODO: Should I multiply by lambda?
  }
  M = _M; //A: Replace M with the reduced matrix

  for (Entry& x : X) x.first = M * x.first; //A: Apply tc(x) for each training vector
  times.push_back(get_time() - start);
}

void System::guess(char *pathIn, uint k) { //U: Makes a guess for an entire file of vectors //TODO: Repeated code with the constructor
  read_vectors(pathIn, Y);

  for (Entry& y : Y) {
    time_t start = get_time();
    guess(y, k);
    times.push_back(get_time() - start);
  }
}

void System::guess(Entry& y, uint k) { //U: Makes a guess for a single vector
  y.first = M * (y.first - avg) / pow(X.size() - 1, 0.5); //A: x = tc(x)
  //S: Find k nearest vectors 
  SortedDistances distances(k);
  for (Entry& x : X)
    distances.push_back(x.second, (y.first - x.first).norm());
  y.second = distances.consensus(); //A: Set the label with most consensus
}

void System::read_vectors(char *pathIn, std::list<Entry>& X) { //U: Reads a file of vectors and adds them to X
  std::string line; char ignore;
  uint n = 0;
  std::ifstream fin(pathIn);
  if (!fin.is_open()) throw std::invalid_argument("invalid pathIn");

  //S: Read header line
  std::getline(fin, line); 
  for (const char &c : line) n += c == ','; //A: n = #cols - 1 = size of each vector
  
  //S: Read vectors
  while (std::getline(fin, line)) {
    X.emplace_back(n, 0); //A: Vector
    Entry& x = X.back(); 

    std::stringstream str(line);
    str >> x.second; //A: Read the label
    for (uint i = 0; i < n; i++) str >> ignore >> x.first(i, 0); //A: Read each pixel while skipping over commas
  }
  fin.close();
}

void System::save_results(char *pathOut, bool usePCA) const { //U: Writes the times, eigenvectors, and the results to a file
  std::ofstream fout(pathOut);
  for (const time_t& t : times) fout << t << ' '; fout << '\n';
  if (usePCA) fout << M; fout << '\n'; //A: Print PCA's eigenvectors
  for (const Entry& y : Y) fout << y.second << ' ';
  fout.close();
}
