#include "utils.h"
#include "system.h"

int main(int argc, char* argv[]) {
  //S: Parse arguments //TODO: -k -usePCA -n -niters
  if (argc < 4) throw std::invalid_argument("Missing arguments");
  char *trainPath = argv[1],
       *testPath = argv[2], 
       *resultsPath = argv[3];
  uint k = argc >= 5 ? std::stoi(argv[4]) : 5, //U: Number of nearest neighbors to use //TODO: Better default
       n = argc >= 6 ? std::stoi(argv[5]) : 0, //U: Amount of eigenvectors calc'd for PCA //NOTE: Default is never used 
       niter = argc >= 7 ? std::stoi(argv[6]) : 1000; //U: Number of iterations to calc eigenvectors for PCA
  bool usePCA = argc > 5;

  printf("TP2 Metodos Numericos\nF. Galileo Cappella Lewi\ntrainPath: %s, testPath: %s, resultsPath: %s\nk: %i, usePCA: %s", trainPath, testPath, resultsPath, k, usePCA ? "true" : "false");
  if (usePCA) printf(", n: %i, niter: %i", n, niter);
  printf("\n");
  
  System sys(trainPath);
  if (usePCA) sys.calc_pca(n, niter);
  sys.guess(testPath, k);
  sys.save_results(resultsPath, usePCA);

  return 1;
}
