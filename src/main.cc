#include "utils.h"
#include "system.h"

int main(int argc, char* argv[]) {
  //S: Parse arguments //TODO: -k -usePCA -n -niters
  if (argc < 4) throw std::invalid_argument("Missing arguments");
  char *trainPath = argv[1],
       *testPath = argv[2], 
       *resultsPath = argv[3];

  uint k = 5, //U: Number of nearest neighbors to use //TODO: Better default
       n = 0, //U: Amount of eigenvectors calc'd for PCA //NOTE: Default means no PCA 
       niter = 1000; //U: Number of iterations to calc eigenvectors for PCA

  for (uint i = 4; i < argc; i++) { //A: Parse additional arguments
    char *thisArg = argv[i];
    if (thisArg[0] == '-')
      if (i+1 < argc)
        if (thisArg[1] == 'k') k = std::stoi(argv[i+1]);
        else if (thisArg[1] == 'n') n = std::stoi(argv[i+1]);
        else if (thisArg[1] == 'i') niter = std::stoi(argv[i+1]);
        else std::invalid_argument("Wrong option");
      else throw std::invalid_argument("Incomplete arguments");
  }
  bool usePCA = n > 0;

  printf("TP2 Metodos Numericos\nF. Galileo Cappella Lewi\ntrainPath: %s, testPath: %s, resultsPath: %s\nk: %i, usePCA: %s", trainPath, testPath, resultsPath, k, usePCA ? "true" : "false");
  if (usePCA) printf(", n: %i, niter: %i", n, niter);
  printf("\n");
  
  System sys(trainPath);
  if (usePCA) sys.calc_pca(n, niter);
  sys.guess(testPath, k);
  sys.save_results(resultsPath, usePCA);

  return 1;
}
