#include "utils.h"
#include "system.h"

int main(int argc, char* argv[]) {
  System sys(argv[1]);
  //sys.calc_pca(argv[5]);
  sys.guess(argv[2], 1);
  sys.save_results(argv[3]);

  return 1;
}
