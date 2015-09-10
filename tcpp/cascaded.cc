#include <iostream>
#include <cstdlib>

#include "cto.hpp"

int main(int argc, char **argv) {
  std::cerr << "Creating CTO object." << std::endl;
  const char *fn = argv[1];
  unsigned int min_s = (argv[2] ? atoi(argv[2]) : 55);
  if (!fn) {
    std::cerr << "Usage: " << argv[0] << " [input_file] [min_samples]" << std::endl;
    exit(1);
  }
  CTO cto = CTO(std::string(fn));//, new Smote(fn, "bla", 5));
  
  std::cerr << "Done. Initiating oversampling with " << min_s << " docs..." << std::endl;
  cto.oversampling(min_s);
  std::cerr << "Done..." << std::endl;
}
