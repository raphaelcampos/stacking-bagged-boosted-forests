#include <cstdlib>
#include <iostream>
#include <fstream>

#include "utils.hpp"
#include "fs_bns.hpp"

int main(int argc, char **argv) {
  if (!argv[1]) {
    std::cerr << "Please specify input file." << std::endl;
    exit(1);
  }
  if (!argv[2]) {
    std::cerr << "Please specify the percentage of features to be removed." << std::endl;
    exit(1);
  }
  double p = atof(argv[2]);
  std::string input(argv[1]);
  fs_bns fs = fs_bns(input, p);
  fs.select();

  std::ifstream file(input.data());
  if (file) {
    std::string line;
    while (file >> line) {
      std::cout << fs.filter(line) << std::endl;
    }
    file.close();
  }
  else {
    std::cerr << "Error while opening training file." << std::endl;
    exit(1);
  }
  return 0;
}
