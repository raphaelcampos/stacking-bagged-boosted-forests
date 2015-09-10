#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "generator.hpp"
#include "utils.hpp"

int main(int argc, char **argv) {
  const char *input = argv[1];
  bool use_avg = (argv[2] == NULL);

  if (!input) {
    std::cerr << "Gimme an input and a give u the heaven." << std::endl;
    exit(1);
  }

  if (!argv[2]) {
    std::cerr << "I'll use the avg class size." << std::endl;
  }

  Smote smote(std::string(input), std::string("bla"), 5);

  std::ifstream file(input);
  if (file) {
    std::string ln;
    std::map<std::string, unsigned int> sizes;
    while (std::getline(file, ln)) {
      std::vector<std::string> tokens;
      Utils::string_tokenize(ln, tokens, ";");
      std::string tp = tokens[1];
      std::string cl = tokens[2];
      sizes[Utils::get_index(cl, tp)]++;
      std::cout << ln << std::endl;
    }
    file.close();

    unsigned int max = 0, min = 99999, sum = 0;
    std::map<std::string, unsigned int>::iterator it = sizes.begin();
    while (it != sizes.end()) {
      if (it->second > max) max = it->second;
      if (it->second < min) min = it->second;
      sum += it->second;
      ++it;
    }

    unsigned int avg = ( (use_avg) ? (min + max)/2 : atoi(argv[2]) ); //sum / sizes.size();
    std::cerr << "b=" << avg << std::endl;
    it = sizes.begin();
    while (it != sizes.end()) {
      std::vector<std::string> tokens;
      Utils::string_tokenize(it->first, tokens, "-");
      std::string cl = tokens[0];
      unsigned int tp = atoi(tokens[1].data());
      unsigned int r = (it->second < avg ? (avg - it->second) : 0);
      for (unsigned int i = 0; i < r; i++) {
        std::string smo = smote.generate_sample(cl, tp, tp);
        if (smo.empty()) std::cout << ln << std::endl;
        else std::cout << smo << std::endl;
      }
      if (it->second < avg) it->second += r;

      ++it;
    }

    it = sizes.begin();
    while (it != sizes.end()) {
      std::cerr << " > cl=" << it->first << " remain=" << it->second << std::endl;
      ++it;
    }
  }
  else {
    std::cerr << "Unable to open the input data file." << std::endl;
    exit(1);
  }
}
