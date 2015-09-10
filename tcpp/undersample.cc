#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <cmath>

#include "utils.hpp"

bool pick(double ratio) {
  return (ratio > 0.0 && ratio < 1.0) ? (((double) rand() / (RAND_MAX)) < ratio) : true;
}

void sample(const char *input, double ratio) {
  std::map<std::string, std::set<int> > samples;
  std::ifstream file(input);
  if (file) {
    std::string ln;
    while (std::getline(file, ln)) {
      std::vector<std::string> tokens;
      Utils::string_tokenize(ln, tokens, ";");
      // docId;class;t1;f1;t2;f2...
      int id = atoi(tokens[0].data());
      std::string cl = tokens[1];
      samples[cl].insert(id);
    }
    file.close();
  }
  else {
    std::cerr << "Failed to open input file." << std::endl;
    exit(1);
  }

  std::set<int> sampled_ids;
  std::map<std::string, std::set<int> >::const_iterator cit = samples.begin();
  while (cit != samples.end()) {
    std::set<int>::const_iterator dit = cit->second.begin();
    while (dit != cit->second.end()) {
      if (pick(ratio)) sampled_ids.insert(*dit);
      ++dit;
    }
    ++cit;
  }

  file.open(input);
  if (file) {
    std::string ln;
    while (std::getline(file, ln)) {
      std::vector<std::string> tokens;
      Utils::string_tokenize(ln, tokens, ";");
      // docId;class;t1;f1;t2;f2...
      int id = atoi(tokens[0].data());
      if (sampled_ids.find(id) != sampled_ids.end()) {
        std::cout << ln << std::endl;
      }
    }
    file.close();
  }
  else {
    std::cerr << "Failed to open input file." << std::endl;
    exit(1);
  }


}

int main(int argc, char **argv) {
  char *input = NULL;
  if (!(input = argv[1])) {
    std::cerr << "Usage: " << argv[0] << " [input] [opt:ratio]" << std::endl;
    exit(1);
  }

  double ratio = -1;
  if (argc == 3) ratio = atof(argv[2]);
  sample(input, ratio);
  return 0;
}
