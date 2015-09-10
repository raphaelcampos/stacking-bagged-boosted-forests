#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <cmath>

#include "utils.hpp"

void convert(const char *input, const char *trn=NULL) {
  std::map<int, int> df;
  int N = 0;
  if (trn) {
    std::ifstream file(input);
    if (file) {
      std::string ln;
      while (std::getline(file, ln)) {
        std::vector<std::string> tokens;
        Utils::string_tokenize(ln, tokens, ";");
        // docId;class;t1;f1;t2;f2...
        std::map<int, int> tfs;
        for (unsigned int i = 2; i < tokens.size()-1; i += 2) {
          df[atoi(tokens[i].data())]++;
        }
        N++;
      }
    }
  }

  std::ifstream file(input);
  if (file) {
    std::string ln;
    while (std::getline(file, ln)) {
      std::vector<std::string> tokens;
      Utils::string_tokenize(ln, tokens, ";");
      // docId;class;t1;f1;t2;f2...
      std::map<int, double> tfs;
      for (unsigned int i = 2; i < tokens.size()-1; i += 2) {
        int t = atoi(tokens[i].data());
        double tf  = (trn) ? 1.0 + log(atof(tokens[i+1].c_str())) : atof(tokens[i+1].c_str());
        double idf = (trn) ? log((static_cast<double>(N) + 1.0) / (static_cast<double>((df[t] + 1.0)))) : 1;
        tfs[t] = tf*idf;
      }

      std::stringstream data;
      data << tokens[1].data();
      std::map<int, double>::const_iterator it = tfs.begin();
      while (it != tfs.end()) {
        data << " " << (it->first+1) << ":" << it->second;
        ++it;
      }
      std::cout << data.str() << std::endl;
    }
    file.close();
  }
  else {
    std::cerr << "Failed to open input file." << std::endl;
    exit(1);
  }

}

int main(int argc, char **argv) {
  char *input;
  if (!(input = argv[1])) {
    std::cerr << "Usage: " << argv[0] << " [input] [opt:training_for_calc_tfidf]" << std::endl;
    exit(1);
  }

  char *trn = NULL;
  if (argc == 3) trn = argv[2];
  convert(input, trn);
  return 0;
}
