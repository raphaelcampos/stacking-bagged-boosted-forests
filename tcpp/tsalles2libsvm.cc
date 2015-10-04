#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sstream>

#include "utils.hpp"

void convert(const char *input) {
  std::ifstream file(input);
  if (file) {
    std::string ln;
    while (std::getline(file, ln)) {
      std::vector<std::string> tokens;
      Utils::string_tokenize(ln, tokens, ";");
      // docId;tp;class=C;t1;f1;t2;f2...
      std::map<int, int> tfs;
      for (unsigned int i = 2; i < tokens.size(); i += 2) {
        tfs[atoi(tokens[i].data())] = atoi(tokens[i+1].data());
      }
      std::vector<std::string> tmp;
      std::string cl = tokens[1];
      if (tokens[1].size() > 6) cl.replace(0, 6, "");

      std::stringstream data;
      data << cl;
      std::map<int, int>::const_iterator it = tfs.begin();
      while (it != tfs.end()) {
        data << " " << (it->first + 1) << ":" << it->second;
        ++it;
      }
      std::cout << data.str() << std::endl;
    }
  }
  else {
    std::cerr << "Failed to open input file." << std::endl;
    exit(1);
  }

}

int main(int argc, char **argv) {
  char *input;
  if (!(input = argv[1])) {
    std::cerr << "Usage: " << argv[0] << " [input]" << std::endl;
    exit(1);
  }
  convert(input);
  return 0;
}
