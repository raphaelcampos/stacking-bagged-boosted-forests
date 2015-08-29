#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <map>

  void string_tokenize(const std::string &str,
                       std::vector<std::string> &tokens,
                       const std::string &delimiters = " ") {
    std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    std::string::size_type pos = str.find_first_of(delimiters, lastPos);
    while (std::string::npos != pos || std::string::npos != lastPos) {
      tokens.push_back(str.substr(lastPos, pos - lastPos));
      lastPos = str.find_first_not_of(delimiters, pos);
      pos = str.find_first_of(delimiters, lastPos);
    }
  }

void convert(const char *input) {
  std::ifstream file(input);
  if (file) {
    std::string ln;
    unsigned int id = 1;
    while (std::getline(file, ln)) {
      std::vector<std::string> tokens;
      string_tokenize(ln, tokens, " ");
      // input_format: class t1:f1 t2:f2 ...
      // output_format: docId;class=C;t1;f1;t2;f2...
      std::map<int, double> tfs;
      for (unsigned int i = 1; i < tokens.size(); i++) {
        std::vector<std::string> pair;
        string_tokenize(tokens[i], pair, ":");
        tfs[atoi(pair[0].data())] = atof(pair[1].data());
      }
      
      std::stringstream data;
      data << id << " " << tokens[0];
      std::map<int, double>::const_iterator it = tfs.begin();
      while (it != tfs.end()) {
        data << " " << it->first << " " << it->second;
        ++it;
      }
      std::cout << data.str() << std::endl; id++;
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
