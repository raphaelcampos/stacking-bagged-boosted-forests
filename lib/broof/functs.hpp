#ifndef FUNCTS_HPP__
#define FUNCTS_HPP__

#include <string>
#include <vector>
#include <map>
#include <cstdlib>
#include <iostream>

typedef std::map< std::string, std::map< std::string, long double > > Matrix;

inline std::string get_idx(const std::string &a, const std::string &b) {
  return a + "-" + b;
}

unsigned int get_value(const std::map<std::string, unsigned int> &h,
                       const std::string &k) {
  std::map<std::string, unsigned int>::const_iterator it = h.find(k);
  if (it != h.end()) return it->second;
  return 0;
}

inline void stringTokenize(const std::string &str, std::vector<std::string> &tokens,
                           const std::string &delimiters = " ") {
  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  std::string::size_type pos = str.find_first_of(delimiters, lastPos);
  while (std::string::npos != pos || std::string::npos != lastPos) {
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    lastPos = str.find_first_not_of(delimiters, pos);
    pos = str.find_first_of(delimiters, lastPos);
  }
}

void panic(const char *msg) {
  std::cerr << "[ERROR] " << msg << std::endl;
  exit(1);
}

#endif
