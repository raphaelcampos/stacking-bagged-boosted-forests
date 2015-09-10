#ifndef SIMILARITY_H__
#define SIMILARITY_H__

#include <string>

template<typename T>
class Similarity {
 public:
  Similarity(const std::string &c, const T &s) :
    class_name(c), similarity(s) {}

  bool operator< (const Similarity &rhs) const {
    return similarity < rhs.similarity;
  }

  std::string class_name;
  T similarity;
};

#endif
