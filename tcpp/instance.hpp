#ifndef INSTANCE_HPP__
#define INSTANCE_HPP__

#include <set>

#include "feature.hpp"

class Instance {
 public:
  Instance(const std::string &id, const std::string &cl = "")
    : id_(id), class_(cl) {}

  void add_attribute(const std::string &k, const double v) {
    attributes_.insert(Feature(k, v));
  }
 private:
  std::string id_;
  std::string class_;
  std::set<Feature> attributes_;
};

#endif
