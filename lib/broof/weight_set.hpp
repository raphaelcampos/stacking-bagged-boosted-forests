#ifndef WEIGHT_SET_HPP__
#define WEIGHT_SET_HPP__

#include <string>
#include <unordered_map>

class WeightSet {
 public:
  WeightSet() {}

  double get(std::string i) {
    auto it = w_.find(i);
    if (it == w_.end()) return 1.0;
    return it->second;
  }

  double set(std::string i, double w) {
    return w_[i] = w;
  }

  bool empty() { return w_.size(); }

 private:
  std::unordered_map<std::string, double> w_;

};

#endif
