#ifndef WEIGHT_SET_HPP__
#define WEIGHT_SET_HPP__

#include <map>

class WeightSet {
 public:
  WeightSet() {}

  double get(std::string i) {
    if (w_.find(i) == w_.end()) return 1.0;
    return w_[i];
  }

  double set(std::string i, double w) {
    return w_[i] = w;
  }

  bool empty() { return w_.size(); }

 private:
  std::map<std::string, double> w_;

};

#endif
