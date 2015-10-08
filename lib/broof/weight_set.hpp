#ifndef WEIGHT_SET_HPP__
#define WEIGHT_SET_HPP__

#include <map>

class WeightSet {
 public:
  WeightSet(float init_prob=1.0) : init_prob(init_prob) {}

  double get(std::string i) {
    if (w_.find(i) == w_.end()) return init_prob;
    return w_[i];
  }

  double set(std::string i, double w) {
    return w_[i] = w;
  }

  bool empty() { return w_.size(); }

  std::map<std::string, double>::const_iterator begin() const
  {
    w_.begin();
  }

  std::map<std::string, double>::const_iterator end() const
  {
    w_.end();
  }

 private:
  std::map<std::string, double> w_;
  float init_prob;

};

#endif
