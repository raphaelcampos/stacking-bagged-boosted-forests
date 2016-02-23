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

  int size() { return w_.size(); }

  std::map<std::string, double>::const_iterator begin() const
  {
    w_.begin();
  }

  std::map<std::string, double>::const_iterator end() const
  {
    w_.end();
  }

  void normalize(){
    double norm = 0;
    for (std::map<std::string, double>::const_iterator it = w_.begin(); it != w_.end(); ++it)
    {
      norm += it->second;
    }
    for (std::map<std::string, double>::iterator it = w_.begin(); it != w_.end(); ++it)
    {
      it->second /= norm;
    }
    //init_prob /= norm;
  }

 private:
  std::map<std::string, double> w_;
  float init_prob;

};

#endif