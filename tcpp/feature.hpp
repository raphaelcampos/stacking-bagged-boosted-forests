#ifndef FEATURE_HPP__
#define FEATURE_HPP__

#include <string>

class Feature {
 public:
  Feature(const std::string &k = "", const double v = 0.0)
    : key_(k), value_(v) {}

  double value() const { return value_; }
  std::string key() const { return key_; }

  bool operator< (const Feature &rhs) const {
    return (key_.compare(rhs.key_) < 0);
  }

  double set_value(const double v) { value_ = v; }

 private:
  std::string key_;
  double value_;
};

#endif
