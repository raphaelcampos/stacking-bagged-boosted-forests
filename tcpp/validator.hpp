#ifndef VALIDATOR_HPP__
#define VALIDATOR_HPP__

#include <cstdlib>
#include "utils.hpp"

#include "supervised_classifier.hpp"

class Validator {
 public:
  Validator(const std::string &fn,
            unsigned int r,
            SupervisedClassifier *c = NULL) :
    cc(c), input(fn), replications(r) { srand48(12345); }

  SupervisedClassifier *get_classifier() { return cc; }
  void set_classifier(SupervisedClassifier *c) {
    cc = c;
  }

  virtual void do_validation() = 0;

 protected:
  SupervisedClassifier *cc;
  std::string input;
  unsigned int replications;
};

#endif
