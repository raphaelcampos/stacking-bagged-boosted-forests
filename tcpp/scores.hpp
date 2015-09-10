#ifndef SCORES_HPP__
#define SCORES_HPP__

#include <fstream>
#include <iostream>
#include <string>
#include <set>
#include <queue>
#include <cstdlib>

#include "similarity.h"

class SupervisedClassifier;
class Outputer;

template<typename T>
class Scores {
 friend class SupervisedClassifier;
 friend class Outputer;
 public:
  Scores(const std::string &id, const std::string &real_class) :
  id_(id), real_class_(real_class) {}

  std::string get_id() const { return id_; }
  std::string get_class() const { return real_class_; }
  bool empty() const { return scores_.empty(); }

  void add(const std::string &c, T s) {scores_.push(Similarity<T>(c, s));}
  void pop() { scores_.pop(); }
  Similarity<T> top() const { return scores_.top(); }

 private:
  std::string id_;
  std::string real_class_;
  std::priority_queue<Similarity<T>,
                      std::vector<Similarity<T> > > scores_;
};

#endif
