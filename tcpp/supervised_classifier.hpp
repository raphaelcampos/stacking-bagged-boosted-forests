#ifndef SUPERVISED_CLASSIFIER_HPP__
#define SUPERVISED_CLASSIFIER_HPP__

#include <fstream>
#include <iostream>
#include <string>
#include <set>
#include <queue>
#include <cstdlib>
#include <cstdio>

#include "utils.hpp"
#include "abstract_feature_selector.hpp"
#include "scores.hpp"
#include "outputer.hpp"

enum DistType {COSINE, L2, L1};

class SupervisedClassifier {
 public:

  SupervisedClassifier(unsigned int r=0)
    : total_docs(0), fs(NULL), out(NULL), round(r), out_file(""), raw(false), dist_type(COSINE) {}

  void set_round(unsigned int i) { round = i; }
  unsigned int get_round() const { return round; }

  void use_raw_weights() { raw=true; }
  void use_computed_weights() { raw=false; }
  bool is_raw_weights() { return raw; }

  void set_distance(DistType d) { dist_type = d; }

  void set_feature_selector(AbstractFeatureSelector *f) { fs = f; }

  virtual void set_outputer(Outputer *o) { out = o; }
  Outputer *get_outputer() const { return out; }
  void set_output_file(const std::string &o) { out_file = o; }

  virtual void train(const std::string &train_fn);
  virtual void test(const std::string &new_test);

  virtual bool parse_train_line(const std::string &l) = 0;
  virtual void parse_test_line(const std::string &l) = 0;
  virtual void reset_model() = 0;

  virtual bool check_train_line(const std::string &l) const
    { return true; }
  virtual bool check_test_line(const std::string &l) const
    { return check_train_line(l); }

  void reset() {
    total_docs = 0;
    classes.clear();
    vocabulary.clear();
    reset_model();
  };

  virtual ~SupervisedClassifier() {};

  size_t classes_size() const
    { return classes.size(); }
  void classes_add(const std::string &v)
    { classes.insert(v); }
  std::set<std::string>::const_iterator classes_begin()
    { return classes.begin(); }
  std::set<std::string>::const_iterator classes_end()
    { return classes.end(); }

  void vocabulary_add(const int &c)
    { vocabulary.insert(c); }
  std::set<int>::const_iterator vocabulary_begin()
    { return vocabulary.begin(); }
  std::set<int>::const_iterator vocabulary_end()
    { return vocabulary.end(); }
  size_t vocabulary_size() const
    { return vocabulary.size(); }

  unsigned int get_total_docs() const { return total_docs; }

 protected:

  std::set<std::string> classes;
  std::set<int> vocabulary;
  unsigned int total_docs;
  AbstractFeatureSelector *fs;
  Outputer *out;
  unsigned int round;
  std::string out_file;
  bool raw;
  DistType dist_type;

};

void SupervisedClassifier::train(const std::string &train_fn) {
  std::ifstream file(train_fn.data());
  if (file) {
    std::cerr << "[SUPERVISED CLASSIFIER] Training..." << std::endl;
    if (fs != NULL) fs->select(train_fn);
    std::string line;
    while (std::getline(file, line)) {
      if (fs != NULL) line = fs->filter(line);
      if (parse_train_line(line)) total_docs++;
    }
    file.close();
  }
  else {
    std::cerr << "Error while opening training file." << std::endl;
    exit(1);
  }
}

void SupervisedClassifier::test(const std::string &test_fn) {
  std::ifstream file(test_fn.data());
  if (file) {
    bool free_out = false;
    if (out == NULL) {
//      if (out_file.empty()) out = new BufferedOutputer(10000);
      if (out_file.empty()) {
        setbuf(stdout, NULL);
        out = new Outputer();
      }
      else out = new FileOutputer(out_file);
      free_out = true;
    }
    std::cerr << "[SUPERVISED CLASSIFIER] Testing..." << std::endl;
    std::string line;
    std::vector<std::string> buffer;
    buffer.reserve(10000);
    while (getline(file, line)) {
      buffer.push_back(line);
      if (buffer.size() == 10000) {
        #pragma omp parallel for num_threads(8) schedule(dynamic)
        for (int i = 0; i < static_cast<int>(buffer.size()); i++) {
          parse_test_line(buffer[i]);
        }
        buffer.clear();
      }
    }
    if (!buffer.empty()) {
      #pragma omp parallel for num_threads(8) schedule(dynamic)
      for (int i = 0; i < static_cast<int>(buffer.size()); i++) {
        parse_test_line(buffer[i]);
      }
    }
    file.close();
    if (free_out) {
      delete out; out = NULL;
    }
  }
  else {
    std::cerr << "Error while opening test file." << std::endl;
    exit(1);
  }
}
#endif
