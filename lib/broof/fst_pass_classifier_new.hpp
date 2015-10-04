#ifndef FST_PASS_CLASSIFIER_HPP__
#define FST_PASS_CLASSIFIER_HPP__

#include <typeinfo>

#include <unistd.h>
#include <fstream>
#include <iostream>
#include <string>
#include <set>
#include <queue>
#include <cstdlib>

#include "supervised_classifier.hpp"

class FirstPassClassifier : public SupervisedClassifier {
 public:
  FirstPassClassifier(SupervisedClassifier *c, unsigned int r=0) :
    SupervisedClassifier(r), cc(c) {}
  virtual ~FirstPassClassifier() {};

  virtual void train(const std::string &train_fn) {
    std::string train_fn_tmp = train_fn + ".intermediate";
    std::ifstream file(train_fn.data());
    std::ofstream of(train_fn_tmp.data());
    if (file) {
      std::string line;
      while (file >> line) of << change_line(line) << std::endl;
      of.close();
      file.close();
      cc->train(train_fn_tmp);
    }
    else {
      std::cerr << "Error while opening training file." << std::endl;
      exit(1);
    }
    unlink(train_fn_tmp.data());
  }

  virtual void test(const std::string &test_fn) {
    std::string test_fn_tmp = test_fn + ".intermediate";
    std::ifstream file(test_fn.data());
    std::ofstream of(test_fn_tmp.data());
    if (file) {
      std::string line;
      while (file >> line) of << change_line(line) << std::endl;
      of.close();
      file.close();
      Outputer *tmp = cc->get_outputer();
      cc->set_outputer(get_outputer());
      cc->test(test_fn_tmp);
      cc->set_outputer(tmp); tmp = NULL;
    }
    else {
      std::cerr << "Error while opening training file." << std::endl;
      exit(1);
    }
    unlink(test_fn_tmp.data());
  }

  // dummy definitions (this class delegates work)
  virtual bool parse_train_line(const std::string &l) { return true; };
  virtual void parse_test_line(const std::string &l) { return; };

  virtual void reset_model() { cc->reset_model(); }

 protected:
  std::string change_line(const std::string &l) {
    std::vector<std::string> tokens;
    Utils::string_tokenize(l, tokens, ";");
    // input format: doc_id;year;CLASS=class_name;{term_id;tf}+
    if ((tokens.size() < 5) || (tokens.size() % 2 == 0)) return false;

    std::string id = tokens[0];
    std::string year = tokens[1];
    std::string doc_class = tokens[2];

    std::string nl = id + ";" + Utils::get_index(doc_class, year);
    for (unsigned int i = 3; i < tokens.size(); i++) {
      nl = nl + ";" + tokens[i];
    }
    return nl;
  }

  SupervisedClassifier *cc;
};
#endif
