#ifndef RANDOM_SUBSAMPLING_HPP__
#define RANDOM_SUBSAMPLING_HPP__

#include <typeinfo>

#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

#include "validator.hpp"

class RandomSubsampling : public Validator {
 public:
  RandomSubsampling(const std::string &fn,
                    const std::string &sub_a,
                    const std::string &sub_b, double p,
                    unsigned int r,
                    SupervisedClassifier *c = NULL) :
    Validator(fn, r, c), fn_a(sub_a), fn_b(sub_b), percentage(p) {}

  virtual void do_validation() {
    if (cc == NULL) {
      std::cerr << "NULL classifier. Aborting." << std::endl;
      return;
    }

    unsigned int total = 0;
    std::vector<unsigned int> offsets; offsets.reserve(2048);
    std::ifstream file(Validator::input.data());
    if (file) {
      std::string line;
      unsigned int of = file.tellg();
      while (file >> line) {
        if (cc->check_train_line(line)) {
          offsets.push_back(of);
          total++;
        }
        of = file.tellg();
      }
      file.close();
    }
    else {
      std::cerr << "Error while opening training file ("
                << Validator::input << ")."  << std::endl;
      return;
    }

    unsigned int fold_size = total * percentage;

    for (unsigned int i = 0; i < Validator::replications; i++) {
      random_shuffle(offsets.begin(), offsets.end());

      std::ifstream source_file(Validator::input.data());
      std::ofstream trn(fn_a.data());
      std::ofstream tst(fn_b.data());
      if (!trn || !tst) {
        std::cerr << "Failed to create intermediate training/test files."
                  << std::endl;
        return;
      }

      for (unsigned int j = 0; j < total; j++) {
        std::string line;
        source_file.seekg(offsets[j]);
        source_file >> line;
        // test instance
        if (j < fold_size) {
          tst << line << std::endl;
        }
        else { // train instance
          trn << line << std::endl;
        }
      }
      trn.close(); tst.close(); source_file.close();

      if (cc != NULL) {
        cc->reset();
        cc->set_round(i);
        cc->train(fn_a);
        cc->test(fn_b);
      }
      unlink(fn_a.data());
      unlink(fn_b.data());
    }
  }

 private:
  std::string fn_a;
  std::string fn_b;
  double percentage;
};

#endif
