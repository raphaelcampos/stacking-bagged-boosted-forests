#ifndef KFOLD_HPP__
#define KFOLD_HPP__

#include <typeinfo>

#include <unistd.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

#include "validator.hpp"

class KFold : public Validator {
 public:
  KFold(const std::string &fn, unsigned int r,
        SupervisedClassifier *c = NULL) :
    Validator(fn, r, c) {}

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
      std::cerr << "Error while opening training file." << std::endl;
      return;
    }

    std::vector<unsigned int> indices; indices.reserve(offsets.size());
    kfold(Validator::replications, offsets, indices);

    for (unsigned int i = 0; i < Validator::replications; i++) {
      std::ifstream source_file(Validator::input.data());
      std::stringstream ss_trn; ss_trn << ".train_" << i << ".tmp";
      std::stringstream ss_tst; ss_tst << ".test_"  << i << ".tmp";
      std::ofstream trn(ss_trn.str().data());
      std::ofstream tst(ss_tst.str().data());

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
        if (indices[j] == i) {
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
        std::stringstream ss;
        ss << "#" << i;
        if (cc->get_outputer() == NULL) {
          std::cout << ss.str() << std::endl;
        }
        else {
          cc->get_outputer()->output(ss.str());
        }
        std::cerr << "[KFOLD] Starting iteration " << i << std::endl;
        std::cerr << "[KFOLD] Training..." << std::endl;
        cc->train(ss_trn.str());
        std::cerr << "[KFOLD] Testing..." << std::endl;
        cc->test(ss_tst.str());
      }
      unlink(ss_trn.str().data());
      unlink(ss_tst.str().data());
    }
  }

 private:

  void shuffle(std::vector<unsigned int> &a) {
    int idx = static_cast<int>(a.size());
    while (idx > 1) {
      int rnd_idx = Utils::random(idx);		
      idx--;					
      unsigned int tmp = a[idx];	
      a[idx] = a[rnd_idx];
      a[rnd_idx] = tmp;
    }
  }

  void kfold(const unsigned int k,
             const std::vector<unsigned int> &a,
             std::vector<unsigned int> &out) {
    out.clear();
    out.resize(a.size());
    double inc = static_cast<double> (k) /
                 static_cast<double> (a.size());

    for (size_t i = 0; i < a.size(); i++)
      out[i] = ceil((i + 0.9) * inc) - 1;

    shuffle(out);
  }


};

#endif
