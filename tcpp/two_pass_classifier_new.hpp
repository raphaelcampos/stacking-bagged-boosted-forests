#ifndef TWO_PASS_CLASSIFIER_HPP_H__
#define TWO_PASS_CLASSIFIER_HPP_H__

#include <unistd.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <vector>
#include <set>
#include <queue>
#include <utility>
#include <fstream>
#include <stdlib.h>

#include "supervised_classifier.hpp"
#include "temporal_classifier.hpp"
#include "fst_pass_classifier.hpp"
#include "random_subsampling.hpp" 
#include "outputer.hpp"

class TwoPassOutputer : public FileOutputer {
 public:
  TwoPassOutputer(const std::string &fn) :
    FileOutputer(fn) {}

  virtual void output(Scores<double> &sco, const double n=1.0) {
    std::stringstream ss;
    if (!sco.empty()) {
      Similarity<double> s = sco.top();
      ss << s.class_name << ";" << (s.similarity/n);
      sco.pop();
    }
    ss << std::endl;

    #pragma omp critical (output)
    {
      #pragma omp flush(os_)
      os_ << ss.str() << std::flush;
    }
  }

  virtual ~TwoPassOutputer() {}
};

class TwoPassClassifier : public SupervisedClassifier {
 public:
  TwoPassClassifier(SupervisedClassifier *fst,
                    TemporalClassifier *tmp, unsigned int r=0) :
    SupervisedClassifier(r),
    fst_(FirstPassClassifier(fst)), tmp_cl_(tmp),
    twf_(".estimatedTWF"),
    trn_a_(".trn1"), trn_b_(".trn2"), inter_(".interm") {}

  virtual bool parse_train_line(const std::string &l);
  virtual void parse_test_line(const std::string &l);
  virtual bool check_train_line(const std::string &l) const;

  virtual void reset_model() {
    fst_.reset_model();
    tmp_cl_->reset_model();
  }

  virtual void train(const std::string &train);
  virtual void test(const std::string &test);

  virtual ~TwoPassClassifier() {}

 private:
  FirstPassClassifier fst_;
  TemporalClassifier *tmp_cl_;

  std::string train_fn;

  void estimate_twf();
  std::string twf_;

  const std::string trn_a_;
  const std::string trn_b_;
  const std::string inter_;
};

bool TwoPassClassifier::check_train_line(const std::string &line) const {
  return fst_.check_train_line(line);
}

bool TwoPassClassifier::parse_train_line(const std::string &line) {
 return fst_.parse_train_line(line);
}

void TwoPassClassifier::train(const std::string &trn) {
  fst_.set_outputer(new FileOutputer(inter_));
  fst_.reset();
  RandomSubsampling rs(trn, trn_a_, trn_b_, 0.3, 1, &fst_);
  rs.do_validation();
  train_fn = trn;
}

void TwoPassClassifier::estimate_twf() {
  std::ifstream file(inter_.data());
  if (file) {
    std::map<int,int> histogram;
    std::string line;
    int normalizer = -9999.99;
    while (std::getline(file, line)) {
      std::vector<std::string> tokens;
      // format: doc_id CLASS=true_class-true_year CLASS=pred_class-pred_year:value
      Utils::string_tokenize(line, tokens, " ");
      if (tokens.size() > 2) {
        std::string real = tokens[1];
        std::string pred = tokens[2];
        tokens.clear();
        Utils::string_tokenize(real, tokens, "-");
        int p_r = atoi(tokens[1].data());
        tokens.clear();

        Utils::string_tokenize(pred, tokens, "-");
        pred = tokens[1];
        tokens.clear();

        Utils::string_tokenize(pred, tokens, ":");
        int p   = atoi(tokens[0].data());  // predicted time point

        int diff = p_r - p;
        histogram[diff]++;
        std::map<int, int>::iterator hIt = histogram.find(diff);
        if (normalizer < hIt->second) normalizer = hIt->second;
      }
    }
    file.close();
    unlink(inter_.data());

    Outputer *out = new TwoPassOutputer(twf_);
    std::map<int, int>::iterator it = histogram.begin();
    while (it != histogram.end()) {
      std::stringstream ss_temp_dist;
      ss_temp_dist << it->first;
      double w = static_cast<double>(it->second) /
                 static_cast<double>(normalizer);
      Scores<double> sco("", "");
      sco.add(ss_temp_dist.str(), w);
      out->output(sco);
      ++it;
    }
    delete out; out = NULL;
    file.close();
  }
  else {
    std::cerr << "Error while opening file with predictions for TWF ("
              << twf_ << ")." << std::endl;
    exit(1);
  }
}

void TwoPassClassifier::parse_test_line(const std::string &line) {
  std::cerr << "[TwoPass] do not enter here !!!!" << std::endl;
  // stub
 return;
}

void TwoPassClassifier::test(const std::string &test_fn) {
  estimate_twf();
  tmp_cl_->reset();
  tmp_cl_->set_twf_file(twf_);
  tmp_cl_->read_twf(twf_);
  Outputer *tmp = tmp_cl_->get_outputer();
  if (get_outputer() != NULL) tmp_cl_->set_outputer(get_outputer());
  tmp_cl_->train(train_fn);
  tmp_cl_->test(test_fn);
  if (get_outputer() != NULL) tmp_cl_->set_outputer(tmp);
  tmp = NULL;
}
#endif
