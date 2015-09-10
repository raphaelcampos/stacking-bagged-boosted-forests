#ifndef NB_LOG_HPP__
#define NB_LOG_HPP__

#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <vector>
#include <set>
#include <queue>
#include <utility>
#include <fstream>
#include <cstdlib>

#include "similarity.h"
#include "supervised_classifier.hpp"
#include "utils.hpp"
#include "scores.hpp"

class nb_log : public virtual SupervisedClassifier {
 public:
  nb_log(unsigned int round=0, double a=1.0, double l=0.0, bool u=false) :
    SupervisedClassifier(round), total_terms(0), alpha_(a), lambda_(l), unif_prior_(u) {}

  virtual bool parse_train_line(const std::string &l);
  virtual void parse_test_line(const std::string &l);
  virtual bool check_train_line(const std::string &l) const;
  
  virtual void reset_model() {
    TF.clear();
    DF.clear();
    sumTF.clear();
    n_t.clear();
    total_terms = 0;
  }

  void set_alpha(double a) { alpha_ = a; }
  double get_alpha() { return alpha_; }
  virtual ~nb_log() {}

 protected:
  virtual double apriori(const std::string &doc_class) {
    return static_cast<double>(df(doc_class) + alpha_) /
           static_cast<double>(get_total_docs() + alpha_*static_cast<double>(classes_size()));
  }

  virtual double term_conditional(const int &term_id,
                          const std::string &doc_class) {
  return static_cast<double>(tf(term_id, doc_class) + alpha_) /
         static_cast<double>(sum_tf(doc_class) + alpha_*static_cast<double>(vocabulary_size()));
  }

  unsigned int tf(const int term_id,
                  const std::string &doc_class) {
    return Utils::get_value(TF, Utils::get_index(term_id, doc_class));
  }
  double sum_tf(const std::string &k) {return Utils::get_value(sumTF,k);}
  double df(const std::string &k) { return Utils::get_value(DF, k); }

  double nt(const int &k) { return Utils::get_value(n_t, k); }
  double get_total_terms() { return total_terms; }

  std::map<std::string, double> TF;
  std::map<std::string, double> DF;
  std::map<std::string, double> sumTF;
  std::map<int, double> n_t;

  double total_terms;
  double alpha_;
  double lambda_;
  bool unif_prior_;
};

bool nb_log::check_train_line (const std::string &line) const {
  std::vector<std::string> tokens; tokens.reserve(100);
  Utils::string_tokenize(line, tokens, ";");
  // input format: doc_id;class_name;{term_id;tf}+
  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return false;
  return true;
}

bool nb_log::parse_train_line (const std::string &line) {
  std::vector<std::string> tokens; tokens.reserve(100);
  Utils::string_tokenize(line, tokens, ";");
  // input format: doc_id;CLASS=class_name;{term_id;tf}+
  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return false;

  std::string id = tokens[0];

  std::string doc_class = tokens[1];
  classes_add(doc_class);

  DF[doc_class]++;

  for (size_t i = 2; i < tokens.size()-1; i+=2) {
    unsigned int tf = atoi(tokens[i+1].data());
    double term_id = atof(tokens[i].data());
    vocabulary_add(term_id);

    std::string index = Utils::get_index(term_id, doc_class);
    TF[index] += tf;
    sumTF[doc_class] += tf;

    n_t[term_id] += tf;
    total_terms += tf;
  }

  return true;
}

void nb_log::parse_test_line(const std::string &line) {
  std::vector<std::string> tokens; tokens.reserve(100);
  Utils::string_tokenize(line, tokens, ";");
  // input format: doc_id;class_name;{term_id;tf}+
  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return;

  double normalizer = 0.0;

  std::string id = tokens[0];

  Scores<double> similarities(id, tokens[1]);
  std::set<std::string>::const_iterator cIt = classes_begin();
  while (cIt != classes_end()) {
    std::string cur_class = *cIt;
    double aPriori = unif_prior_ ? 0.0 : log(apriori(cur_class));

    double probCond = 0.0;
    for (size_t i = 2; i < tokens.size()-1; i+=2) {
      int term_id = atoi(tokens[i].data());
      double tf = atof(tokens[i+1].data());
      double val = term_conditional(term_id, cur_class);
      double num_t = nt(term_id);

      // jelinek-mercer smoothing (linear interpolation)
      val = lambda_ * ( num_t  / total_terms) + (1.0-lambda_) * val;

      probCond += tf * log(val);
    }

    double sim = aPriori + probCond;
    similarities.add(cur_class, sim);
    normalizer = normalizer + sim;
    ++cIt;
  }
  get_outputer()->output(similarities);
}
#endif
