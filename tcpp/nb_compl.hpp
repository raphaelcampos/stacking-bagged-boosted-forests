#ifndef NB_COMPL__
#define NB_COMPL__

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
#include "nb_log.hpp"
#include "utils.hpp"
#include "scores.hpp"

class nb_compl : public nb_log {
 public:
  nb_compl(unsigned int round=0, double a=1.0, double l=0.0) : nb_log(round, a, l), alpha_(a), lambda_(l) {}

  virtual void parse_test_line(const std::string &l);

  virtual ~nb_compl() {}

 protected:
  virtual double apriori(const std::string &doc_class) {
    return static_cast<double>(get_total_docs() - df(doc_class) + alpha_) /
           static_cast<double>(get_total_docs() + alpha_ * static_cast<double>(classes_size()));
  }

  virtual double term_conditional(const int &term_id,
                          const std::string &doc_class) {
    return static_cast<double>(nt(term_id) - tf(term_id, doc_class) + alpha_) /
           static_cast<double>(total_terms - sum_tf(doc_class)
                                 + alpha_ * static_cast<double>(vocabulary_size()));

  }
  double alpha_;
  double lambda_;
};

void nb_compl::parse_test_line(const std::string &line) {
  std::vector<std::string> tokens;
  Utils::string_tokenize(line, tokens, ";");
  // input format: doc_id;CLASS=class_name;{term_id;tf}+
  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return;

  double normalizer = 0.0;

  std::string id = tokens[0];

  Scores<double> similarities(id, tokens[1]);

  std::set<std::string>::const_iterator cIt = classes_begin();
  while (cIt != classes_end()) {
    std::string cur_class = *cIt;

    double aPriori = log(apriori(cur_class));

    double probCond = 0.0;
    for (size_t i = 2; i < tokens.size()-1; i+=2) {
      int term_id = atoi(tokens[i].data());
      unsigned int tf = atoi(tokens[i+1].data());
      double val = term_conditional(term_id, cur_class);
      double num_t = nt(term_id);
      
      // jelinek-mercer smoothing (linear interpolation)
      
      val = lambda_ * ( num_t  / total_terms) + (1.0-lambda_) * val;

      probCond += tf * log(val);
     
    }

    double sim = aPriori - probCond;
    similarities.add(cur_class, sim);
    normalizer = normalizer + sim;
    ++cIt;
  }

  get_outputer()->output(similarities, normalizer);
}
#endif
