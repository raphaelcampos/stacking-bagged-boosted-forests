#ifndef NB_HPP__
#define NB_HPP__

#include <typeinfo>

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

class nb : public virtual SupervisedClassifier {
 public:
  nb(unsigned int round=0, double a=1.0, double l=0.0, bool u=false) :
    SupervisedClassifier(0),
    total_terms("0.0e+800"), alpha_("0.0e+800"), lambda_("0.0e+800"), unif_prior_(u) {
      alpha_ += static_cast<MyBig>(a);
      lambda_ += static_cast<MyBig>(l);
    }

  virtual bool parse_train_line(const std::string &l);
  virtual void parse_test_line(const std::string &l);
  virtual bool check_train_line(const std::string &l) const;

  virtual void reset_model() {
    TF.clear();
    sumTF.clear();
    DF.clear();
    n_t.clear();
    total_terms = "0.0e+800";
  }

  MyBig get_alpha() { return alpha_; }
  void set_alpha(MyBig a) { MyBig z("0.0e+800"); alpha_ = z + a; }
  
  virtual ~nb() {}

 protected:
  virtual MyBig apriori(const std::string &doc_class) {
    return (static_cast<MyBig>(df(doc_class)) + alpha_) /
           (static_cast<MyBig>(get_total_docs()) + (alpha_ * static_cast<MyBig>(classes_size())));
  }

  virtual MyBig term_conditional(const int &term_id,
                         const std::string &doc_class) {
  return (static_cast<MyBig>(tf(term_id, doc_class)) + alpha_) /
         (static_cast<MyBig>(sum_tf(doc_class)) + (alpha_ * static_cast<MyBig>(vocabulary_size())));
  }

  unsigned int tf(const int term_id,
                  const std::string &doc_class) {
    return Utils::get_value(TF, Utils::get_index(term_id, doc_class));
  }
  unsigned int sum_tf(const std::string &k) {return Utils::get_value(sumTF,k);}
  unsigned int df(const std::string &k) { return Utils::get_value(DF, k); }

  MyBig nt(const int &k) { return Utils::get_value(n_t, k); }
  MyBig get_total_terms() { return total_terms; }

  std::map<std::string, unsigned int> TF;
  std::map<std::string, unsigned int> sumTF;
  std::map<std::string, unsigned int> DF;
  std::map<int, MyBig> n_t;

  MyBig total_terms;
  MyBig alpha_;
  MyBig lambda_;
  bool unif_prior_;
};

bool nb::check_train_line(const std::string &line) const {
  std::vector<std::string> tokens; tokens.reserve(100);
  Utils::string_tokenize(line, tokens, ";");
  // input format: doc_id;CLASS=class_name;{term_id;tf}+
  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return false;
  else return true;
}

bool nb::parse_train_line(const std::string &line) {
  std::vector<std::string> tokens; tokens.reserve(100);
  Utils::string_tokenize(line, tokens, ";");
  // input format: doc_id;CLASS=class_name;{term_id;tf}+
  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return false;

  std::string id = tokens[0];

  // remove CLASS= mark
  std::string doc_class = tokens[1];
  classes_add(doc_class);

  // update document frequency
  DF[doc_class]++;

  // retrieve each term frequency and update occurrencies
  for (size_t i = 2; i < tokens.size()-1; i+=2) {
    unsigned int tf = atoi(tokens[i+1].data());
    int term_id = atoi(tokens[i].data());
    vocabulary_add(term_id);

    std::string index = Utils::get_index(term_id, doc_class);
    TF[index] += tf;
    sumTF[doc_class] += tf;

    if (n_t.find(term_id) == n_t.end())
      n_t[term_id] = (MyBig) tf;
    else n_t[term_id] += (MyBig) tf;
    total_terms += (MyBig) tf;
  }
  return true;
}

void nb::parse_test_line(const std::string &line) {
  std::vector<std::string> tokens; tokens.reserve(100);
  Utils::string_tokenize(line, tokens, ";");
  // input format: doc_id;CLASS=class_name;{term_id;tf}+
  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return;

  MyBig normalizer = "0.0e+800";

  std::string id = tokens[0];

  Scores<MyBig> similarities(id, tokens[1]);

  std::set<std::string>::const_iterator cIt = classes_begin();
  while (cIt != classes_end()) {
    std::string cur_class = *cIt;
    MyBig aPriori = unif_prior_ ? MyBig("1.0") : apriori(cur_class);

    MyBig probCond = "1.0";
    for (size_t i = 2; i < tokens.size()-1; i+=2) {
      int term_id = atoi(tokens[i].data());
      unsigned int tf = atoi(tokens[i+1].data());
      MyBig val = term_conditional(term_id, cur_class);
      MyBig num_t = nt(term_id);

      val = lambda_ * (num_t / total_terms) +
            (static_cast<MyBig>(1.0)-lambda_) * val;

      for (size_t j = 0; j < tf; j++) probCond *= val;
    }

    MyBig sim = aPriori * probCond;
    similarities.add(cur_class, sim);
    normalizer = normalizer + sim;

    ++cIt;
  }
  get_outputer()->output(similarities, normalizer);
}
#endif
