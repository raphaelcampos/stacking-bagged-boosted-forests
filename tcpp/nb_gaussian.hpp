#ifndef NB_GAUSSIAN_HPP__
#define NB_GAUSSIAN_HPP__

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
#include <limits>

#include "similarity.h"
#include "supervised_classifier.hpp"
#include "utils.hpp"
#include "scores.hpp"

class nb_gaussian : public virtual SupervisedClassifier {
 public:
  nb_gaussian(unsigned int round=0, double a=1.0) :
    SupervisedClassifier(round), alpha_(a) {}

  virtual bool parse_train_line(const std::string &l);
  virtual void parse_test_line(const std::string &l);
  virtual bool check_train_line(const std::string &l) const;
  
  virtual void reset_model() {   
    DF.clear();
    sumTF.clear();
    squared_sumTF.clear();
    count_terms.clear();
  }

  void set_alpha(double a) { alpha_ = a; }
  double get_alpha() { return alpha_; }
  virtual ~nb_gaussian() {}

 protected:
  virtual double apriori(const std::string &doc_class) {
    return (df(doc_class) + alpha_) /
           (get_total_docs() + alpha_*static_cast<double>(classes_size()));
  }
  

  virtual double term_conditional(const int &term_id, const double &term_freq,
                          const std::string &doc_class) {
      std::string idx = Utils::get_index(term_id, doc_class);

      if(get_count(idx) == 0) return 1.0;

      double expected_value = sum_tf(idx) / (get_count(idx));
      double variance = ((squared_sum_tf(idx) - ((sum_tf(idx)*sum_tf(idx)) / (get_count(idx)))) 
                                                  / (get_count(idx)));
      double diff = (term_freq-expected_value);
      
//      if (variance == 0.0) variance = 1.0;
           
      double normal_a = (variance > 0 ) ? 1.0/sqrt(2*M_PI*variance) : 0.0;
      double normal_exp = (variance > 0.0) ? ((-0.5) * (diff * diff) / variance) : 0.0;
      normal_exp = exp(normal_exp);

      double ret_value = normal_a * normal_exp;      
      return ret_value;
      
  }

 
  double sum_tf(const std::string &k) {return Utils::get_value(sumTF,k);}
  double squared_sum_tf(const std::string &k) {return Utils::get_value(squared_sumTF, k);}
  double df(const std::string &k) { return Utils::get_value(DF, k); }
  unsigned int get_count(const std::string &k) {return Utils::get_value(count_terms, k);}

  std::map<std::string, double> DF;
  std::map<std::string, double> sumTF;
  std::map<std::string, double> squared_sumTF;
  std::map<std::string, unsigned int> count_terms;

  double alpha_;
};

bool nb_gaussian::check_train_line (const std::string &line) const {
  std::vector<std::string> tokens;
  Utils::string_tokenize(line, tokens, ";");
  // input format: doc_id;class_name;{term_id;tf}+
  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return false;
  return true;
}

bool nb_gaussian::parse_train_line (const std::string &line) {
  std::vector<std::string> tokens;
  Utils::string_tokenize(line, tokens, ";");
  // input format: doc_id;CLASS=class_name;{term_id;tf}+
  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return false;

  std::string id = tokens[0];

  std::string doc_class = tokens[1];
  classes_add(doc_class);

  DF[doc_class]++;

  for (size_t i = 2; i < tokens.size()-1; i+=2) {
    double tf = atof(tokens[i+1].data());
    
    int term_id = atoi(tokens[i].data());
    vocabulary_add(term_id);

    std::string idx = Utils::get_index(term_id, doc_class);
    
    if(sumTF.find(idx) == sumTF.end())
      sumTF[idx] = tf;
    else
      sumTF[idx] += tf;
    
    if(squared_sumTF.find(idx) == squared_sumTF.end())
      squared_sumTF[idx] = (tf*tf);
    else
      squared_sumTF[idx] += (tf*tf);

    if(count_terms.find(idx) == count_terms.end())
      count_terms[idx] = 1;
    else
      count_terms[idx]++;

  }

  return true;
}

void nb_gaussian::parse_test_line(const std::string &line) {
  std::vector<std::string> tokens;
  Utils::string_tokenize(line, tokens, ";");
  // input format: doc_id;class_name;{term_id;tf}+
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
      double tf = atof(tokens[i+1].data());
      double val = term_conditional(term_id, tf, cur_class);
      if(val != 0.0) 
        probCond += log(val);
    }

    double sim = aPriori + probCond;
    similarities.add(cur_class, sim);
    normalizer = normalizer + sim;
    ++cIt;
  }
  get_outputer()->output(similarities);
}
#endif
