#ifndef TEMPORAL_NB_HPP__
#define TEMPORAL_NB_HPP__

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

#include "supervised_classifier.hpp"
#include "nb.hpp"
#include "temporal_classifier.hpp"

class temporal_nb : public nb, public TemporalClassifier {
 public:
  temporal_nb(const std::string &twf, unsigned int r=0, double a=1.0, double b=1.0, double l=0.0,
     bool u=false, bool w = false, bool gw = false, unsigned int w_sz=0) :
       nb(r, a, l), TemporalClassifier(twf,r,b,w,gw,w_sz), lambda_("0.0e+800"), unif_prior_(u) {
    lambda_ += static_cast<MyBig>(l);
  }

  virtual bool parse_train_line(const std::string &l);
  virtual void parse_test_line(const std::string &l);
  virtual bool check_train_line(const std::string &l) const;

  virtual double twf(unsigned int p_r, unsigned int p) {
    return TemporalClassifier::twf(p_r, p);
  }

  virtual void reset_model() {
    TemporalClassifier::reset_model();   
    nb::reset_model();
    aPrioriNumCache.clear();
    aPrioriDenCache.clear();
    probCondDenCache.clear();
  }

  virtual ~temporal_nb() {}

 private:
  std::map<std::string, MyBig> aPrioriNumCache;
  std::map<std::string, MyBig> aPrioriDenCache;
  std::map<std::string, MyBig> probCondDenCache;

  MyBig lambda_;
  bool unif_prior_;

  MyBig apriori_num(const std::string &doc_class,const std::string &ref_year);
  MyBig apriori_den(const std::string &ref_year);
  virtual MyBig apriori(const std::string &doc_class,
                        const std::string &ref_year,
                        const MyBig &den) {
    return apriori_num(doc_class, ref_year) / den;
  }

  MyBig term_conditional_num(const int &term_id,
                             const std::string &cur_class,
                             const std::string &ref_year);
  MyBig term_conditional_den(const std::string &cur_class,
                             const std::string &ref_year);
  virtual MyBig term_conditional(const int &term_id,
                                 const std::string &cur_class,
                                 const std::string &ref_year,
                                 const MyBig &den) {
    return term_conditional_num(term_id, cur_class, ref_year) / den;
  }
};

bool temporal_nb::check_train_line(const std::string &line) const {
  std::vector<std::string> tokens;
  Utils::string_tokenize(line, tokens, ";");
  // input format: doc_id;year;class_name;{term_id;tf}+
  if ((tokens.size() < 5) || (tokens.size() % 2 == 0)) return false;
  else return true;
}

bool temporal_nb::parse_train_line(const std::string &line) {
  std::vector<std::string> tokens;
  Utils::string_tokenize(line, tokens, ";");
  // input format: doc_id;year;class_name;{term_id;tf}+
  if ((tokens.size() < 5) || (tokens.size() % 2 == 0)) return false;

  std::string id = tokens[0];
  std::string year = tokens[1];
  years_add(year);

  std::string doc_class = tokens[2];
  classes_add(doc_class);

  std::string cl_yr_idx = Utils::get_index(doc_class, year);
  // update document frequency
  DF[cl_yr_idx]++;

  // retrieve each term frequency and update occurrencies
  for (size_t i = 3; i < tokens.size()-1; i+=2) {
    unsigned int tf = atoi(tokens[i+1].data());
    int term_id = atoi(tokens[i].data());
    vocabulary_add(term_id);

    std::string trm_cl_yr_idx = Utils::get_index(term_id,
                                Utils::get_index(doc_class, year));
    TF[trm_cl_yr_idx] += tf;
    sumTF[cl_yr_idx] += tf;

    if (n_t.find(term_id) == n_t.end())
      n_t[term_id] = (MyBig) tf;
    else n_t[term_id] += (MyBig) tf;
    total_terms += (MyBig) tf;
  }
  return true;
}

MyBig temporal_nb::apriori_den(const std::string &ref_year) {
  if (aPrioriDenCache.find(ref_year) == aPrioriDenCache.end()) {
    MyBig aPrioriDen = "0.0e+800";
    std::set<std::string>::const_iterator cIt = classes_begin();
    while (cIt != classes_end()) {
      std::set<std::string>::const_iterator yIt = years_begin();
      while (yIt != years_end()) {
        std::string idx = Utils::get_index(*(cIt), *(yIt));
        aPrioriDen += static_cast<MyBig>(
                        static_cast<double>(Utils::get_value(DF, idx)) *
                        twf(atoi(ref_year.data()), atoi(yIt->data())));
        ++yIt;
      }
      ++cIt;
    }
    aPrioriDen = aPrioriDen + (alpha_ * static_cast<MyBig>(classes_size()));

    #pragma omp critical(apriori_den_cache)
    {
      aPrioriDenCache[ref_year] = aPrioriDen;
    }
    return aPrioriDen;
  }
  else return aPrioriDenCache[ref_year];
}

MyBig temporal_nb::apriori_num(const std::string &doc_class,
                               const std::string &ref_year) {
  std::string idx = Utils::get_index(doc_class, ref_year);
  if (aPrioriNumCache.find(idx) == aPrioriNumCache.end()) {
    MyBig aPrioriNum = "0.0e+800";
    std::set<std::string>::const_iterator yIt = years_begin();
    while (yIt != years_end()) {
      std::string cl_yr_idx = Utils::get_index(doc_class, *(yIt));
      aPrioriNum += static_cast<MyBig>(
                      static_cast<double>(Utils::get_value(DF, cl_yr_idx)) *
                      twf(atoi(ref_year.data()), atoi(yIt->data())));
      ++yIt;
    }
    aPrioriNum = aPrioriNum + alpha_;
    #pragma omp critical(apriori_num_cache)
    {
      aPrioriNumCache[idx] = aPrioriNum;
    }
    return aPrioriNum;
  }
  else return aPrioriNumCache[idx];
}

MyBig temporal_nb::term_conditional_den(const std::string &cur_class,
                                        const std::string &ref_year) {
  std::string idx = Utils::get_index(cur_class, ref_year);
  if (probCondDenCache.find(idx) == probCondDenCache.end()) {
    MyBig condDen = "0.0e+800";
    std::set<std::string>::const_iterator yIt = years_begin();
    while (yIt != years_end()) {
      std::string cl_yr_idx = Utils::get_index(cur_class, *(yIt));
      condDen += static_cast<MyBig>(
                   static_cast<double>(Utils::get_value(sumTF, cl_yr_idx)) *
                   twf(atoi(ref_year.data()), atoi(yIt->data())));
      ++yIt;
    }
    condDen = condDen + (alpha_ * static_cast<MyBig>(vocabulary_size()));
    #pragma omp critical(probcond_den_cache)
    {
      probCondDenCache[idx] = condDen;
    }
    return condDen;
  }
  else return probCondDenCache[idx];
}

MyBig temporal_nb::term_conditional_num(const int &term_id,
                                        const std::string &cur_class,
                                        const std::string &ref_year) {
  MyBig condNum = "0.0e+800";
  std::set<std::string>::const_iterator yIt = years_begin();
  while(yIt != years_end()) {
    std::string idx = Utils::get_index(term_id,
                        Utils::get_index(cur_class, *(yIt)));
    condNum += static_cast<MyBig>(
                 static_cast<double>(Utils::get_value(TF, idx)) *
                 twf(atoi(ref_year.data()), atoi(yIt->data())));
    ++yIt;
  }
  condNum = condNum + alpha_;
  return condNum;
}

void temporal_nb::parse_test_line(const std::string &line) {
  std::vector<std::string> tokens;
  Utils::string_tokenize(line, tokens, ";");
  // input format: doc_id;year;class_name;{term_id;tf}+
  if ((tokens.size() < 5) || (tokens.size() % 2 == 0)) return;

  MyBig normalizer = "0.0e+800";

  std::string id = tokens[0];
  std::string ref_year = tokens[1];

  Scores<MyBig> similarities(id, tokens[2]);

  MyBig aprioriDen = apriori_den(ref_year);

  std::set<std::string>::const_iterator cIt = classes_begin(); 
  while(cIt != classes_end()) {
    std::string cur_class = *cIt;

    MyBig aPriori = apriori(cur_class, ref_year, aprioriDen);

    MyBig condDen = term_conditional_den(cur_class, ref_year);

    MyBig probCond = "0.0e+800"; probCond += MyBig("1.0");
    for (size_t i = 3; i < tokens.size()-1; i+=2) {
      int term_id = atoi(tokens[i].data());
      unsigned int tf = atoi(tokens[i+1].data());
      MyBig val = term_conditional(term_id, cur_class, ref_year, condDen);
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
