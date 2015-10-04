#ifndef TEMPORAL_NB_LOG_HPP__
#define TEMPORAL_NB_LOG_HPP__

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

#include "nb_log.hpp"
#include "temporal_classifier.hpp"

class temporal_nb_log : public nb_log, public TemporalClassifier {
 public:
  temporal_nb_log(const std::string &twf, unsigned int r=0, double a=1.0, double b=1.0, double l=0.0,
      bool u=false, bool w = false, bool gw = false, unsigned int w_sz=0) :
    nb_log(r, a, l), TemporalClassifier(twf,r,b,w,gw,w_sz), lambda_(l), unif_prior_(u) {}

  virtual bool parse_train_line(const std::string &l);
  virtual void parse_test_line(const std::string &l);
  virtual bool check_train_line(const std::string &l) const;

  virtual double twf(unsigned int p_r, unsigned int p) {
    return TemporalClassifier::twf(p_r, p);
  }

  virtual void reset_model() {
    TemporalClassifier::reset_model();
    nb_log::reset_model();
    aPrioriNumCache.clear();
    aPrioriDenCache.clear();
    probCondDenCache.clear();
  }

  virtual ~temporal_nb_log() {}

 private:
  std::map<std::string, double> aPrioriNumCache;
  std::map<std::string, double> aPrioriDenCache;
  std::map<std::string, double> probCondDenCache;

  double lambda_;
  bool unif_prior_;

  double apriori_num(const std::string &doc_class,const std::string &ref_year);
  double apriori_den(const std::string &ref_year);
  virtual double apriori(const std::string &doc_class,
                         const std::string &ref_year,
                         const double den) {
    return apriori_num(doc_class, ref_year) / den;
  }

  double term_conditional_num(const int &term_id,
                              const std::string &cur_class,
                              const std::string &ref_year);
  double term_conditional_den(const std::string &cur_class,
                              const std::string &ref_year);
  virtual double term_conditional(const int &term_id,
                                  const std::string &cur_class,
                                  const std::string &ref_year,
                                  const double den) {
    return term_conditional_num(term_id, cur_class, ref_year) / den;
  }
};

bool temporal_nb_log::check_train_line(const std::string &line) const {
  std::vector<std::string> tokens;
  Utils::string_tokenize(line, tokens, ";");
  // input format: doc_id;year;class_name;{term_id;tf}+
  if ((tokens.size() < 5) || (tokens.size() % 2 == 0)) return false;
  else return true;
}

bool temporal_nb_log::parse_train_line(const std::string &line) {
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

    n_t[term_id] += tf;
    total_terms += tf;
  }
  return true;
}

double temporal_nb_log::apriori_den(const std::string &ref_year) {
  if (aPrioriDenCache.find(ref_year) == aPrioriDenCache.end()) {
    double aPrioriDen = 0.0;
    std::set<std::string>::const_iterator cIt = classes_begin();
    while (cIt != classes_end()) {
      std::set<std::string>::const_iterator yIt = years_begin();
      while (yIt != years_end()) {
        std::string idx = Utils::get_index(*(cIt), *(yIt));
        aPrioriDen += static_cast<double>(Utils::get_value(DF, idx)) *
                        twf(atoi(ref_year.data()), atoi(yIt->data()));
        ++yIt;
      }
      ++cIt;
    }
    aPrioriDen = aPrioriDen + (alpha_ * static_cast<double>(classes_size()));
    #pragma omp critical(apriori_den_cache_log)
    {
      aPrioriDenCache[ref_year] = aPrioriDen;
    }
    return aPrioriDen;
  }
  else return aPrioriDenCache[ref_year];
}

double temporal_nb_log::apriori_num(const std::string &doc_class,
                               const std::string &ref_year) {
  std::string idx = Utils::get_index(doc_class, ref_year);
  if (aPrioriNumCache.find(idx) == aPrioriNumCache.end()) {
    double aPrioriNum = 0.0;
    std::set<std::string>::const_iterator yIt = years_begin();
    while (yIt != years_end()) {
      std::string cl_yr_idx = Utils::get_index(doc_class, *(yIt));
      aPrioriNum += static_cast<double>(Utils::get_value(DF, cl_yr_idx)) *
                      twf(atoi(ref_year.data()), atoi(yIt->data()));
      ++yIt;
    }
    aPrioriNum = aPrioriNum + alpha_;
    #pragma omp critical(apriori_num_cache_log)
    {
      aPrioriNumCache[idx] = aPrioriNum;
    }
    return aPrioriNum;
  }
  else return aPrioriNumCache[idx];
}

double temporal_nb_log::term_conditional_den(const std::string &cur_class,
                                        const std::string &ref_year) {
  std::string idx = Utils::get_index(cur_class, ref_year);
  if (probCondDenCache.find(idx) == probCondDenCache.end()) {
    double condDen = 0.0;
    std::set<std::string>::const_iterator yIt = years_begin();
    while (yIt != years_end()) {
      std::string cl_yr_idx = Utils::get_index(cur_class, *(yIt));
      condDen += static_cast<double>(Utils::get_value(sumTF, cl_yr_idx)) *
                   twf(atoi(ref_year.data()), atoi(yIt->data()));
      ++yIt;
    }
    condDen = condDen + (alpha_ * static_cast<double>(vocabulary_size()));
    #pragma omp critical(probcond_den_cache_log)
    {
      probCondDenCache[idx] = condDen;
    }
    return condDen;
  }
  else return probCondDenCache[idx];
}

double temporal_nb_log::term_conditional_num(const int &term_id,
                                        const std::string &cur_class,
                                        const std::string &ref_year) {
  double condNum = 0.0;
  std::set<std::string>::const_iterator yIt = years_begin();
  while(yIt != years_end()) {
    std::string idx = Utils::get_index(term_id,
                        Utils::get_index(cur_class, *(yIt)));
    condNum += static_cast<double>(Utils::get_value(TF, idx)) *
                 twf(atoi(ref_year.data()), atoi(yIt->data()));
    ++yIt;
  }
  condNum = condNum + alpha_;
  return condNum;
}

void temporal_nb_log::parse_test_line(const std::string &line) {
  std::vector<std::string> tokens;
  Utils::string_tokenize(line, tokens, ";");
  // input format: doc_id;year;class_name;{term_id;tf}+
  if ((tokens.size() < 5) || (tokens.size() % 2 == 0)) return;

  double normalizer = 0.0;

  std::string id = tokens[0];
  std::string ref_year = tokens[1];

  Scores<double> similarities(id, tokens[2]);

  double aprioriDen = apriori_den(ref_year);
  std::set<std::string>::const_iterator cIt = classes_begin();
  while(cIt != classes_end()) {
    std::string cur_class = *cIt;

    double aPriori = log(apriori(cur_class, ref_year, aprioriDen));

    double condDen = term_conditional_den(cur_class, ref_year);

    double probCond = 0.0;
    for (size_t i = 3; i < tokens.size()-1; i+=2) {
      int term_id = atoi(tokens[i].data());
      unsigned int tf = atoi(tokens[i+1].data());
      double val = term_conditional(term_id, cur_class, ref_year, condDen);
      double num_t = nt(term_id);

      val = lambda_ * (num_t / total_terms) + (1.0-lambda_) * val;
      probCond += tf * log(val);
    }

    double sim = aPriori + probCond;
    similarities.add(cur_class, sim);
    normalizer = normalizer + sim;
    ++cIt;
  }
  get_outputer()->output(similarities/*, normalizer*/);
}
#endif
