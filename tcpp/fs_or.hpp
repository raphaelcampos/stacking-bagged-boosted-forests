#ifndef FS_ODDS_RATIO_HPP__
#define FS_ODDS_RATIO_HPP__

#include <limits>
#include <cmath>

#include "feature_selector.hpp"

class fs_odds_ratio : public FeatureSelector {
 public:
  fs_odds_ratio(const std::string &i, double p, bool r=false) :
    FeatureSelector(i, p,r) {}

 protected:
  void compute_weights();
  void compute_weights_round_robin();
};

void fs_odds_ratio::compute_weights_round_robin() {
  std::set<std::string>::iterator cIt = classes.begin();
  while (cIt != classes.end()) {
    std::set<int>::iterator vIt = vocabulary_class[*cIt].begin();
    while (vIt != vocabulary_class[*cIt].end()) {
      double n = static_cast<double>(n_) + 4.0;
//      double n_c = static_cast<double>(Utils::get_value(n_c_, *cIt) + 2.0);
      double n_k = static_cast<double>(Utils::get_value(n_k_, *vIt) + 2.0);
      double n_kc = static_cast<double>(Utils::get_value(n_kc_,
                                        Utils::get_index(*vIt, *cIt)) + 1.0);

      double p_kc   = n_kc / n;
      double p_knc  = (n_k - n_kc) / n;

      if (p_kc == 0.0 || p_knc == 0.0) {
        std::cerr << "Term  = " << *vIt << std::endl;
        std::cerr << "Class = " << *vIt << std::endl << std::endl;
        std::cerr << "P(k,c)   = " << p_kc   << std::endl;
        std::cerr << "P(k,c')  = " << p_knc  << std::endl;
        exit(1);
      }

      double b = (p_kc * (1.0 - p_knc)) / ((1.0 - p_kc) * p_knc);
      push_term(*cIt, Term(*vIt, b));
      ++vIt;
    }
    ++cIt;
  }
}

void fs_odds_ratio::compute_weights() {
  std::cerr << "[ODDS RATIO] Computing weights (voc_size=" << vocabulary.size() << ")..." << std::endl;
  if (rr) {
    compute_weights_round_robin();
    return;
  }

  std::set<int>::iterator vIt = vocabulary.begin();
  while (vIt != vocabulary.end()) {
    double max_or = std::numeric_limits<double>::min();
    std::set<std::string>::iterator cIt = classes.begin();
    while (cIt != classes.end()) {
      double n = static_cast<double>(n_) + 4.0;
//      double n_c = static_cast<double>(Utils::get_value(n_c_, *cIt) + 2.0);
      double n_k = static_cast<double>(Utils::get_value(n_k_, *vIt) + 2.0);
      double n_kc = static_cast<double>(Utils::get_value(n_kc_,
                                        Utils::get_index(*vIt, *cIt)) + 1.0);

      double p_kc   = n_kc / n;
      double p_knc  = (n_k - n_kc) / n;

      if (p_kc == 0.0 || p_knc == 0.0) {
        std::cerr << "Term  = " << *vIt << std::endl;
        std::cerr << "Class = " << *vIt << std::endl << std::endl;
        std::cerr << "P(k,c)   = " << p_kc   << std::endl;
        std::cerr << "P(k,c')  = " << p_knc  << std::endl;
        exit(1);
      }

      double w = (p_kc * (1.0 - p_knc)) / ((1.0 - p_kc) * p_knc);
      if (max_or < w) max_or = w;

      ++cIt;
    }
    push_term(Term(*vIt, max_or));
    ++vIt;
  }
}
#endif
