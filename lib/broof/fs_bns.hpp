#ifndef FS_BNS_HPP__
#define FS_BNS_HPP__

#include <limits>
#include <cmath>

#include "feature_selector.hpp"

class fs_bns : public FeatureSelector {
 public:
  fs_bns(const std::string &i, double p, bool rr=false) :
    FeatureSelector(i, p, rr) {}

 protected:
  void compute_weights();
  void compute_weights_round_robin();

 private:
  // helper functions
  double RationalApproximation(double t) {
    double c[] = {2.515517, 0.802853, 0.010328};
    double d[] = {1.432788, 0.189269, 0.001308};
    return t - ((c[2]*t + c[1])*t + c[0]) /
               (((d[2]*t + d[1])*t + d[0])*t + 1.0);
  }

  double NormalCDFInverse(double p) {
    if (p <= 0.0 || p >= 1.0) {
      std::cerr << "Invalid input argument (" << p
                << "); must be larger than 0 but less than 1." << std::endl;
      exit(1);
    }
    if (p < 0.5) return -RationalApproximation( sqrt(-2.0*log(p)) );
    else return RationalApproximation( sqrt(-2.0*log(1-p)) );

    return 0.0; // never reaches here!
  }
};

void fs_bns::compute_weights_round_robin() {
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

      double b = fabs(NormalCDFInverse(p_kc) - NormalCDFInverse(p_knc));
      push_term(*cIt, Term(*vIt, b));
      ++vIt;
    }
    ++cIt;
  }
}

void fs_bns::compute_weights() {
  std::cerr << "[BNS] Computing weights (voc_size=" << vocabulary.size() << ")..." << std::endl;
  if (rr) {
    compute_weights_round_robin();
    return;
  }

  std::set<int>::iterator vIt = vocabulary.begin();
  while (vIt != vocabulary.end()) {
    double max_bns = std::numeric_limits<double>::min();
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

      double b = fabs(NormalCDFInverse(p_kc) - NormalCDFInverse(p_knc));
      if (max_bns < b) max_bns = b;
      ++cIt;
    }
    push_term(Term(*vIt, max_bns));
    ++vIt;
  }
}
#endif
