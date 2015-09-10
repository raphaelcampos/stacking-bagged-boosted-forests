#ifndef TTEST_HPP__
#define TTEST_HPP__

#include <map>
#include <fstream>
#include <iostream>
#include <cmath>
#include <sstream>

#include "utils.hpp"
#include "stats.hpp"

class ConfInt {
 public:
  ConfInt(const double l, const double u) : ci(l, u) {}
  double lower_bound() {return ci.first;}
  double upper_bound() {return ci.second;}
 private:
  std::pair<double, double> ci;
};

class TTest {
 public:
  TTest(const std::string &res1, const std::string &res2) {
    sample1_.parse_result(res1);
    sample2_.parse_result(res2);
  }

  // RESTRICTION: TWO-SIDED TEST. 10 samples, 99 confidence :-) deadline up there !!
  // df:9    ;    t-quantile: 0.995    ;    t-value: 3.25
  ConfInt ttest_mic_f1() {
    double t_value = 0.978;
    // compute the difference
    unsigned int num_trials = sample1_.num_trials();
    if (num_trials != sample2_.num_trials()) {
      std::cerr << "Currently, just PAIRED test. Num_trials differ. Aborting." << std::endl;
      exit(1);
    }

    double sum_diff = 0.0;
    for (unsigned int i = 0; i < num_trials; i++) {
      double diff = (sample1_.micro_f1(i) - sample2_.micro_f1(i));
      sum_diff += diff;
    }
    double mean_diff = sum_diff / static_cast<double>(num_trials);

    double sd_diff = 0.0;
    for (unsigned int i = 0; i < num_trials; i++) {
      double diff = (sample1_.micro_f1(i) - sample2_.micro_f1(i));
      sd_diff += pow(diff - mean_diff, 2.0);
    }
    sd_diff = sqrt(sd_diff / static_cast<double>(num_trials - 1));

    double std_err = sd_diff / sqrt(static_cast<double>(num_trials));

    double lower_bound = mean_diff - t_value * std_err;
    double upper_bound = mean_diff + t_value * std_err;
    return ConfInt(lower_bound, upper_bound);
  }

  ConfInt ttest_mac_f1() {
    double t_value = 0.978;
    // compute the difference
    unsigned int num_trials = sample1_.num_trials();
    if (num_trials != sample2_.num_trials()) {
      std::cerr << "Currently, just PAIRED test. Num_trials differ. Aborting." << std::endl;
      exit(1);
    }

    double sum_diff = 0.0;
    for (unsigned int i = 0; i < num_trials; i++) {
      double diff = (sample1_.macro_f1(i) - sample2_.macro_f1(i));
      sum_diff += diff;
    }
    double mean_diff = sum_diff / static_cast<double>(num_trials);

    double sd_diff = 0.0;
    for (unsigned int i = 0; i < num_trials; i++) {
      double diff = (sample1_.macro_f1(i) - sample2_.macro_f1(i));
      sd_diff += pow(diff - mean_diff, 2.0);
    }
    sd_diff = sqrt(sd_diff / static_cast<double>(num_trials - 1));
    double std_err = sd_diff / sqrt(static_cast<double>(num_trials));

    double lower_bound = mean_diff - t_value * std_err;
    double upper_bound = mean_diff + t_value * std_err;
    return ConfInt(lower_bound, upper_bound);
  }

 private:
  Statistics sample1_;
  Statistics sample2_;
};

#endif
