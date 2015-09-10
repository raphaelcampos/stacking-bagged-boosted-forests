#ifndef STATS_HPP__
#define STATS_HPP__

#include <map>
#include <fstream>
#include <iostream>
#include <cmath>

#include "utils.hpp"

class TrialStatistics;
class Statistics;

class Contigency {
 public:
  friend class TrialStatistics;

  Contigency() : TP_(0), TN_(0), FP_(0), FN_(0) {}

  void clear() { TP_ = 0; TN_ = 0; FP_ = 0; FN_ = 0; }
 private:
  unsigned int TP_;
  unsigned int TN_;
  // TPR = TP_ / (TP_ + FN_)
  unsigned int FP_;
  unsigned int FN_;
  // FPR = FP_ / ((FP_ + TN_))
};

class TrialStatistics {
 public:
 friend class Statistics;
  TrialStatistics() : summarized_(false) {}

  void add_prediction(const std::string &t, const std::string &p) {
    preds_[t][p]++;
    classes_.insert(t);
  }

  double micro_f1() {
    if (!summarized_) summary();

    double sum_tp = 0.0; // numerators
    double sum_tp_fp = 0.0, sum_tp_fn = 0.0; // precision/recall denominators

    std::map<std::string, Contigency>::const_iterator it = contigency_.begin();
    while (it != contigency_.end()) {
      sum_tp    += it->second.TP_;
      sum_tp_fp += it->second.TP_ + it->second.FP_;
      sum_tp_fn += it->second.TP_ + it->second.FN_;
      ++it;
    }
    double p_mic = sum_tp / sum_tp_fp;
    double r_mic = sum_tp / sum_tp_fn;

    // F1: harmonic mean of p and r
    return (2.0 * p_mic * r_mic) / (p_mic + r_mic);
  }

  double macro_f1() {
    if (!summarized_) summary();

    double sum_f1 = 0.0;

    std::map<std::string, Contigency>::const_iterator it = contigency_.begin();
    while (it != contigency_.end()) {
      double sum_tp    = it->second.TP_;
      double sum_tp_fp = it->second.TP_ + it->second.FP_;
      double sum_tp_fn = it->second.TP_ + it->second.FN_;
      double p = (sum_tp_fp == 0.0) ? 1.0 : sum_tp / sum_tp_fp;
      double r = (sum_tp_fn == 0.0) ? 0.0 : sum_tp / sum_tp_fn;
      double f1 = ((p+r) == 0.0) ? 0.0 : (2.0 * p * r) / (p + r);
//      std::cerr << "F1(" << it->first << ") = " << f1 << std::endl;
      sum_f1 += f1;
      ++it;
    }
    return sum_f1 / classes_.size();
  }

  double micro_auc() {
    if (!summarized_) summary();

    double sum_tp = 0.0, sum_fp = 0.0; // numerators
    double sum_tp_fn = 0.0, sum_fp_tn = 0.0; // precision/recall denominators

    // TPR = TP_ / (TP_ + FN_)
    // FPR = FP_ / ((FP_ + TN_))

    std::map<std::string, Contigency>::const_iterator it = contigency_.begin();
    while (it != contigency_.end()) {
      sum_tp    += it->second.TP_;
      sum_tp_fn += it->second.TP_ + it->second.FN_;

      sum_fp    += it->second.FP_;
      sum_fp_tn += it->second.FP_ + it->second.TN_;
      ++it;
    }
    double tpr_mic = sum_tp / sum_tp_fn;
    double fpr_mic = sum_fp / sum_fp_tn;

    // AUC: Area under the curve
    return (1.0 + tpr_mic - fpr_mic) / 2.0;
  }

  double macro_auc() {
    if (!summarized_) summary();

    double sum_auc = 0.0;

    std::map<std::string, Contigency>::const_iterator it = contigency_.begin();
    while (it != contigency_.end()) {
      double sum_tp    = it->second.TP_;
      double sum_tp_fn = it->second.TP_ + it->second.FN_;
      double tpr = (sum_tp_fn == 0.0) ? 1.0 : sum_tp / sum_tp_fn;

      double sum_fp    = it->second.FP_;
      double sum_fp_tn = it->second.FP_ + it->second.TN_;
      double fpr = (sum_fp_tn == 0.0) ? 1.0 : sum_fp / sum_fp_tn;

      double auc = ((1.0 + tpr - fpr) / 2.0);

      sum_auc += ((1.0 + tpr - fpr) / 2.0);
      ++it;
    }
    return sum_auc / classes_.size();
  }

  double auc_per_class() {
    double sum_auc = 0.0;
    std::map<std::string, std::map<std::string, double> > bla;
    std::map<std::string, Contigency>::const_iterator it_i = contigency_.begin();
    while (it_i != contigency_.end()) {
      std::map<std::string, Contigency>::const_iterator it_j = it_i; ++it_j;
      while (it_j != contigency_.end()) {
        if (it_i->first.compare(it_j->first) != 0) {
          double tp    = preds_[it_i->first][it_i->first];
          double tp_fn = tp + preds_[it_i->first][it_j->first];
          double tpr = (tp_fn == 0.0) ? 0.0 : tp / tp_fn;

          double fp    = preds_[it_j->first][it_i->first];
          double fp_tn = fp + preds_[it_j->first][it_j->first];
          double fpr = (fp_tn == 0.0) ? 0.0 : fp / fp_tn;

          double auc = ((1.0 + tpr - fpr) / 2.0);
          bla[it_i->first][it_j->first] = auc;
          sum_auc += auc;
        }
        ++it_j;
      }
      ++it_i;
    }
/*
    std::map<std::string, std::map<std::string, double> >::const_iterator it = bla.begin();
    while (it != bla.end()) {
      std::cerr << "AUC(" << it->first << ") =";
      std::map<std::string, double>::const_iterator it_n = it->second.begin();
      while (it_n != it->second.end()) {
        std::cerr << " " << it_n->first << ":" << it_n->second;
        ++it_n;
      }
      std::cerr << std::endl;
      ++it;
    }
*/
    double c_sz = static_cast<double>(classes_.size());
    return ( (2.0*sum_auc) / (c_sz*(c_sz-1.0)) );
  }

  double micro_sensitivity() {
    if (!summarized_) summary();

    double sum_tp = 0.0;
    double sum_tp_fn = 0.0;

    std::map<std::string, Contigency>::const_iterator it = contigency_.begin();
    while (it != contigency_.end()) {
      sum_tp    += it->second.TP_;
      sum_tp_fn += it->second.TP_ + it->second.FN_;
      ++it;
    }
    double sensitivity_mic = sum_tp / sum_tp_fn;

    // Sensivity
    return sensitivity_mic;
  }

  double micro_specificity() {
    if (!summarized_) summary();

    double sum_tn = 0.0;
    double sum_fp_tn = 0.0;

    std::map<std::string, Contigency>::const_iterator it = contigency_.begin();
    while (it != contigency_.end()) {
      sum_tn    += it->second.TN_;
      sum_fp_tn += it->second.FP_ + it->second.TN_;
      ++it;
    }
    double specificity_mic = sum_tn / sum_fp_tn;

    // Specificity
    return specificity_mic;
  }

  double macro_sensitivity() {
    if (!summarized_) summary();

    double sum_sensivity = 0.0;

    std::map<std::string, Contigency>::const_iterator it = contigency_.begin();
    while (it != contigency_.end()) {
      double sum_tp    = it->second.TP_;
      double sum_tp_fn = it->second.TP_ + it->second.FN_;
      sum_sensivity += sum_tp / sum_tp_fn;
      ++it;
    }
    return sum_sensivity / classes_.size();
  }

  double macro_specificity() {
    if (!summarized_) summary();

    double sum_specificity = 0.0;

    std::map<std::string, Contigency>::const_iterator it = contigency_.begin();
    while (it != contigency_.end()) {
      double sum_tn    = it->second.TN_;
      double sum_fp_tn = it->second.FP_ + it->second.TN_;
      sum_specificity += sum_tn / sum_fp_tn;
      ++it;
    }
    return sum_specificity / classes_.size();
  }

 private:
  //         true class  -> [predicted class -> count]
  std::map<std::string, std::map<std::string, unsigned int> > preds_;
  std::map<std::string, Contigency> contigency_;
  std::set<std::string> classes_;

  bool summarized_;

  void summary() {
    if (summarized_) return;

    std::set<std::string>::const_iterator pos_it = classes_.begin(); // positive class
    while (pos_it != classes_.end()) {
      contigency_[*pos_it].clear();

      std::set<std::string>::const_iterator g_it = classes_.begin(); // gold standard
      while (g_it != classes_.end()) {

        std::set<std::string>::const_iterator p_it = classes_.begin(); // predicted class
        while (p_it != classes_.end()) {

          if (g_it->compare(*pos_it) != 0) { // evaluate negative classes
            if (p_it->compare(*pos_it) == 0) contigency_[*pos_it].FP_ += preds_[*g_it][*p_it];
            else contigency_[*pos_it].TN_ += preds_[*g_it][*p_it];
          }
          else { // evaluate predictions of positive classes
            if (p_it->compare(*g_it) == 0) contigency_[*pos_it].TP_ += preds_[*g_it][*p_it];
            else contigency_[*pos_it].FN_ += preds_[*g_it][*p_it];
          }
          ++p_it;
        }
        ++g_it;
      }
      ++pos_it;
    }
    summarized_ = true;
  }

};

class Statistics {
 public:
  /*
    id    actual  predicted class  |--> disconsider remaining classes
    86332 CLASS=1 CLASS=1:0.179719 CLASS=3:0.140017 CLASS=8:0.130019 CLASS=7:0.121796 CLASS=2:0.116108
  */
  void parse_result(const std::string &fn) {
    std::ifstream file(fn.data());
    if (file) {
      std::string line;
      unsigned int trial = 0;
      bool fst_ln = true;
      while (std::getline(file, line)) {
        if (line[0] == '#') {
          fst_ln = false;
          trial++;
          stats_.push_back(TrialStatistics());
          continue;
	}
        else if (fst_ln) {
          fst_ln = false;
          trial++;
          stats_.push_back(TrialStatistics());
        }
        std::map<std::string, double> scores;
        // format: doc_id CLASS=true_class CLASS=pred_class:value
        std::vector<std::string> tokens;
        Utils::string_tokenize(line, tokens, " ");
        if (tokens.size() < 3) continue;

        std::string id_doc = tokens[0];

        std::string real = tokens[1];
        std::string pred = tokens[2];
        tokens.clear();
        Utils::string_tokenize(pred, tokens, ":");
        pred = tokens[0];

        stats_.back().add_prediction(real, pred);
      }
      file.close();
      stats_.back().summary();
    }
    else {
      std::cerr << "Unable to open result data." << std::endl;
      exit(1);
    }
  }

  double micro_f1(unsigned int trial) {
    if (trial > stats_.size()) {
      std::cerr << "Invalid trial number: " << trial << " (just " << stats_.size() << " trials available)." << std::endl;
      exit(1);
    }
    return 100*stats_[trial].micro_f1();
  }

  double macro_f1(unsigned int trial) {
    if (trial > stats_.size()) {
      std::cerr << "Invalid trial number: " << trial << " (just " << stats_.size() << " trials available)." << std::endl;
      exit(1);
    }
    return 100*stats_[trial].macro_f1();
  }

  unsigned int num_trials() { return stats_.size(); }

  void summary() {
    double sum[9] = {0.0};
    double s_sum[9] = {0.0};
    std::string names[9] = {
      "Micro-F1",
      "Macro-F1",
      "Micro-AUC",
      "Macro-AUC",
      "Micro-Sensitivity",
      "Macro-Sensitivity",
      "Micro-Specificity",
      "Macro-Specificity",
      "AUC_total"};

    std::map<std::string, double> auc_pc_sum;
    std::map<std::string, double> auc_pc_s_sum;

    for (unsigned int i = 0; i < stats_.size(); i++) {
      double mic_f1  = stats_[i].micro_f1();
      double mac_f1  = stats_[i].macro_f1();
//      double mic_auc = stats_[i].micro_auc();
//      double mac_auc = stats_[i].macro_auc();
//      double mic_sen = stats_[i].micro_sensitivity();
//      double mac_sen = stats_[i].macro_sensitivity();
//      double mic_spe = stats_[i].micro_specificity();
//      double mac_spe = stats_[i].macro_specificity();
//      double auc     = stats_[i].auc_per_class();

      sum[0] += mic_f1;
      sum[1] += mac_f1;
//      sum[2] += mic_auc;
//      sum[3] += mac_auc;
//      sum[4] += mic_sen;
//      sum[5] += mac_sen;
//      sum[6] += mic_spe;
//      sum[7] += mac_spe;
//      sum[8] += auc;

      s_sum[0] += pow(mic_f1, 2);
      s_sum[1] += pow(mac_f1, 2);
//      s_sum[2] += pow(mic_auc, 2);
//      s_sum[3] += pow(mac_auc, 2);
//      s_sum[4] += pow(mic_sen, 2);
//      s_sum[5] += pow(mac_sen, 2);
//      s_sum[6] += pow(mic_spe, 2);
//      s_sum[7] += pow(mac_spe, 2);
//      s_sum[8] += pow(auc, 2);
    }
 
    std::cout << "STATISTICS FOR " << stats_.size() << " TRIAL" << (stats_.size() > 1 ? "S." : ".") << std::endl;
    for (unsigned int i = 0; i < 2/*9*/; i++) {
      double mean = sum[i] / stats_.size();
      double sd = (stats_.size() > 1) ? sqrt(s_sum[i] / stats_.size() - pow(mean, 2)) : 0.0;
      std::cout << names[i] << ": " << (mean*100.0) << " += " << (sd*100.0) << std::endl;
    }  
  }

 private:
  std::vector<TrialStatistics> stats_;
};

#endif
