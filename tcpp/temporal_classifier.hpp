#ifndef TEMPORAL_CLASSIFIER_HPP__
#define TEMPORAL_CLASSIFIER_HPP__

#include <fstream>

#include "utils.hpp"
#include "supervised_classifier.hpp"

class TemporalClassifier : public virtual SupervisedClassifier {
 public:
  TemporalClassifier(const std::string &twf_fn, unsigned int r=0, double b=1.0,
      bool w = false, bool gw = false, unsigned int w_sz=0) :
    SupervisedClassifier(r), twf_fn_(twf_fn), window_(w), grad_w_(gw), window_sz_(w_sz), min_twf_(0.0), beta_(b)
      { read_twf(twf_fn_); }

  void years_add(const std::string &c)
    { years_.insert(c); }
  std::set<std::string>::const_iterator years_begin()
    { return years_.begin(); }
  std::set<std::string>::const_iterator years_end()
    { return years_.end(); }

  void set_twf_file(const std::string &twf) { twf_fn_ = twf; }
  void set_beta(const double b) { beta_ = b; } 

  virtual void reset_model() {
    twf_.clear();
    years_.clear();
  }

  // p_r is the reference point in time
  // p is any point in time
  virtual double twf(unsigned int p_r, unsigned int p) {
    if (!window_ && !grad_w_ && twf_.empty()) read_twf(twf_fn_);
    int temp_dist = p - p_r;
    if (window_ && (static_cast<unsigned int>(abs(temp_dist)) > window_sz_)) return beta_ * min_twf_;
    if (window_ && !grad_w_ && (static_cast<unsigned int>(abs(temp_dist)) <= window_sz_)) return beta_; // janela binaria FIXME
    std::map<int, double>::const_iterator it = twf_.find(temp_dist);
    if (it != twf_.end()) return beta_ * it->second;
    else return beta_ * min_twf_;
  }

  virtual void read_twf(const std::string &twf_fn) {
    if (window_ && !grad_w_) return;
    std::ifstream file(twf_fn.data());
    if (file) {
      twf_.clear();
      std::string line;
      while (file >> line) {
        std::vector<std::string> tokens;
        Utils::string_tokenize(line, tokens, ";");
        int delta = atoi(tokens[0].data());
        double value = atof(tokens[1].data());
        twf_[delta] = value;
        // REMOVE-ME IF WORSE THAN JUST 0.0
        if (value < min_twf_) min_twf_ = value;
      }
      file.close();
    }
  }

 private:
  std::string twf_fn_;
  bool window_;
  bool grad_w_;
  unsigned int window_sz_;
  std::map<int, double> twf_;
  std::set<std::string> years_;
  double min_twf_;

 protected:
  double beta_;
};
#endif
