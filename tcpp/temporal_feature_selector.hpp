#ifndef TEMPORAL_FEATURE_SELECTOR_HPP__
#define TEMPORAL_FEATURE_SELECTOR_HPP__

#include <string>
#include <set>

#include "abstract_feature_selector.hpp"

class TemporalFeatureSelector : public AbstractFeatureSelector {
 public:
  TemporalFeatureSelector(const std::string &i, double p) :
    AbstractFeatureSelector(i, p) {}

  virtual void select(); // feature selection method

  virtual std::string filter(const std::string &l);

  virtual ~FeatureSelector() {}

 protected:
  virtual void compute_weights() = 0;
};

std::string TemporalFeatureSelector::filter(const std::string &l) {
  std::vector<std::string> tokens;
  Utils::string_tokenize(l, tokens, ";");
  // input format: doc_id;year;CLASS=class_name;{term_id;tf}+
  if ((tokens.size() < 5) || (tokens.size() % 2 == 0)) return l;
  std::stringstream res;
  res << tokens[0] << ";" << tokens[1] << ";" << tokens[2];
  for (unsigned int i = 3; i <= tokens.size()-1; i+=2) {
    std::set<int>::iterator it = filtered.find(atoi(tokens[i].data()));
    if (it == filtered.end()) res << ";" << tokens[i] << ";" << tokens[i+1];
  }
  return res.str();
}

void TemporalFeatureSelector::select() {
  std::ifstream file(input.data());
  std::string line;
  if (file) {
    while (file >> line) {
      std::vector<std::string> tokens;
      Utils::string_tokenize(line, tokens, ";");
      n_++;
      std::string doc_class = tokens[2];
      classes.insert(doc_class);
      n_c_[doc_class]++;
      for (unsigned int i = 3; i < tokens.size()-1; i+=2) {
        int term_id = atoi(tokens[i].data());
        int freq = atoi(tokens[i+1].data());
        vocabulary.insert(term_id);
        n_k_[term_id]++;
        std::string idx = Utils::get_index(term_id, doc_class);
        n_kc_[idx]++;
        tf_[idx] += freq;
        tfc_[doc_class] += freq;
      }
    }
  }
  else {
    std::cerr << "[FS] Failed to open input file." << std::endl;
    exit(1);
  }
  compute_weights();
  fill_filtered();
}
#endif
