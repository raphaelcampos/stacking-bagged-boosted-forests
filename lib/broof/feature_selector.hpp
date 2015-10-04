#ifndef FEATURE_SELECTOR_HPP__
#define FEATURE_SELECTOR_HPP__

#include <string>
#include <set>

#include "abstract_feature_selector.hpp"

class FeatureSelector : public AbstractFeatureSelector {
 public:
  FeatureSelector(const std::string &i, double p, bool rr=false) :
    AbstractFeatureSelector(i, p, rr) {}

  virtual void select(); // feature selection method

  virtual std::string filter(const std::string &l);

  virtual ~FeatureSelector() {}

 protected:
  virtual void compute_weights() = 0;
};

std::string FeatureSelector::filter(const std::string &l) {
  std::vector<std::string> tokens;
  Utils::string_tokenize(l, tokens, ";");
  // input format: doc_id;year;CLASS=class_name;{term_id;tf}+
  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return l;
  std::stringstream res;
  res << tokens[0] << ";" << tokens[1];
  std::string doc_class = tokens[1];
  for (unsigned int i = 2; i < tokens.size()-1; i+=2) {
    if (!rr) {
      std::set<int>::iterator it = filtered.find(atoi(tokens[i].data()));
      if (it == filtered.end()) res << ";" << tokens[i] << ";" << tokens[i+1];
    }
    else {
      std::set<int>::iterator it = filtered_class[doc_class].find(atoi(tokens[i].data()));
      if (it == filtered_class[doc_class].end()) res << ";" << tokens[i] << ";" << tokens[i+1];
    }
  }
  return res.str();
}

void FeatureSelector::select() {
  std::ifstream file(input.data());
  std::string line;
  if (file) {
    while (file >> line) {
      std::vector<std::string> tokens;
      Utils::string_tokenize(line, tokens, ";");
      n_++;
      std::string doc_class = tokens[1];
      classes.insert(doc_class);
      n_c_[doc_class]++;
      for (unsigned int i = 2; i < tokens.size()-1; i+=2) {
        int term_id = atoi(tokens[i].data());
        int freq = atoi(tokens[i+1].data());
        if (rr) vocabulary_class[doc_class].insert(term_id);
        else vocabulary.insert(term_id);
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
