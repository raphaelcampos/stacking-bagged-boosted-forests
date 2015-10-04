#ifndef RF_ADA_H__
#define RF_ADA_H__

#include <string>
#include <sstream>
#include <map>
#include <set>
#include <list>
#include <vector>
#include <stack>
#include <iostream>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <ctime>

#include "supervised_classifier.hpp"
#include "utils.hpp"
#include "dt.hpp"
#include "rf.hpp"

class RF_ADA : public SupervisedClassifier {
  public:
    RF_ADA(unsigned int r, double m=1.0, unsigned int max_trees=200, unsigned int maxh=0)
      : SupervisedClassifier(r), m_(m), max_trees_(max_trees) {
       docs_processed_ = 0;
      for (unsigned int i = 0; i < max_trees_; i++) {
        RF * rf = new RF(round, m_, 50/*i*/, maxh);
        if (raw) rf->use_raw_weights();
        rf->set_doc_delete(false);
        ensemble_[i] = rf;
      }
    }
    ~RF_ADA();
    bool parse_train_line(const std::string&);
    void train(const std::string&);
    void parse_test_line(const std::string&);
    void reset_model();
    Scores<double> classify(const DTDocument*, std::map<const DTDocument*, double>&);
  private:
    double m_;
    unsigned int max_trees_;
    unsigned int docs_processed_;
    std::map<unsigned int, RF*> ensemble_;
};

RF_ADA::~RF_ADA(){
  reset_model();
}

void RF_ADA::reset_model(){
  for(int i = 0; i < ensemble_.size(); i++){
    delete ensemble_[i];
  }
}

bool RF_ADA::parse_train_line(const std::string& line) {

  std::vector<std::string> tokens; tokens.reserve(100);
  Utils::string_tokenize(line, tokens, ";");
  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return false;
  DTDocument * doc = new DTDocument();
  std::string doc_id = tokens[0];
  doc->set_id(doc_id);
  std::string doc_class = tokens[1];
  doc->set_class(doc_class);
  for (size_t i = 2; i < tokens.size()-1; i+=2) {
    double tf = atof(tokens[i+1].data());
    int term_id = atoi(tokens[i].data());
    double tf_idf = (raw) ? tf : 1.0 + log(tf);
    doc->insert_term(term_id, tf_idf);
  }
  for (unsigned int i = 0; i < max_trees_; i++) {
    ensemble_[i]->add_document(doc);
  }

  return true;

}

void RF_ADA::train(const std::string& train_fn) {
  SupervisedClassifier::train(train_fn);
  WeightSet w;
  for (unsigned int i = 0; i < max_trees_; i++) {
    ensemble_[i]->build(&w);
  }
}

Scores<double> RF_ADA::classify(const DTDocument* doc, std::map<const DTDocument*, double>& sim){
  Scores<double> similarities(doc->get_id(), doc->get_class());
  std::map<std::string, double> sco;
  for (unsigned int i = 0; i < max_trees_; i++) {
    double oob_err = ensemble_[i]->avg_oob_err();
    Scores<double> s = ensemble_[i]->classify(doc);
    while (!s.empty()) {
      Similarity<double> sim = s.top();
      sco[sim.class_name] += sim.similarity * (oob_err == 0.0 ? 1.0 : oob_err == 1.0 ? 0.0 : log((1.0-oob_err)/oob_err));
      s.pop();
    }
  }
  std::map<std::string, double>::const_iterator s_it = sco.begin();
  while (s_it != sco.end()) {
    similarities.add(s_it->first, s_it->second);
    ++s_it;
  }
  return similarities;
}

void RF_ADA::parse_test_line(const std::string& line){
  std::vector<std::string> tokens; tokens.reserve(100);
  Utils::string_tokenize(line, tokens, ";");

  double test_size = 0.0;
  std::map<const DTDocument*, double> doc_similarities;

  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return;

  DTDocument * doc = new DTDocument();
  std::string doc_id = tokens[0];
  doc->set_id(doc_id);
  std::string doc_class = tokens[1];
  doc->set_class(doc_class);

  for (size_t i = 2; i < tokens.size()-1; i+=2) {
    double tf = atof(tokens[i+1].data());
    int term_id = atoi(tokens[i].data());
    double tf_idf = (raw) ? tf : 1.0 + log(tf);
    doc->insert_term(term_id, tf_idf);
  }

  Scores<double> similarities = classify(doc, doc_similarities);
  docs_processed_++;
//  std::cerr.precision(4);
//  std::cerr.setf(std::ios::fixed);
//  std::cerr << "\r" << docs_processed_ << ".";
  
  get_outputer()->output(similarities);
  delete doc;
}

#endif
