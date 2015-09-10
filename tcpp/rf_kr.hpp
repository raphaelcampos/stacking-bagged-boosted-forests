#ifndef RF_KR_H__
#define RF_KR_H__

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

class RF_KR : public SupervisedClassifier{
  public:
    RF_KR(unsigned int r, double m=1.0, unsigned int k=30, unsigned int num_trees=10) : SupervisedClassifier(r), num_trees_(num_trees), k_(k), m_(m){docs_processed_ = 0;}
    ~RF_KR();
    bool parse_train_line(const std::string&);
    void parse_test_line(const std::string&);
    void reset_model();
    Scores<double> classify(const DTDocument*);
  private:
    std::vector<const DTDocument*> docs_;
    double m_;
    unsigned int k_;
    unsigned int num_trees_;
    
    unsigned int docs_processed_;
};


RF_KR::~RF_KR(){
  reset_model();
}

void RF_KR::reset_model(){
  for(int i = 0; i < docs_.size(); i++){
    delete docs_[i];
  }
  docs_.clear();
  m_ = 1.0;
  num_trees_ = 0;
  k_ = 30;
}

bool RF_KR::parse_train_line(const std::string& line){
  std::vector<std::string> tokens;
  Utils::string_tokenize(line, tokens, ";");
  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return false;
  DTDocument * doc = new DTDocument();
  std::string doc_id = tokens[0];
  doc->set_id(doc_id);
  std::string doc_class = tokens[1];
  doc->set_class(doc_class);
  for (size_t i = 2; i < tokens.size()-1; i+=2) {
    unsigned int tf = atoi(tokens[i+1].data());
    int term_id = atoi(tokens[i].data());
    doc->insert_term(term_id, tf);
  }
  docs_.push_back(doc);
  return true;
}

Scores<double> RF_KR::classify(const DTDocument* doc){
  Scores<double> similarities(doc->get_id(), doc->get_class());

  RF * rf = new RF(round, m_, num_trees_);
  rf->set_doc_delete(false);
  std::set<const DTDocument*> doc_bag;
  while(doc_bag.size() < k_){
    #pragma omp critical (rand_call)
    {
      doc_bag.insert(docs_[rand() % docs_.size()]);
    }
  }
  rf->add_document_bag(doc_bag);
  rf->build();

  similarities = rf->classify(doc);

  delete rf;
  return similarities;
}

void RF_KR::parse_test_line(const std::string& line){
  std::vector<std::string> tokens;
  Utils::string_tokenize(line, tokens, ";");

  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return;

  DTDocument * doc = new DTDocument();
  std::string doc_id = tokens[0];
  doc->set_id(doc_id);
  std::string doc_class = tokens[1];
  doc->set_class(doc_class);
  for (size_t i = 2; i < tokens.size()-1; i+=2) {
    unsigned int tf = atoi(tokens[i+1].data());
    int term_id = atoi(tokens[i].data());

    doc->insert_term(term_id, tf);
  }

  Scores<double> similarities = classify(doc);
  docs_processed_++;
  std::cerr.precision(4);
  std::cerr.setf(std::ios::fixed);
  std::cerr << "\r" << double(docs_processed_)/docs_.size() * 900 << "%";
  
  get_outputer()->output(similarities);
  delete doc;
}

#endif
