#ifndef RF_H__
#define RF_H__

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

#ifndef CUSTOM_NT
#define NUM_THREADS 8
#endif

class RF : public SupervisedClassifier{
  public:
    RF(unsigned int r, double m=1.0, unsigned int num_trees=10) : SupervisedClassifier(r), num_trees_(num_trees), m_(m), doc_delete_(true) { trees_.reserve(num_trees); srand(time(NULL)); }
    ~RF();
    bool parse_train_line(const std::string&);
    void train(const std::string&);
    void parse_test_line(const std::string&);
    void reset_model();
    void add_document(const DTDocument*);
    void add_document_bag(std::set<const DTDocument*>&);
    void build();
    void set_doc_delete(const bool&);
    Scores<double> classify(const DTDocument*);
  private:
    std::vector<DT*> trees_;
    std::vector<const DTDocument*> docs_;
    double m_;
    unsigned int num_trees_;
    bool doc_delete_;
};

RF::~RF(){
  reset_model();
}

void RF::set_doc_delete(const bool& dd){
  doc_delete_ = dd;
}

void RF::reset_model(){
  for(int i = 0; i < num_trees_; i++){
    delete trees_[i];
  }
  trees_.clear();
  if(doc_delete_){
    for(int i = 0; i < docs_.size(); i++){
      delete docs_[i];
    }
    docs_.clear();
  }
  m_ = 1.0;
  num_trees_ = 0;
}

bool RF::parse_train_line(const std::string& line){
  std::vector<std::string> tokens;
  Utils::string_tokenize(line, tokens, " ");
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

void RF::train(const std::string& train_fn){
  SupervisedClassifier::train(train_fn);
  build();
}

void RF::add_document(const DTDocument* doc){
  docs_.push_back(doc);
}

void RF::add_document_bag(std::set<const DTDocument*>& bag){
  std::set<const DTDocument*>::const_iterator cIt = bag.begin();
  while(cIt != bag.end()){
    docs_.push_back(*cIt);
    ++cIt;
  }
}

void RF::build(){
  const unsigned int docs_size = docs_.size();
  #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
  for(int i = 0; i < num_trees_; i++){
    std::set<const DTDocument*> bag;
    for(int j = 0; j < docs_size; j++){
      #pragma omp critical (rand_call)
      {
        bag.insert(docs_[rand() % docs_size]);
      }
    }
    trees_[i] = new DT(round);
    trees_[i]->add_document_bag(bag);
    trees_[i]->build(m_);
    /*if(i%50 == 0)
      #pragma omp critical (printing)
      {
        std::cerr << ".";
      }*/
  }
  //std::cerr << std::endl;
}

Scores<double> RF::classify(const DTDocument* doc){
  Scores<double> similarities(doc->get_id(), doc->get_class());
  std::map<std::string, double> scores;
  std::map<std::string, unsigned int> trees_count;
  for(int i = 0; i < num_trees_; i++){
    std::map<std::string, double> partial_scores = trees_[i]->get_partial_scores(doc);
    std::map<std::string, double>::const_iterator cIt = partial_scores.begin();
    while(cIt != partial_scores.end()){
      std::map<std::string, double>::iterator it = scores.find(cIt->first);
      if(it == scores.end()){
        it = (scores.insert(std::make_pair(cIt->first, 0.0))).first;
      }
      it->second += cIt->second;

      std::map<std::string, unsigned int>::iterator it_tc = trees_count.find(cIt->first);
      if(it_tc == trees_count.end()){
        it_tc = (trees_count.insert(std::make_pair(cIt->first, 0))).first;
      }
      (it_tc->second)++;

      ++cIt;
    }
  }
  std::map<std::string, double>::const_iterator cIt_s = scores.begin();
  while(cIt_s != scores.end()){
    std::map<std::string, unsigned int>::const_iterator cIt_t = trees_count.find(cIt_s->first);
    if(cIt_t == trees_count.end()){
      similarities.add(cIt_s->first, 0.0);
    }
    //else if(cIt_t->second != 1){
    //  similarities.add(cIt_s->first, cIt_s->second / log(cIt_t->second));
    //}
    else{
      similarities.add(cIt_s->first, cIt_s->second / (1.0 + log(cIt_t->second)));
    }
    ++cIt_s;
  }
  return similarities;
}

void RF::parse_test_line(const std::string& line){
  std::vector<std::string> tokens;
  Utils::string_tokenize(line, tokens, " ");

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
  
  get_outputer()->output(similarities);
  delete doc;
}

#endif
