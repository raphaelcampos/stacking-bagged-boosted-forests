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
#include "weight_set.hpp"

#define NUM_THREADS 8

typedef std::pair<float, unsigned int> pair_;

bool compPairs(pair_ lhs, pair_ rhs) { return lhs.first < rhs.first; }


template<class Iterator>
std::vector<pair_> getProbAcumulatedVector(Iterator first, Iterator last, WeightSet *w) {

  std::vector<pair_> v;

  unsigned int i = 0;
  while(first != last){
    pair_ p = std::make_pair(((v.empty())? 0.0 : v.back().first) + w->get((*first)->get_id()), i);
    v.push_back(p);

    ++i;
    ++first;
  }

  return v;
}

unsigned chooseDoc(std::vector<pair_> &v){
  double r = ((double) rand() / (RAND_MAX)) * v.back().first ;
  pair_ p = *std::lower_bound(v.begin(), v.end(), pair_(r, 0), compPairs);

  return p.second;
}


class RF : public SupervisedClassifier{
  public:
    RF(unsigned int r, double m=1.0, unsigned int num_trees=10, unsigned int maxh=0, bool trn_err=false) : SupervisedClassifier(r), num_trees_(num_trees), m_(m), doc_delete_(true), maxh_(maxh), trn_err_(trn_err), trees_(num_trees, NULL){ total_oob_ = 0.0; srand(time(NULL)); oob_.resize(num_trees); }
    ~RF();
    bool parse_train_line(const std::string&);
    void train(const std::string&);
    void parse_test_line(const std::string&);
    void reset_model();
    void add_document(const DTDocument*);
    void add_document_bag(std::set<const DTDocument*>&);
    WeightSet *build(WeightSet * w = NULL);
    void set_doc_delete(const bool&);
    Scores<double> classify(const DTDocument*);
    double avg_oob_err() { return (oob_err_.size() > 0) ? total_oob_/oob_err_.size() : 0.0; }
    int n_classes(){ return classes_.size(); }
  private:
    std::set<std::string> classes_;
    std::vector<DT*> trees_;
    std::vector<const DTDocument*> docs_;
    std::vector<std::vector<const DTDocument*> > oob_;
    double total_oob_;
    std::vector<double> oob_err_;
    double m_;
    unsigned int num_trees_;
    bool doc_delete_;
    unsigned int maxh_;
    bool trn_err_;
};

RF::~RF(){
  reset_model();
}

void RF::set_doc_delete(const bool& dd){
  doc_delete_ = dd;
}

void RF::reset_model(){
  for(int i = 0; i < num_trees_; i++){
    if(trees_[i] != NULL) delete trees_[i];
  }
  trees_.clear();
  if(doc_delete_){
    for(int i = 0; i < docs_.size(); i++){
      delete docs_[i];
      docs_[i] = NULL;
    }
    docs_.clear();
    oob_.clear();
    total_oob_ = 0.0;
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
    double tf_idf = (raw) ? tf : tf; // 1.0 + log(tf); // FIXME
    doc->insert_term(term_id, tf_idf);
  }
  for (unsigned int i = 0; i < num_trees_; i++) {
    oob_[i].push_back(doc);
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
  classes_.insert(doc->get_class());
  for (unsigned int i = 0; i < num_trees_; i++) {
    oob_[i].push_back(doc);
  } 
}

void RF::add_document_bag(std::set<const DTDocument*>& bag){
  std::set<const DTDocument*>::const_iterator cIt = bag.begin();
  while(cIt != bag.end()){
    docs_.push_back(*cIt);
    ++cIt;
  }
}

// Should return
// 1) Map: out-of-bag sample ID -> bool misclassified ?
WeightSet *RF::build(WeightSet *w) {
  const unsigned int docs_size = docs_.size();
  std::vector<pair_> probAcumulatedVector;
  WeightSet w_copy;

  if(w != NULL){
    w->normalize();
    probAcumulatedVector = getProbAcumulatedVector(docs_.begin(), docs_.end(), w);
    w_copy = *w;
  }

  #pragma omp parallel for num_threads(24)
  for(int i = 0; i < num_trees_; ++i) {
    std::set<const DTDocument*> bag;
    for(int j = 0; j < docs_size; ++j) {
      #pragma omp critical (oob_update)
      {
        unsigned int rndIdx = (w != NULL)? chooseDoc(probAcumulatedVector) : rand() % docs_size;
        bag.insert(docs_[rndIdx]);
        if (!trn_err_) {
          oob_[i][rndIdx] = NULL; // it isn't an oob sample
        }
      }
    }
    trees_[i] = new DT(round, maxh_);
    trees_[i]->add_document_bag(bag);
    trees_[i]->build(m_);
    //std::cerr << "trr_oob[" << i << "] = " << miss << "/" << total << "=" << oob_err << std::endl;

    // evaluate OOB error
    double miss = 0.0, total = 0.0;
    std::map<unsigned int, bool> is_miss;
    for (unsigned int oobidx = 0; oobidx < oob_[i].size(); oobidx++) {
      if (oob_[i][oobidx] != NULL) {
        std::map<std::string, double> partial_scores = trees_[i]->get_partial_scores(oob_[i][oobidx]);
        // see if its a missclassification
        double max = -9999.99;
        std::string maxCl;
        std::map<std::string, double>::const_iterator cIt = partial_scores.begin();
        while(cIt != partial_scores.end()) {
          if (cIt->second > max) {
            maxCl = cIt->first;
            max = cIt->second;
          }
          ++cIt;
        }
        double weight =  (w != NULL)? w->get(oob_[i][oobidx]->get_id()) : 1.0;
        if (maxCl != oob_[i][oobidx]->get_class()) {
          is_miss[oobidx] = true;
          miss += weight;
        } else {
          is_miss[oobidx] = false;
        }
        total += weight;
      }
    }
    double oob_err = total == 0.0 ? 0.0 : (miss / total);
    double alpha = oob_err == 0.0 ? 1.0 : oob_err == 1.0 ? 0.0 : ((1.0-oob_err)/oob_err);

    #pragma omp critical (oob_update)
    {
      if (w != NULL) { 
        {
          //printf("%lf - %d - %d\n", total, docs_.size(), w_copy.size());
          //if(isnan(oob_err))
          //  std::cerr << "  WeightUpdate RF " << i << ": " << miss << "/" << total << "=" << oob_err << " alpha=" << alpha << std::endl;
          for (unsigned int oobidx = 0; oobidx < oob_[i].size(); oobidx++) {
            if (oob_[i][oobidx] != NULL) {
              if (is_miss[oobidx]) {
                //std::cerr << oob_[i][oobidx]->get_id() << "    IS_MISS w=" << w->get(oob_[i][oobidx]->get_id()) << " dw=" << exp(alpha) << std::endl;
                w_copy.set(oob_[i][oobidx]->get_id(),  w_copy.get(oob_[i][oobidx]->get_id()) * exp(alpha));
              } else {
                //std::cerr << oob_[i][oobidx]->get_id() << "    IS_HIT w=" << w->get(oob_[i][oobidx]->get_id()) << " dw=" << exp(-alpha) << std::endl;
                w_copy.set(oob_[i][oobidx]->get_id(),  w_copy.get(oob_[i][oobidx]->get_id()) * exp(-alpha));
              }
              //std::cerr << "    OOB " << oobidx << " w=" << w->get(oob_[i][oobidx]->get_id()) << " is_miss=" << (is_miss[oobidx] ? "yes" : "no") << " oob_err=" << oob_err << " alpha=" << alpha << std::endl;
            }
          }
        }
      }
      oob_err_.push_back(oob_err);
      total_oob_ += oob_err;
    }
  }
  if(w != NULL)
    (*w) = w_copy;
  
}

Scores<double> RF::classify(const DTDocument* doc){
  Scores<double> similarities(doc->get_id(), doc->get_class());
  std::map<std::string, double> scores;
  std::map<std::string, unsigned int> trees_count;
  for(int i = 0; i < num_trees_; i++) {
    std::map<std::string, double> partial_scores = trees_[i]->get_partial_scores(doc);
      double weight = oob_err_[i];
      if(isnan(weight)){
        //printf("%lf\n", weight);
        continue;
      }
      double adjust = exp(1.0 - weight); //(weight == 0.0 ? 1.0 : weight == 1.0 ? 0.0 : log((1.0-weight)/weight));
      /*#pragma omp critical (print_oob)
      {
        std::cerr << "TEST " << i << " oob=" << weight << " w=" << adjust << std::endl;
      }*/

    std::map<std::string, double>::const_iterator cIt = partial_scores.begin();
    while(cIt != partial_scores.end()) {
      std::map<std::string, double>::iterator it = scores.find(cIt->first);
      if(it == scores.end()) {
        it = (scores.insert(std::make_pair(cIt->first, 0.0))).first;
      }
      it->second += cIt->second * adjust;

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
    if(cIt_t == trees_count.end()) {
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
    double tf_idf = (raw) ? tf : 1.0 + log(tf);
    doc->insert_term(term_id, tf_idf);
  }

  Scores<double> similarities = classify(doc);
  
  get_outputer()->output(similarities);
  delete doc;
}

#endif
