#ifndef RF_KNN_H__
#define RF_KNN_H__

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

class KNN_Term_IDF{
  public:
    KNN_Term_IDF(unsigned int t, double i) : term_id(t), idf(i)/*, max_tf(1.0)*/ {}
    unsigned int term_id;
    bool operator<(const KNN_Term_IDF& b) const {return term_id < b.term_id;}
    mutable double idf; //I don't want KNN_Term_IDF to be const, but std::map stores it as const. Only term_id is used to build std::map internal tree
                        //so changing idf (which is necessary) won't cause any problem in this sense. Using mutable is the easiest way to go then
};

class KNN_Doc_TF{
  public:
    KNN_Doc_TF(const DTDocument* const& d, double tf) : doc(d), tf_idf(tf) {}
    bool operator<(const KNN_Doc_TF& b) const {return doc < b.doc;}
    const DTDocument* doc;
    mutable double tf_idf;
};

class KNN_Doc_Sim{
  public:
    KNN_Doc_Sim(const DTDocument* const& d, double s) : doc(d), sim(s) {}
    bool operator<(const KNN_Doc_Sim& b) const {return sim < b.sim;}
    const DTDocument* doc;
    double sim;
};

class RF_KNN : public SupervisedClassifier{
  public:
    RF_KNN(unsigned int r, double m=1.0, unsigned int k=30, unsigned int num_trees=10)
      : SupervisedClassifier(r), num_trees_(num_trees), k_(k), m_(m), maxIDF(0.0) {docs_processed_ = 0; std::cerr << k << std::endl;}
    ~RF_KNN();
    bool parse_train_line(const std::string&);
    void train(const std::string&);
    void parse_test_line(const std::string&);
    void reset_model();
    Scores<double> classify(const DTDocument*, std::map<const DTDocument*, double>&);
  private:
    std::vector<const DTDocument*> docs_;
    double m_;
    unsigned int k_;
    unsigned int num_trees_;
    
    void insert_knn_term(unsigned int term_id, const DTDocument* const& doc, double tf);
    void updateIDF();
    void updateTFIDF(); //and update doc sizes
    std::map<KNN_Term_IDF, std::set<KNN_Doc_TF> > knn_term_list_;
    std::map<const DTDocument*, double> knn_doc_sizes_;
    double maxIDF;
    unsigned int docs_processed_;
};

void RF_KNN::insert_knn_term(unsigned int term_id, const DTDocument* const& doc, double tf){
  KNN_Term_IDF t_idf(term_id, 0.0);
  KNN_Doc_TF doc_tw(doc, tf);

  std::map<KNN_Term_IDF, std::set<KNN_Doc_TF> >::iterator it = knn_term_list_.find(t_idf);
  if(it == knn_term_list_.end()){
    std::set<KNN_Doc_TF> doc_set;
    doc_set.insert(doc_tw);
    knn_term_list_.insert(std::make_pair(t_idf, doc_set));
  }
  else{
    (it->second).insert(doc_tw);
  }
}


void RF_KNN::updateIDF(){
  if (raw) return;
  std::map<KNN_Term_IDF, std::set<KNN_Doc_TF> >::iterator it = knn_term_list_.begin();
  while(it != knn_term_list_.end()){
    (it->first).idf = log10((static_cast<double>(docs_.size()) + 1.0)/ ((it->second).size() + 1.0));
    if(maxIDF < (it->first).idf) maxIDF = (it->first).idf;
    ++it;
  }
}

void RF_KNN::updateTFIDF(){
  std::map<KNN_Term_IDF, std::set<KNN_Doc_TF> >::iterator it = knn_term_list_.begin();
  while(it != knn_term_list_.end()){
    std::set<KNN_Doc_TF>::iterator it_d = (it->second).begin();
    while(it_d != (it->second).end()){
      if (!raw) it_d->tf_idf *= (it->first).idf / maxIDF;
      knn_doc_sizes_[it_d->doc] += it_d->tf_idf * it_d->tf_idf; //update doc sizes
      ++it_d;
    }
    ++it;
  } 
}

RF_KNN::~RF_KNN(){
  reset_model();
}

void RF_KNN::reset_model(){
  for(int i = 0; i < docs_.size(); i++){
    delete docs_[i];
  }
  docs_.clear();
  m_ = 1.0;
  num_trees_ = 0;
  k_ = 30;
}

bool RF_KNN::parse_train_line(const std::string& line){
  std::vector<std::string> tokens; tokens.reserve(100);
  Utils::string_tokenize(line, tokens, ";");
  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return false;
  DTDocument * doc = new DTDocument();
  double maxTF = 0;
  std::string doc_id = tokens[0];
  doc->set_id(doc_id);
  std::string doc_class = tokens[1];
  doc->set_class(doc_class);
  for (size_t i = 2; i < tokens.size()-1; i+=2) {
    double tf = atof(tokens[i+1].data());
    int term_id = atoi(tokens[i].data());
    doc->insert_term(term_id, tf);
    if(tf > maxTF){
      maxTF = tf;
    }
    insert_knn_term(term_id, doc, (raw) ? tf : 1.0+log(tf));
  }
  docs_.push_back(doc);
  return true;

}

void RF_KNN::train(const std::string& train_fn){
  SupervisedClassifier::train(train_fn);
  updateIDF();
  updateTFIDF();
}

Scores<double> RF_KNN::classify(const DTDocument* doc, std::map<const DTDocument*, double>& sim){
  Scores<double> similarities(doc->get_id(), doc->get_class());
  std::priority_queue<KNN_Doc_Sim, std::vector<KNN_Doc_Sim> > ordered_docs;
  std::map<const DTDocument*, double>::iterator it = sim.begin();
  while(it != sim.end()){
    double s = it->second;
    switch(dist_type) {
      case L2:
        s = 1.0 - sqrt(s);
        break;
      case L1:
        s = 1.0 - s;
        break;
    }
    KNN_Doc_Sim pqel(it->first, s);
    ordered_docs.push(pqel);
    ++it;
  }
  RF * rf = new RF(round, m_, num_trees_);
  if (raw) rf->use_raw_weights();
  rf->set_doc_delete(false);
  unsigned count = 0;
  while(!ordered_docs.empty() && count < k_){
    KNN_Doc_Sim pqel = ordered_docs.top();
    rf->add_document(pqel.doc);
    ordered_docs.pop();
    count++;
  }

  rf->build();

  similarities = rf->classify(doc);

  delete rf;
  return similarities;

  /*
  unsigned count = 0;
  std::map<std::string, double> class_scores;
  while(!ordered_docs.empty() && count < k_){
    KNN_Doc_Sim pqel = ordered_docs.top();
    class_scores[pqel.doc->get_class()] += pqel.sim;
    ordered_docs.pop();
    count++;
  }
  std::map<std::string, double>::iterator it_c = class_scores.begin();
  while(it_c != class_scores.end()){
    similarities.add(it_c->first, it_c->second);
    ++it_c;
  }
  return similarities;
  */
}

void RF_KNN::parse_test_line(const std::string& line){
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
    KNN_Term_IDF knn_t(term_id, 0.0);

    double tf_idf = (raw) ? tf : 1.0 + log(tf);
    std::map<KNN_Term_IDF, std::set<KNN_Doc_TF> >::iterator it = knn_term_list_.find(knn_t);
    if(it != knn_term_list_.end()){
      if (!raw) tf_idf *= (it->first).idf / maxIDF;
      test_size += tf_idf * tf_idf;
    }

    doc->insert_term(term_id, tf);
  }

  std::map<TermID, double>::const_iterator cit_t = doc->terms_begin();
  while(cit_t != doc->terms_end()){
    KNN_Term_IDF knn_t(cit_t->first, 0.0);
    std::map<KNN_Term_IDF, std::set<KNN_Doc_TF> >::iterator it_d = knn_term_list_.find(knn_t);
    if(it_d != knn_term_list_.end()){
      std::set<KNN_Doc_TF>::iterator it_s = (it_d->second).begin();
      while(it_s != (it_d->second).end()){
        double tf_idf = (raw) ? cit_t->second : 1.0 + log(cit_t->second);
        if (!raw) tf_idf *= (it_d->first).idf / maxIDF;
        switch(dist_type) {
          case L2:
            doc_similarities[it_s->doc] += pow((it_s->tf_idf/sqrt(knn_doc_sizes_[it_s->doc])) - (tf_idf/sqrt(test_size)), 2.0);
            break;
          case L1:
            doc_similarities[it_s->doc] += abs((it_s->tf_idf/sqrt(knn_doc_sizes_[it_s->doc])) - (tf_idf/sqrt(test_size)));
            break;
          case COSINE:
          default:
            doc_similarities[it_s->doc] += (it_s->tf_idf/sqrt(knn_doc_sizes_[it_s->doc])) * (tf_idf/sqrt(test_size));
            break;
        }
        ++it_s;
      }
    }
    ++cit_t;
  }

  Scores<double> similarities = classify(doc, doc_similarities);
  docs_processed_++;
  std::cerr.precision(4);
  std::cerr.setf(std::ios::fixed);
  std::cerr << "\r" << double(docs_processed_)/docs_.size() * 900 << "%";
  
  get_outputer()->output(similarities);
  delete doc;
}

#endif
