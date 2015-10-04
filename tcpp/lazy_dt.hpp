#ifndef LAZY_DT_H__
#define LAZY_DT_H__

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

#include "dt.hpp"

class LazyDT : public DT{
  public:
    LazyDT(unsigned int r) : DT(r) {}
    void test(const std::string&);
    void parse_test_line(const std::string&);
    void train(const std::string&);
    bool parse_train_line(const std::string&);
    Scores<double> build_classify(const DTDocument*);
    void reset_model();
  private:
    std::map<unsigned int, std::set<const DTDocument*> > inverted_idx;
    void inverted_idx_insert(unsigned int, const DTDocument*);
};

Scores<double> LazyDT::build_classify(const DTDocument* doc){
  while(root_->split_lazy(doc));
  Scores<double> similarities(doc->get_id(), doc->get_class());
  std::map<std::string, unsigned int>::const_iterator cIt = root_->classes_begin();
  while(cIt != root_->classes_end()){
    similarities.add(cIt->first, cIt->second);
    ++cIt;
  }
  return similarities;
}

void LazyDT::reset_model(){
  delete root_;
  while(!stack_.empty()){
    delete stack_.top();
    stack_.pop();
  }

  root_ = new Node();
  root_->make_it_root();
}

void LazyDT::inverted_idx_insert(unsigned int term_id, const DTDocument* doc){
  std::map<unsigned int, std::set<const DTDocument*> >::iterator it = inverted_idx.find(term_id);
  if(it == inverted_idx.end()){
    it = inverted_idx.insert(make_pair(term_id, std::set<const DTDocument*>())).first;
  }
  it->second.insert(doc);
}

bool LazyDT::parse_train_line(const std::string& line){
  std::vector<std::string> tokens;
  Utils::string_tokenize(line, tokens, ";");
  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return false;
  DTDocument * doc = new DTDocument();
  std::string doc_id = tokens[0];
  doc->set_id(doc_id);
  std::string doc_class = tokens[1];
  classes_add(doc_class);
  doc->set_class(doc_class);
  for (size_t i = 2; i < tokens.size()-1; i+=2) {
    unsigned int tf = atoi(tokens[i+1].data());
    unsigned int term_id = atoi(tokens[i].data());
    vocabulary_add(term_id);
    doc->insert_term(term_id, tf);

    inverted_idx_insert(term_id, doc);
  }
  bag_.insert(doc);
  return true;
}

void LazyDT::parse_test_line(const std::string& line){
  std::vector<std::string> tokens;
  Utils::string_tokenize(line, tokens, ";");

  std::set<const DTDocument*> projection;

  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return;

  DTDocument * doc = new DTDocument();
  std::string doc_id = tokens[0];
  doc->set_id(doc_id);
  std::string doc_class = tokens[1];
  doc->set_class(doc_class);
  for (size_t i = 2; i < tokens.size()-1; i+=2) {
    unsigned int tf = atoi(tokens[i+1].data());
    unsigned int term_id = atoi(tokens[i].data());
    doc->insert_term(term_id, tf);

    std::map<unsigned int, std::set<const DTDocument*> >::iterator it = inverted_idx.find(term_id);
    if(it != inverted_idx.end()){
      std::set<const DTDocument*>::iterator it_d = it->second.begin();
      while(it_d != it->second.end()){
        projection.insert(*it_d);
        ++it_d;
      }
    }
  }

  root_->add_document_bag(projection);

  Scores<double> similarities = build_classify(doc);
  reset_model();
  get_outputer()->output(similarities);

  delete doc;
}

void LazyDT::train(const std::string& train_fn){
  SupervisedClassifier::train(train_fn);
}

//same as SupervisedClassifier::test, but removed parallelism
void LazyDT::test(const std::string &test_fn) {
  std::ifstream file(test_fn.data());
  if (file) {
    bool free_out = false;
    if (out == NULL) {
//      if (out_file.empty()) out = new BufferedOutputer(10000);
      if (out_file.empty()) {
        setbuf(stdout, NULL);
        out = new Outputer();
      }
      else out = new FileOutputer(out_file);
      free_out = true;
    }
    std::cerr << "[SUPERVISED CLASSIFIER] Testing..." << std::endl;
    std::string line;
    std::vector<std::string> buffer;
    buffer.reserve(10000);
    while (file >> line) {
      buffer.push_back(line);
      if (buffer.size() == 10000) {
        for (int i = 0; i < static_cast<int>(buffer.size()); i++) {
          parse_test_line(buffer[i]);
        }
        buffer.clear();
      }
    }
    if (!buffer.empty()) {
      for (int i = 0; i < static_cast<int>(buffer.size()); i++) {
        parse_test_line(buffer[i]);
      }
    }
    file.close();
    if (free_out) {
      delete out; out = NULL;
    }
  }
  else {
    std::cerr << "Error while opening training file." << std::endl;
    exit(1);
  }
}

//class PLazyRF : public SupervisedClassifier{
//  
//};

#endif
