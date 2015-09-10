#ifndef DT_HPP__
#define DT_HPP__

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
#include "tree.hpp"

class DT : public SupervisedClassifier{
  public:
    DT(unsigned int r, unsigned int maxh=0) : SupervisedClassifier(r) { root_ = new Node(maxh); root_->make_it_root(); }
    ~DT() { reset_model(); }
    void add_document(const DTDocument* doc) { root_->add_document(doc); }
    void add_document_bag(std::set<const DTDocument*>& bag) { root_->add_document_bag(bag); }
    void build(const double m);
    bool parse_train_line(const std::string&);
    void train(const std::string&);
    void parse_test_line(const std::string&);
    void reset_model();
    Scores<double> classify(const DTDocument*) const;
    std::map<std::string, double> get_partial_scores(const DTDocument*) const;
    void print() const;
  private:
    Node * root_;
    std::set<const DTDocument*> bag_;
    std::stack<StackElement*> stack_;
};

void DT::build(const double m){
  TreeData * td;
  if (raw) root_->use_raw_weights();
  td = root_->split(m);
  if(td->is_enabled()){
    StackElement * el = td->get_left_element();
    StackElement * er = td->get_right_element();
    stack_.push(el);
    stack_.push(er);
  }
  delete td;
  while(!stack_.empty()){
    stack_.top()->get_node()->add_document_bag(stack_.top()->get_bag());
    Node *n = stack_.top()->get_node();
    if (raw) n->use_raw_weights();
    TreeData * td = n->split(m);
    delete stack_.top();
    stack_.pop();
    if(td->is_enabled()){
      StackElement * el = td->get_left_element();
      StackElement * er = td->get_right_element();
      stack_.push(el);
      stack_.push(er);
    }
    delete td;
  }
}

void DT::print() const{
  std::stack< std::pair<Node*, unsigned int> > s;
  s.push(std::make_pair(root_, 0));
  while(!s.empty()){
    std::pair<Node*, unsigned int> sel = s.top();
    s.pop();
    for(int i = 0; i < sel.second; i++){
      std::cout << "|";
    }
    std::cout << "(" << sel.first->get_splitting_info().first << "," << sel.first->get_splitting_info().second << ")" << std::endl;
    if(sel.first->left())
      s.push(std::make_pair(sel.first->left(), sel.second+1));
    if(sel.first->right())
      s.push(std::make_pair(sel.first->right(), sel.second+1));
  }
}

void DT::train(const std::string& train_fn){
  SupervisedClassifier::train(train_fn);
  root_->add_document_bag(bag_);
  build(1.0);
  print();
}

bool DT::parse_train_line(const std::string& line){
  std::vector<std::string> tokens; tokens.reserve(100);
  Utils::string_tokenize(line, tokens, ";");
  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return false;
  DTDocument * doc = new DTDocument();
  std::string doc_id = tokens[0];
  doc->set_id(doc_id);
  std::string doc_class = tokens[1];
  classes_add(doc_class);
  doc->set_class(doc_class);
  for (size_t i = 2; i < tokens.size()-1; i+=2) {
    double tf = atof(tokens[i+1].data());
    unsigned int term_id = atoi(tokens[i].data());
    vocabulary_add(term_id);
    doc->insert_term(term_id, tf);
  }
  bag_.insert(doc);
  return true;
}
void DT::parse_test_line(const std::string& line){
  std::vector<std::string> tokens; tokens.reserve(100);
  Utils::string_tokenize(line, tokens, ";");

  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return;

  DTDocument * doc = new DTDocument();
  std::string doc_id = tokens[0];
  doc->set_id(doc_id);
  std::string doc_class = tokens[1];
  doc->set_class(doc_class);
  for (size_t i = 2; i < tokens.size()-1; i+=2) {
    double tf = atof(tokens[i+1].data());
    unsigned int term_id = atoi(tokens[i].data());
    doc->insert_term(term_id, tf);
  }

  Scores<double> similarities = classify(doc);
  delete doc;
  get_outputer()->output(similarities);
}
void DT::reset_model(){
  std::set<const DTDocument*>::const_iterator cIt = bag_.begin();
  while(cIt != bag_.end()){
    delete *cIt;
    ++cIt;
  }
  bag_.clear();
  delete root_;
  while(!stack_.empty()){
    delete stack_.top();
    stack_.pop();
  }
}
Scores<double> DT::classify(const DTDocument* doc) const{
  Node * current_node = root_;
  while(!current_node->is_leaf()){
    if(current_node->splitting_term_in_doc(doc))
      current_node = current_node->right();
    else
      current_node = current_node->left();
  }

  Scores<double> similarities(doc->get_id(), doc->get_class());
  std::map<std::string, double>::const_iterator cIt = current_node->classes_begin();
  while(cIt != current_node->classes_end()){
    similarities.add(cIt->first, cIt->second);
    ++cIt;
  }
  return similarities;
}

std::map<std::string, double> DT::get_partial_scores(const DTDocument* doc) const{
  std::map<std::string, double> results;
  Node * current_node = root_;
  while(!current_node->is_leaf()){
    if(current_node->splitting_term_in_doc(doc))
      current_node = current_node->right();
    else
      current_node = current_node->left();
  }
  std::map<std::string, double>::const_iterator cIt = current_node->classes_begin();
  while(cIt != current_node->classes_end()){
    std::map<std::string, double>::iterator it = (results.insert(std::make_pair(cIt->first, 0.0))).first;
    //it->second = (static_cast<double>(cIt->second) / current_node->get_docs_count())
    //             * (1 - static_cast<double>(root_->get_class_count(cIt->first)) / root_->get_docs_count());
    it->second = cIt->second / current_node->get_docs_count();
    //std::cerr << "cIt->second: " << static_cast<double>(cIt->second) << " current_node->docs_count: " << current_node->get_docs_count() <<
    //             " root->class_count(cl): " << static_cast<double>(root_->get_class_count(cIt->first)) << " root->docs_count: " << root_->get_docs_count() << std::endl;
    ++cIt;
  }
  return results;
}

#endif
