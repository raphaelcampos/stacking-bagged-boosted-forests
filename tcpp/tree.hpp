#ifndef TREE_HPP__
#define TREE_HPP__

#include <string>
#include <sstream>
#include <map>
#include <set>
#include <list>
#include <vector>
#include <iterator>
#include <iostream>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <algorithm>
#include <limits>

#include "supervised_classifier.hpp"
#include "utils.hpp"

#define UNDEFINED 99999999

typedef unsigned int TermID;

class TermInfo;
class Node;
class DTDocument;
class TreeData;
class StackElement;
typedef std::map<TermID, TermInfo> TermContent;

void shuffle(unsigned int * array, unsigned int size, unsigned int elements_shuffle){
  unsigned int swp, rnd;
  for(int i = 0; i < elements_shuffle; i++){
    #pragma omp critical (rand_call)
    {
      rnd = rand() % size;
    }
    swp = array[rnd];
    array[rnd] = array[i];
    array[i] = swp;
  }
}

class Node{
  public:
    Node(unsigned int size=0, unsigned int maxh=0);
    ~Node();
    void add_document(const DTDocument*);
    void add_document_bag(const std::set<const DTDocument*>&);
    void clear_documents();
    void clear_terms();
    void clear_classes();
    std::pair<TermID, double> get_splitting_info() const;
    
    void use_raw_weights() { raw_ = true; }

    TermContent::const_iterator find_term(const TermID)const ;
    TermContent::const_iterator term_at(const unsigned int) const;
    TermContent::const_iterator terms_begin() const;
    TermContent::const_iterator terms_end() const;

    bool splitting_term_in_doc(const DTDocument * doc) const;
    bool is_leaf() const;

    std::map<std::string, double>::const_iterator find_class(const std::string&) const;
    double get_class_count(const std::string&) const;
    std::map<std::string, double>::const_iterator classes_begin() const;
    std::map<std::string, double>::const_iterator classes_end() const;

    unsigned int get_docs_count() const { return docs_count_; }

    TreeData * split(const double m);

    void make_it_root() { is_root_ = true; }
    
    Node * left() const;
    Node * right() const;
  private:
    void find_splitting_term(const double m);
    void find_splitting_term();
    void define_split(const int m_int);
    void define_split();
    void define_split_gauss(const int m_int);
    void define_split_gauss();

    bool is_split_left(const DTDocument *d);

    void update_term(const TermID term, const double tf, const std::string &cl);
    std::set<const DTDocument*> documents_;
    std::vector<TermID> * terms_;
    unsigned int num_terms_;
    TermContent term_counts_;
    std::map<std::string, double> class_counts_;
    unsigned int docs_count_;
    Node * left_;
    Node * right_;
    TermID splitting_term_;
    double splitting_term_cutpoint_;
    bool is_root_;
    bool raw_;
    unsigned int maxh_;
    unsigned int size_;
};


class TermInfo{
  public:
    TermInfo() {docs_count_ = 0;}
    TermInfo(const unsigned int c, const std::map<std::string, double>& m){ docs_count_ = c; class_counts_ = m; }
    TermInfo(const unsigned int c) { docs_count_ = c; }
    void clear() { class_counts_.clear(); docs_count_ = 0;}
    void increment() { docs_count_++; }
    void increment(const unsigned int inc) { docs_count_ += inc; }
    void increment_class(const std::string&, const double tf=1.0);
    unsigned int get_docs_count() const { return docs_count_; }
    double get_class_count(const std::string&) const;
//    double get_class_count_sq(const std::string&) const;
  private:
    unsigned int docs_count_;
    std::map<std::string, double> class_counts_;
//    std::map<std::string, double> class_counts_sq_;

};

void TermInfo::increment_class(const std::string& cl, const double inc){
  class_counts_[cl]+=inc;
//  it = class_counts_sq_.find(cl);
//  if(it != class_counts_sq_.end()){
//    (it->second) += (inc*inc);
//  }
//  else{
//    class_counts_sq_[cl] = (inc*inc);
//  }
}

double TermInfo::get_class_count(const std::string& cl) const {
  std::map<std::string, double>::const_iterator cIt = class_counts_.find(cl);
  if(cIt != class_counts_.end()){
    return cIt->second;
  }
  else{
    return 0.0;
  }
}

//double TermInfo::get_class_count_sq(const std::string& cl) const {
//  std::map<std::string, double>::const_iterator cIt = class_counts_sq_.find(cl);
//  if(cIt != class_counts_sq_.end()){
//    return cIt->second;
//  }
//  else{
//    return 0.0;
//  }
//}

class DTDocument{
  public:
    DTDocument() {};
    ~DTDocument() {
      doc_terms_.clear();
    };
    std::string get_id() const;
    void set_id(const std::string&);
    std::string get_class() const;
    void set_class(const std::string&);
    std::map<TermID, double>::const_iterator find_term(const TermID) const;
    std::map<TermID, double>::const_iterator terms_begin() const;
    std::map<TermID, double>::const_iterator terms_end() const;
    void insert_term(const TermID, const double);
    void print(const std::string& fn) const;
  private:
    std::string doc_id_;
    std::string doc_class_;
    std::map<TermID, double> doc_terms_;
};

void DTDocument::print(const std::string& fn) const{
  std::ofstream out_file(fn.data(), std::ios::app);
  out_file << doc_id_ << ";" << doc_class_;
  std::map<TermID, double>::const_iterator cIt = doc_terms_.begin();
  while(cIt != doc_terms_.end()){
    out_file << ";" << cIt->first << ";" << cIt->second;
    ++cIt;
  }
  out_file << std::endl;
}

std::string DTDocument::get_id() const { return doc_id_; }

void DTDocument::set_id(const std::string& id){ doc_id_ = id; }

std::string DTDocument::get_class() const { return doc_class_; }

void DTDocument::set_class(const std::string& cl){ doc_class_ = cl; }

std::map<TermID, double>::const_iterator DTDocument::find_term(const TermID term_id) const {
  return doc_terms_.find(term_id);
}

std::map<TermID, double>::const_iterator DTDocument::terms_begin() const {
  return doc_terms_.begin();
}

std::map<TermID, double>::const_iterator DTDocument::terms_end() const {
  return doc_terms_.end();
}

void DTDocument::insert_term(const TermID term_id, const double term_freq){
  doc_terms_.insert(std::pair<TermID, double>(term_id, term_freq));
}

class StackElement{
  public:
    StackElement() : raw_(false) {}
    ~StackElement() {bag_.clear(); }
    Node * get_node() { return node_; }
    void use_raw_weights() { raw_ = true; }
    std::set<const DTDocument*>& get_bag(){ return bag_; }
    void set_node(Node * nd) {
      node_ = nd;
      if (node_ != NULL && raw_) node_->use_raw_weights();
    }
    void add_document(const DTDocument * doc) { bag_.insert(doc); }
  private:
    Node * node_;
    std::set<const DTDocument*> bag_;
    bool raw_;
};

class TreeData{
  public:
    TreeData();
    ~TreeData();
    void use_raw_weights () { raw_ = true; }
    void add_document_lbag(const DTDocument *);
    void add_document_rbag(const DTDocument *);
    void set_left_node(Node * nd) { left_element_->set_node(nd); }
    void set_right_node(Node * nd) { right_element_->set_node(nd); }
    StackElement * get_right_element() { return right_element_; }
    StackElement * get_left_element() { return left_element_; }
    void enable() { control_ = true; }
    void disable() { control_ = false; }
    bool is_enabled () const { return control_; }
  private:
    StackElement * right_element_;
    StackElement * left_element_;
    bool control_;
    bool raw_;
};

TreeData::TreeData() { 
  right_element_ = new StackElement();
  left_element_ = new StackElement();
  right_element_->set_node(NULL);
  left_element_->set_node(NULL);
  control_ = false;
  raw_ = false;
}
TreeData::~TreeData() {
  if(!control_){
    delete right_element_;
    delete left_element_;
  }
}

void TreeData::add_document_lbag(const DTDocument * doc){
  left_element_->add_document(doc);
}

void TreeData::add_document_rbag(const DTDocument * doc){
  right_element_->add_document(doc);
}


Node::Node(unsigned int size, unsigned int maxh){
  terms_ = new std::vector<TermID>();
  left_= NULL;
  right_= NULL;
  splitting_term_ = UNDEFINED;
  splitting_term_cutpoint_ = 0.0;
  is_root_ = false;
  docs_count_ = 0;
  num_terms_ = 0;
  raw_ = false;
  size_=size;
  maxh_=maxh;
}

std::pair<TermID, double> Node::get_splitting_info() const{
  return std::pair<TermID, double>(splitting_term_, splitting_term_cutpoint_);
}

Node::~Node(){
  clear_documents();
  clear_terms();
  clear_classes();
  delete left_;
  delete right_;
}

bool Node::is_split_left(const DTDocument *d) {
  return (d->find_term(splitting_term_) != d->terms_end());
}

TreeData * Node::split(const double m){
  TreeData * ret = new TreeData();
  if (raw_) ret->use_raw_weights();
  if ((size_ > 0 && size_ == maxh_) || class_counts_.size() < 2){
    ret->disable();
    clear_terms();
    clear_documents();
    return ret;
  }
  else{
    find_splitting_term(m);
    if(splitting_term_ == UNDEFINED){
      ret->disable();
      clear_terms();
      clear_documents();
      return ret;
    }
    else{
      ret->enable();
      size_++;
      left_  = new Node(size_, maxh_); ret->set_left_node(left_);
      right_ = new Node(size_, maxh_); ret->set_right_node(right_);
      if (raw_) {
        left_->use_raw_weights();
        right_->use_raw_weights();
      }
      std::set<const DTDocument*>::const_iterator cIt = documents_.begin();
      while(cIt != documents_.end()){
        if (is_split_left(*cIt)) ret->add_document_rbag(*cIt);
        else ret->add_document_lbag(*cIt);
        ++cIt;
      }
      clear_documents();
      clear_terms();
      if(!is_root_) clear_classes();
      return ret;
    }
  }
}

void Node::define_split_gauss() {
  TermContent::const_iterator cIt_v = term_counts_.begin();
  unsigned long offset = Utils::random(term_counts_.size());
  std::advance(cIt_v, offset);
  splitting_term_ = cIt_v->first;
}

void Node::define_split() {
  double highest_ig = 0.0, ig = 0.0;
  double parc1, parc2, parc3;
  TermContent::const_iterator cIt_v = term_counts_.begin();
  while(cIt_v != term_counts_.end()){
    ig = 0.0;
    parc1 = parc2 = parc3 = 0.0;
    std::map<std::string , double>::const_iterator cIt_c = class_counts_.begin();
    while(cIt_c != class_counts_.end()){
      double n = docs_count_;
      double n_c = cIt_c->second;
      double n_k = cIt_v->second.get_docs_count();
      double n_kc = cIt_v->second.get_class_count(cIt_c->first);

      double p_c, p_nc, p_k, p_nk, p_kc, p_nkc;
      if(n <= 0){
        p_c = 0.0;
        p_nc = 0.0;
        p_k = 0.0;
        p_nk = 0.0;
        p_kc = 0.0;
        p_nkc = 0.0;
      }
      else{
        p_c  = n_c / n;
        p_nc = (n - n_c) / n;

        p_k  = n_k / n;
        p_nk = (n - n_k) / n;

        p_kc   = n_kc / n;
        p_nkc  = (n_c - n_kc) / n;
      }

      if(p_c > 0)
        parc1 += p_c  * log(p_c);
      if(p_kc > 0)
        parc2 += p_kc * log(p_kc / p_k);
      if(p_nkc > 0)
        parc3 += p_nkc * log(p_nkc / p_nk);

      ++cIt_c;
    }
    ig = - parc1 + parc2 + parc3;
    if(ig > highest_ig){
      splitting_term_ = cIt_v->first;
      highest_ig = ig;
    }
    ++cIt_v;
  }
}

void Node::find_splitting_term(){
  /*if (raw_) define_split_gauss();
  else*/ define_split();


  if(splitting_term_ != UNDEFINED){
    double avg_tf = 0.0;
    std::set<const DTDocument*>::const_iterator cIt_d = documents_.begin();
    while(cIt_d != documents_.end()){
      std::map<TermID, double>::const_iterator cIt_td = (*cIt_d)->find_term(splitting_term_);
      if(cIt_td != (*cIt_d)->terms_end()){
        avg_tf += cIt_td->second;
      }
      ++cIt_d;
    }
    avg_tf /= documents_.size();
    splitting_term_cutpoint_ = avg_tf;
  }
}

void Node::define_split_gauss(const int m_int) {
  unsigned int * selected_terms = new unsigned int[num_terms_];
  for(unsigned int i = 0; i < num_terms_; i++) selected_terms[i] = i;
  if(num_terms_ != terms_->size()){
    std::cerr << "bad terms size" << std::endl;
    exit(1);
  }
  shuffle(selected_terms, num_terms_, m_int);

  // FIXME BNS OR GAUSSIAN MODELING
  TermContent::const_iterator cIt_v = term_at(selected_terms[0]);
  splitting_term_ = cIt_v->first;
}

void Node::define_split(const int m_int) {
  double highest_ig = 0.0, smallest_ig = std::numeric_limits<double>::max();
  splitting_term_ = UNDEFINED;
  unsigned int * selected_terms = new unsigned int[num_terms_];
  for(unsigned int i = 0; i < num_terms_; i++) selected_terms[i] = i;
  if(num_terms_ != terms_->size()){
    std::cerr << "bad terms size" << std::endl;
    exit(1);
  }
  shuffle(selected_terms, num_terms_, m_int);

  TermContent::const_iterator cIt_v;
  for(int i = 0; i < m_int; i++){
    cIt_v = term_at(selected_terms[i]);
    double ig = 0.0;
    double parc1 = 0.0, parc2 = 0.0, parc3 = 0.0;
    std::map<std::string, double>::const_iterator cIt_c = class_counts_.begin();
    while(cIt_c != class_counts_.end()){
      double n = docs_count_;
      double n_c = cIt_c->second;
      double n_k = cIt_v->second.get_docs_count();
      double n_kc = cIt_v->second.get_class_count(cIt_c->first);

      double p_c, p_nc, p_k, p_nk, p_kc, p_nkc;
      if(n <= 0){
        p_c = 0.0;
        p_nc = 0.0;
        p_k = 0.0;
        p_nk = 0.0;
        p_kc = 0.0;
        p_nkc = 0.0;
      }
      else{
        p_c  = n_c / n;
        p_nc = (n - n_c) / n;

        p_k  = n_k / n;
        p_nk = (n - n_k) / n;

        p_kc   = n_kc / n;
        p_nkc  = (n_c - n_kc) / n;
      }

     // try out += p_kc * log(p_kc) for each c.
    
      if(p_c > 0)
        parc1 += p_c  * log(p_c);
      if(p_kc > 0)
        parc2 += p_kc * log(p_kc / p_k);
      if(p_nkc > 0)
        parc3 += p_nkc * log(p_nkc / p_nk);

      ++cIt_c;
    }
    ig = - parc1 + parc2 + parc3;
    if(ig > highest_ig){
      splitting_term_ = cIt_v->first;
      highest_ig = ig;
    }
    if (ig < smallest_ig) {
      smallest_ig = ig;
    }
  }
  if (smallest_ig == highest_ig) splitting_term_ = UNDEFINED;
  delete[] selected_terms; selected_terms = NULL;
}

void Node::find_splitting_term(const double m){
  unsigned int m_int = ceil(m * num_terms_);
  /*if (raw_) define_split_gauss(m_int);
  else*/ define_split(m_int);

  if(splitting_term_ != UNDEFINED){
    double avg_tf = 0.0;
    std::set<const DTDocument*>::const_iterator cIt_d = documents_.begin();
    while(cIt_d != documents_.end()){
      std::map<TermID, double>::const_iterator cIt_td = (*cIt_d)->find_term(splitting_term_);
      if(cIt_td != (*cIt_d)->terms_end()){
        avg_tf += cIt_td->second;
      }
      ++cIt_d;
    }
    avg_tf /= documents_.size();
    splitting_term_cutpoint_ = avg_tf;
  }
}

void Node::update_term(const TermID term, const double tf, const std::string& cl){
  TermContent::iterator it = term_counts_.find(term);
  if(it != term_counts_.end()){
    it->second.increment();
    it->second.increment_class(cl,((raw_)?tf:1));
  }
  else{
    terms_->push_back(term);
    num_terms_++;
    TermInfo t;
    t.increment();
    t.increment_class(cl,((raw_)?tf:1));
    term_counts_[term] = t;
  }
}

void Node::add_document(const DTDocument * doc){
  std::string doc_class = doc->get_class();
  docs_count_++;
  std::map<TermID, double>::const_iterator cIt = doc->terms_begin();
  while(cIt != doc->terms_end()){
    update_term(cIt->first, cIt->second, doc_class);
    ++cIt;
  }
  class_counts_[doc_class]++;
  documents_.insert(doc);
}

void Node::add_document_bag(const std::set<const DTDocument*>& bag){
  std::set<const DTDocument*>::const_iterator cIt = bag.begin();
  while(cIt != bag.end()){
    add_document(*cIt);
    ++cIt;
  }
}

void Node::clear_documents(){
  //docs_count_=0;
  documents_.clear();
}

void Node::clear_terms(){
  if(terms_) terms_->clear();
  delete terms_;
  terms_ = NULL;
  num_terms_ = 0;

  TermContent::iterator tIt = term_counts_.begin();
  while(tIt != term_counts_.end()){
    (tIt->second).clear();
    ++tIt;
  }
  term_counts_.clear();
}

void Node::clear_classes(){
  class_counts_.clear();
}

TermContent::const_iterator Node::find_term(const TermID term_id) const{
  TermContent::const_iterator cIt = term_counts_.find(term_id);
  return cIt;
}

TermContent::const_iterator Node::term_at(const unsigned int pos) const {
  if(pos >= terms_->size()){
    std::cerr << "Bad term position (function call term_at)" << std::endl;
    exit(1);
  }
  else{
    return term_counts_.find((*terms_)[pos]);
  }
}


TermContent::const_iterator Node::terms_begin() const{
  return term_counts_.begin();
}

TermContent::const_iterator Node::terms_end() const{
  return term_counts_.end();
}

bool Node::splitting_term_in_doc(const DTDocument * doc) const{
  if(splitting_term_ == UNDEFINED){
    std::cerr << "unexpected splitting_term_in_doc call" << std::endl;
    exit(1);
  }
  double test_tf = 0.0;
  std::map<TermID, double>::const_iterator cIt_ttf = doc->find_term(splitting_term_);
  return (cIt_ttf != doc->terms_end());
  //  test_tf = cIt_ttf->second;
  //return (test_tf > splitting_term_cutpoint_);
}

bool Node::is_leaf() const{
  return (left_ == NULL && right_ == NULL);
}


std::map<std::string, double>::const_iterator Node::find_class(const std::string& cl) const{
  return class_counts_.find(cl);
}

double Node::get_class_count(const std::string& cl) const {
  std::map<std::string, double>::const_iterator cIt = class_counts_.find(cl);
  if(cIt == class_counts_.end())
    return 0.0;
  else
    return cIt->second;
}

std::map<std::string, double>::const_iterator Node::classes_begin() const{
  return class_counts_.begin();
}

std::map<std::string, double>::const_iterator Node::classes_end() const {
  return class_counts_.end();
}

Node * Node::left() const{
  return left_;
}

Node * Node::right() const{
  return right_;
}

#endif
