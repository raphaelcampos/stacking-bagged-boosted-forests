#include <string>
#include <sstream>
#include <map>
#include <set>
#include <list>
#include <vector>
#include <iostream>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <ctime>

#include "supervised_classifier.hpp"
#include "utils.hpp"
#include "iti.hpp"

class RandomForest : public SupervisedClassifier{
  public:
  std::vector<NodeITI*> trees;
  Scores<double> classify(instance_object*);
  Scores<double> classify_avg(instance_object*);
  Scores<double> classify_perc(instance_object*);
  bool parse_train_line(const std::string&);
  void parse_test_line(const std::string&);
  void reset_model();
  void train(const std::string&);
  void purity_diagnosis();
  void purity_diag(NodeITI*, double*, int*);

  RandomForest(double m=0, int n=10, double d=1.0) : m_percent_(m), num_trees_(n), num_docs_(d) {
    trees.reserve(num_trees_);
  }
  RandomForest(unsigned int r, double m=0, int n=10, double d=1.0, int maxh=0) : SupervisedClassifier(r), num_trees_(n), m_percent_(m), num_docs_(d), max_height_(maxh) {
    trees.reserve(num_trees_);
  }
  ~RandomForest(){
    for(int i = 0; i < num_trees_; i++){
      trees[i]->destroy_tree();
      delete trees[i];
    }
    std::vector<instance_object*>::const_iterator it = instances_.begin();
    while(it != instances_.end()){
      delete *it;
      ++it;
    }
    instances_.clear();
  }
  private:
  std::vector<instance_object*> instances_;
  std::vector<int> variables_;
  int num_trees_;
  double m_percent_;
  double num_docs_;
  int m_;
  int max_height_;
};

void RandomForest::reset_model(){

}

Scores<double> RandomForest::classify(instance_object* inst){
  Scores<double> similarities(inst->id, inst->class_name);
  
  std::set<std::string>::const_iterator cIt = classes_begin();
  while(cIt != classes_end()){
    double score = 0.0;
    for(int i = 0; i < num_trees_; i++){
      NodeITI * node = trees[i];

      if(!node) std::cerr << "NodeITI NULL at start" << std::endl;

      while (node->left || node->right){
        std::map<int, double>::const_iterator it_c = inst->value.find(node->best_variable);
        if(it_c == inst->value.end())
          node = node->right;
        else
          node = node->left;
      }
      std::map<std::string, int>::const_iterator it = node->class_counts.find(*cIt);
      if(it != node->class_counts.end()){
        score += 1.0 / node->class_counts.size();
      }
    }
    similarities.add(*cIt, score);
    
    ++cIt;
  }
  return similarities;
}

Scores<double> RandomForest::classify_avg(instance_object* inst){
  Scores<double> similarities(inst->id, inst->class_name);
  std::set<std::string>::const_iterator cIt = classes_begin();
  while(cIt != classes_end()){
    double score = 0.0;
    int count_trees = 0;

    for(int i = 0; i < num_trees_; i++){
      NodeITI * node = trees[i];      
      if(!node) std::cerr << "NodeITI NULL at start" << std::endl;
      while (node->left || node->right){
        std::map<int, double>::const_iterator it_c = inst->value.find(node->best_variable);
        if(it_c == inst->value.end())
          node = node->right;
        else
          node = node->left;
      }
      std::map<std::string, int>::const_iterator it = node->class_counts.find(*cIt);
      if(it != node->class_counts.end()){
        std::map<std::string, int>::const_iterator it_c_k = trees[i]->class_counts.find(*cIt);
        score += (static_cast<double>(it->second) / static_cast<double>(node->count)) * (1.0 - static_cast<double>(it_c_k->second)/static_cast<double>(trees[i]->count));
        count_trees++;
      }
    }
    if(count_trees == 0)
      similarities.add(*cIt, 0);
    else if(count_trees == 1)
      similarities.add(*cIt, score/log(1.4142));
    else
      similarities.add(*cIt, score/log(count_trees));
    
    ++cIt;
  }
  return similarities;
}

Scores<double> RandomForest::classify_perc(instance_object* inst){
  Scores<double> similarities(inst->id, inst->class_name);
  std::set<std::string>::const_iterator cIt = classes_begin();
  while(cIt != classes_end()){
    double score = 0.0;
    int count_trees = 0;

    for(int i = 0; i < num_trees_; i++){
      NodeITI * node = trees[i];      
      if(!node) std::cerr << "NodeITI NULL at start" << std::endl;

      while (node->left || node->right){
        std::map<int, double>::const_iterator it_c = inst->value.find(node->best_variable);
        if(it_c == inst->value.end())
          node = node->right;
        else
          node = node->left;
      }
      std::map<std::string, int>::const_iterator it = node->class_counts.find(*cIt);
      if(it != node->class_counts.end()){
        std::map<std::string, int>::const_iterator it_c_k = trees[i]->class_counts.find(*cIt);
        score += (static_cast<double>(it->second) / static_cast<double>(node->count)) * (static_cast<double>(trees[i]->count)/(static_cast<double>(it_c_k->second)));
      }
    }
    similarities.add(*cIt, score);
    ++cIt;
  }
  return similarities;
}

bool RandomForest::parse_train_line(const std::string &line){
  std::vector<std::string> tokens;
  Utils::string_tokenize(line, tokens, ";");
  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return false;
  instance_object * instance = new instance_object();
  std::string doc_id = tokens[0];
  instance->set_id(doc_id);
  std::string doc_class = tokens[1];
  classes_add(doc_class);
  instance->set_class(doc_class);
  for (size_t i = 2; i < tokens.size()-1; i+=2) {
    unsigned int tf = atoi(tokens[i+1].data());
    int term_id = atoi(tokens[i].data());
    vocabulary_add(term_id);
    instance->add_term(term_id, static_cast<double>(tf));
  }
  instances_.push_back(instance);
  return true;
}

void RandomForest::parse_test_line(const std::string &line){
  std::vector<std::string> tokens;
  Utils::string_tokenize(line, tokens, ";");

  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return;

  instance_object * instance = new instance_object();
  std::string doc_id = tokens[0];
  instance->set_id(doc_id);
  std::string doc_class = tokens[1];
  instance->set_class(doc_class);
  for (size_t i = 2; i < tokens.size()-1; i+=2) {
    unsigned int tf = atoi(tokens[i+1].data());
    int term_id = atoi(tokens[i].data());
    instance->add_term(term_id, static_cast<double>(tf));
  }

  Scores<double> similarities = classify_avg(instance);
  
  get_outputer()->output(similarities);
  delete instance;
}

void RandomForest::train(const std::string &train_fn){
  if(m_percent_ > 1.0){
    std::cerr << "Number of variables (in percent) must be lesser than or equal to 1.0" << std::endl;
    exit(1);
  }
  
  SupervisedClassifier::train(train_fn);
  if(instances_.size() < 1){
    std::cerr << "Zero docs in train file. Aborting..." << std::endl;
    exit(1);
  }
  std::cerr << "Train file contains " << instances_.size() << " docs." << std::endl;
 
  srand(time(NULL));
  std::set<int>::const_iterator it_voc = vocabulary_begin();
  variables_.reserve(vocabulary_size());
  while(it_voc != vocabulary_end()){
    variables_.push_back(*it_voc);
    ++it_voc;
  }
  if(m_percent_ == 0.0)
    m_ = int(ceil(log(variables_.size())));
  else
    m_ = int(ceil(m_percent_ * variables_.size()));
  std::cerr << "Vocabulary done (" << variables_.size() << " terms, m=" << m_ << ")" << std::endl;
  std::cerr << "Trees max heigth (0 is unlimited): " << max_height_ << std::endl;
  #pragma omp parallel for num_threads(8) schedule(static)
  for(int i = 0; i < num_trees_; i++){
    NodeITI * k_tree;
      k_tree = new NodeITI();
      trees[i] = k_tree;

    std::set<instance_object*> bag;
    for(int j = 0; j < static_cast<int>(ceil(total_docs * num_docs_)); j++){
      #pragma omp critical (bagupdate)
      {
        bag.insert(instances_[rand() % total_docs]);
      }
    }
    
    std::set<instance_object *>::const_iterator it_i = bag.begin();
    while(it_i != bag.end()){
      k_tree->RF_batch(*it_i, variables_, m_);
      ++it_i;
    }
    k_tree->RF_batch(NULL, variables_, m_, max_height_);
    
    #pragma omp critical (printing)
    {
      if(!(i%5))
        std::cerr << ".";
    }
    k_tree->do_clean(k_tree);
  }
  std::cerr << std::endl;
}
