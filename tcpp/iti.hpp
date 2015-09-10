#ifndef ITI_H__
#define ITI_H__

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

class count_object{
  public:
  count_object() : count(0), metric_value(0.0){ }
  ~count_object() { class_counts.clear(); }
  void add(count_object*);
  std::map<std::string, int> class_counts;
  double metric_value;
  int count;
};


class instance_object{
  public:
  instance_object() : class_name("") {}
  ~instance_object() { value.clear(); }
  void add_term(int id, double val){
    value.insert(std::pair<int, double>(id, val));
  }
  void set_class(std::string cn){
    class_name = cn;
  }
  void set_id(std::string i){
    id = i;
  }
  std::string id;
  std::map<int, double> value;
  std::string class_name;
};


class variable_object{
  public:
  variable_object();
  variable_object(std::string , double , int );
  ~variable_object();
  void increment_value_count(std::string , double , int );
  void add(variable_object*);
  std::map<std::string , int> class_counts;
  double 
    cutpoint,
    new_cutpoint,
    metric_value;
  int count;
};


class NodeITI{
  public:
  NodeITI();
  NodeITI(NodeITI*&, NodeITI*&);
  ~NodeITI();
  void add_variable_info(instance_object* );
  void remove_all_variable_info();
  void add_instance(instance_object* );
  void batch(instance_object*);
  void RF_batch(instance_object*, std::vector<int> &, int, int, int);
  int pick_best_variable_by_indirect_metric();
  int RF_pick_best_variable(std::vector<int> &, int);
  void ensure_best_variable(NodeITI*&);
  void pull_up(NodeITI*&, std::string, double);
  void transpose(NodeITI*&, std::string, double);
  void rebuild(NodeITI*&, NodeITI*, NodeITI*, std::string, double);
  int height(int level);
  void do_clean(NodeITI *&);
  void destroy_tree();
  std::list< instance_object* > instances;
  std::map<std::string , int> class_counts;
  int count;
  std::map<int , variable_object* > variables;
  NodeITI
    * right,
    * left;
  int best_variable;
  static long how_many;
  static long deleted;
};
long NodeITI::how_many = 0;
long NodeITI::deleted = 0;

void string_tokenize(const std::string &str,
                       std::vector<std::string> &tokens,
                       const std::string &delimiters = " ") {
    std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    std::string::size_type pos = str.find_first_of(delimiters, lastPos);
    while (std::string::npos != pos || std::string::npos != lastPos) {
      tokens.push_back(str.substr(lastPos, pos - lastPos));
      lastPos = str.find_first_not_of(delimiters, pos);
      pos = str.find_first_of(delimiters, lastPos);
    }
  }

class iti : public SupervisedClassifier{
  public: 
  NodeITI * tree;
  virtual void add_instance(instance_object*) =0;
  virtual bool parse_train_line(const std::string&);
  void parse_test_line(const std::string&);
  void reset_model();
  Scores<double> classify(instance_object*, NodeITI*);

  iti(){
    tree = new NodeITI();
  }
  iti(unsigned int r) : SupervisedClassifier(r) {
    tree = new NodeITI();
  }
};

//not used in RF
bool iti::parse_train_line(const std::string &line){
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
  add_instance(instance);
  //tree->ensure_best_variable(tree);
  return true;
}

Scores<double> iti::classify(instance_object* inst, NodeITI * node){
  if(node->left || node->right){
    std::map<int, double>::const_iterator it_c = inst->value.find(node->best_variable);
    std::map<int, variable_object*>::const_iterator it_v = node->variables.find(node->best_variable);
    if(it_c == inst->value.end())
      return classify(inst, node->right);
    else
      return classify(inst, node->left); 
  }
  else{
    Scores<double> similarities(inst->id, inst->class_name);
    std::set<std::string>::const_iterator cIt = classes_begin();
    while(cIt != classes_end()){
      std::map<std::string, int>::const_iterator it = node->class_counts.find(*cIt);
      if(it == node->class_counts.end()){
        similarities.add(*cIt, 0.0);
      }
      else{
        similarities.add(*cIt, static_cast<double>(it->second));
      }
      ++cIt;
    }
    return similarities;
  }
}

void iti::parse_test_line(const std::string &line){
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

  Scores<double> similarities = classify(instance, tree);
  
  get_outputer()->output(similarities);

}
void NodeITI::do_clean(NodeITI*& caller){
  if(left || right){
    remove_all_variable_info();
    if(this != caller)
      class_counts.clear();
    if(left)
      left->do_clean(caller);
    if(right)
      right->do_clean(caller);
  }
}
void NodeITI::destroy_tree(){
  
  instances.clear();

  if(left){ left->destroy_tree(); delete left; NodeITI::deleted++;}
  if(right){ right->destroy_tree(); delete right; NodeITI::deleted++;}
  
}

void iti::reset_model(){
  tree->destroy_tree();
  tree = new NodeITI();
}

void count_object::add(count_object* c){

  count += c->count;
  metric_value = metric_value > c->metric_value ? metric_value : c->metric_value;
  
  std::map<std::string, int>::const_iterator it_cc = c->class_counts.begin(); //changed iterator to const iterator

  while(it_cc != c->class_counts.end()){
    std::map<std::string, int>::iterator it_cc_current = class_counts.find(it_cc->first); //must be iterator (element modified)
    if(it_cc_current == class_counts.end()){
      class_counts.insert(std::pair<std::string, int>(it_cc->first, it_cc->second));
    }
    else{
      it_cc_current->second += it_cc->second;
    }
    ++it_cc;
  }

}


variable_object::variable_object(){
  cutpoint = new_cutpoint = metric_value = 0.0;
  count = 0;
}

variable_object::~variable_object(){
  class_counts.clear();
}


void variable_object::increment_value_count(std::string class_name, double value, int freq){
  if(class_counts.find(class_name) == class_counts.end()){
    class_counts.insert(std::pair<std::string, int>(class_name, 1));
  }
  else{
    class_counts[class_name]++;
  }
}

void variable_object::add(variable_object* v){
  count += v->count;
  metric_value = metric_value > v->metric_value ? metric_value : v->metric_value;

  std::map<std::string, int>::const_iterator it_cc = v->class_counts.begin();
  
  while(it_cc != v->class_counts.end()){
    std::map<std::string, int>::iterator it_cc_current = class_counts.find(it_cc->first);//must be iterator
    if(it_cc_current == class_counts.end()){
      class_counts.insert(std::pair<std::string, int>(it_cc->first, it_cc->second));
    }
    else{
      it_cc_current->second += it_cc->second;
    }
    ++it_cc;
  }  
}

NodeITI::NodeITI(){
  right = NULL;
  left = NULL;
  count = 0;
  best_variable = -1;
}

//NodeITI info is sum of info at nodes l, r; node left is l and right is r
NodeITI::NodeITI(NodeITI*& l, NodeITI*& r){
  left = l;
  right = r;
  count = 0;

  if(!l && !r) return;

  else if (r){
    std::map<std::string, int>::const_iterator it_cc = r->class_counts.begin(); //changed to const iterator
  
    while(it_cc != r->class_counts.end()){
      std::map<std::string, int>::iterator it_cc_current = class_counts.find(it_cc->first);
      if(it_cc_current == class_counts.end()){
        class_counts.insert(std::pair<std::string, int>(it_cc->first, it_cc->second));
      }
      else{
        it_cc_current->second += it_cc->second;
      }
      ++it_cc;
    }
    std::map<int, variable_object*>::const_iterator it_v = r->variables.begin(); //changed to const iterator
    while(it_v != r->variables.end()){
    std::map<int, variable_object*>::iterator it_v_current = variables.find(it_v->first);
      if(it_v_current == variables.end()){
        variable_object * var = new variable_object();
        it_v_current = variables.insert(std::pair<int, variable_object*>(it_v->first, var)).first;
      }
      it_v_current->second->add(it_v->second);
 
      ++it_v;
    }
    count += r->count;
  }
  
  if(!l) return;

  std::map<std::string, int>::const_iterator it_cc = l->class_counts.begin(); //changed to const iterator
  
  while(it_cc != l->class_counts.end()){
    std::map<std::string, int>::iterator it_cc_current = class_counts.find(it_cc->first);
    if(it_cc_current == class_counts.end()){
      class_counts.insert(std::pair<std::string, int>(it_cc->first, it_cc->second));
    }
    else{
      it_cc_current->second += it_cc->second;
    }
    ++it_cc;
  }

  std::map<int, variable_object*>::const_iterator it_v = l->variables.begin();
  while(it_v != l->variables.end()){
    std::map<int, variable_object*>::iterator it_v_current = variables.find(it_v->first);
    if(it_v_current == variables.end()){
      variable_object * var = new variable_object();
      it_v_current = variables.insert(std::pair<int, variable_object*>(it_v->first, var)).first;
    }

    it_v_current->second->add(it_v->second);
    
    ++it_v;
  }
  count += l->count;
}

NodeITI::~NodeITI(){
  remove_all_variable_info();
  variables.clear();

  instances.clear();

  class_counts.clear();
}

void NodeITI::add_variable_info(instance_object* inst){
  std::map<int, double>::const_iterator it; //changed to const iterator
  it = inst->value.begin();
  while(it != inst->value.end()){
     std::map<int, variable_object* >::iterator it_v = variables.find(it->first);//must be iterator
    if(it_v == variables.end()){
      variable_object * v = new variable_object();
      v->increment_value_count(inst->class_name, it->second, 1);
      v->count++;
      variables.insert(std::pair<int, variable_object* >(it->first, v)); 

    }
    else{
      it_v->second->increment_value_count(inst->class_name, it->second, 1);
      it_v->second->count++;
    }
    ++it;
  }
}

void NodeITI::remove_all_variable_info(){
  std::map<int, variable_object* >::const_iterator it = variables.begin(); //changed to const iterator
  while(it != variables.end()){
    delete it->second;
    ++it;
  }

  variables.clear();

}

void NodeITI::batch(instance_object* inst){
  if(inst){
    class_counts[inst->class_name]++;
    count++;
    add_variable_info(inst);
    instances.push_back(inst);
    
  }
  else{

    if(class_counts.size() <= 1){
      remove_all_variable_info();
      return;
    }
  
    best_variable = pick_best_variable_by_indirect_metric();
      std::map<int , variable_object* >::iterator it_v = variables.find(best_variable);
      if(it_v == variables.end()){
        remove_all_variable_info();
        return;
      }
      it_v->second->cutpoint = it_v->second->new_cutpoint;
      std::list<instance_object*>::const_iterator it_i = instances.begin(); //changed to const iterator
      while(it_i != instances.end()){
        instance_object * new_inst = *it_i;
        std::map<int , double>::const_iterator it_val = new_inst->value.find(best_variable); //changed to const iterator
        if(it_val == new_inst->value.end()){
          if(!right) right = new NodeITI();
          right->batch(new_inst);
        }
        else{
            if(!left) left = new NodeITI();
            left->batch(new_inst);
        }
        ++it_i;
      }
      instances.clear();
      if(left) left->batch(NULL);
      if(right) right->batch(NULL);
   }
}

void NodeITI::RF_batch(instance_object* inst, std::vector<int> &vars, int m, int max_heigth = 5, int depth = 0){
  if(inst){
    class_counts[inst->class_name]++;
    count++;
    add_variable_info(inst);
    instances.push_back(inst);
  }
  else{

    if(class_counts.size() <= 1 || (max_heigth && depth >= max_heigth)){
      remove_all_variable_info();
      return;
    }
  
    best_variable = RF_pick_best_variable(vars, m);
      std::map<int , variable_object* >::iterator it_v = variables.find(best_variable);
      if(it_v == variables.end()){
        remove_all_variable_info();
        return;
      }

      it_v->second->cutpoint = it_v->second->new_cutpoint;
      std::list<instance_object*>::const_iterator it_i = instances.begin(); //changed to const_iterator
      while(it_i != instances.end()){
        instance_object * new_inst = *it_i;
        std::map<int , double>::const_iterator it_val = new_inst->value.find(best_variable); //changed to const_iterator
        if(it_val == new_inst->value.end()){
          if(!right) right = new NodeITI();
          right->RF_batch(new_inst, vars, m);
        }
        else{
          if(!left) left = new NodeITI();
          left->RF_batch(new_inst, vars, m);
        }
        ++it_i;
      }
      instances.clear();
      remove_all_variable_info();
      if(left) left->RF_batch(NULL, vars, m, max_heigth, depth+1);
      if(right) right->RF_batch(NULL, vars, m, max_heigth, depth+1);
   }
}

void NodeITI::add_instance(instance_object* inst){
  std::map<std::string, int>::iterator it_cc = class_counts.find(inst->class_name);//must be iterator
  if(it_cc == class_counts.end()){
    it_cc = class_counts.insert(std::pair<std::string, int>(inst->class_name, 0)).first;
  }
  (it_cc->second)++;
  count++;
  if(left || right){
    add_variable_info(inst);
    std::map<int, variable_object* >::const_iterator it_v = variables.find(best_variable); //changed to const iterator
    if(it_v == variables.end()){
      std::cerr << "best_variable value not defined" << std::endl;
      exit(1);
    }
    std::map<int, double>::const_iterator it_val = inst->value.find(best_variable); //changed iterator to const iterator
    if(it_val == inst->value.end()){
      if(!right) right = new NodeITI();
      right->add_instance(inst);
    }
    else{
      if(it_val->second <= it_v->second->cutpoint){
        if(!left) left = new NodeITI();
        left->add_instance(inst);
      }
      else{
        if(!right) right = new NodeITI();
        right->add_instance(inst);
      }
    }
  }
  else{
    instances.push_back(inst);
    if(class_counts.size() > 1){
       std::list<instance_object* >::const_iterator it_i = instances.begin(); //changed to const iterator
      while(it_i != instances.end()){
        add_variable_info(*it_i);
        ++it_i;
      }
      best_variable = pick_best_variable_by_indirect_metric();
      std::map<int, variable_object* >::iterator it_v = variables.find(best_variable); //must be iterator
      it_v->second->cutpoint = it_v->second->new_cutpoint;
      it_i = instances.begin();
      while(it_i != instances.end()){
        instance_object * new_inst = *it_i;
        std::map<int , double>::const_iterator it_val = new_inst->value.find(best_variable); //changed iterator to const iterator
        if(it_val == new_inst->value.end()){
          if(!right) right = new NodeITI();
          right->add_instance(new_inst);
        }
        else{
          if(it_val->second <= it_v->second->cutpoint){
            if(!left) left = new NodeITI();
            left->add_instance(new_inst);
          }
         else{
            if(!right) right = new NodeITI();
            right->add_instance(new_inst);
          }
        }
        ++it_i;
      }
      instances.clear();
    }
  }
}

//not used in RF
int NodeITI::pick_best_variable_by_indirect_metric(){
  int best_variable_return = -1;
  double best_metric = 0.0;
  double p_tc_best_var = 0.0;
  double ig, parc1, parc2, parc3, cutpoint, p_tc;
  std::map<int, variable_object* >::iterator it_v = variables.begin();
  while(it_v != variables.end()){
    ig = 0.0; cutpoint = 0.0; p_tc = 0.0;
    parc1 = parc2 = parc3 = 0.0;
    std::map<std::string , int>::iterator it_c = class_counts.begin();
    while(it_c != class_counts.end()){
      double n = count;
      double n_c = it_c->second;
      double n_k = it_v->second->count;
      double n_kc = it_v->second->class_counts.find(it_c->first)->second;

      double p_c  = n <= 0 ? 0 : n_c / n;
      double p_nc = n <= 0 ? 0 : (n - n_c) / n;

      double p_k  = n <= 0 ? 0 : n_k / n;
      double p_nk = n <= 0 ? 0 : (n - n_k) / n;

      double p_kc   = n <= 0 ? 0 : n_kc / n;
      double p_nkc  = n <= 0 ? 0 : (n_c - n_kc) / n;

      parc1 += p_c <= 0 ? 0 : p_c  * log(p_c);
      parc2 += p_kc <= 0 ? 0 : p_kc * log(p_kc / p_k);
      parc3 += p_nkc <= 0 ? 0 : p_nkc * log(p_nkc / p_nk);      
      ++it_c;
    }
    ig = - parc1 + parc2 + parc3;

    it_v->second->metric_value = ig;
    if(ig > best_metric){
      best_metric = ig;
      best_variable_return = it_v->first;
    }
    it_v->second->new_cutpoint=cutpoint/it_v->second->count;
    ++it_v;
  }
  if (best_metric == 0 || best_metric == -0)
    return -1;
  return best_variable_return;
}

int NodeITI::RF_pick_best_variable(std::vector<int> &vars, int m){
  std::set<int> m_variables;
  while(m_variables.size() < m){
    #pragma omp critical (bagupdate)
    {
      int r_var = vars[rand() % vars.size()];
      m_variables.insert(r_var);
    }
  }
  
  int best_variable_return = -1;
  double best_metric = 0.0;
  double p_tc_best_var = 0.0;
  double ig, parc1, parc2, parc3, cutpoint, p_tc;
  std::set<int>::const_iterator it_sv = m_variables.begin(); //changed to const iterator
  while(it_sv != m_variables.end()){
    ig = 0.0; cutpoint = 0.0; p_tc = 0.0;
    parc1 = parc2 = parc3 = 0.0;

    std::map<int, variable_object*>::iterator it_v = variables.find(*it_sv); //must be iterator

    if(it_v == variables.end()){
      ++it_sv;
      continue;
    }

    std::map<std::string , int>::const_iterator it_c = class_counts.begin(); //changed to const iterator
    while(it_c != class_counts.end()){
      double n = count;
      double n_c = it_c->second;
      double n_k = it_v->second->count;
      double n_kc = it_v->second->class_counts.find(it_c->first)->second;

      double p_c  = n <= 0 ? 0 : n_c / n;
      double p_nc = n <= 0 ? 0 : (n - n_c) / n;

      double p_k  = n <= 0 ? 0 : n_k / n;
      double p_nk = n <= 0 ? 0 : (n - n_k) / n;

      double p_kc   = n <= 0 ? 0 : n_kc / n;
      double p_nkc  = n <= 0 ? 0 : (n_c - n_kc) / n;

      parc1 += p_c <= 0 ? 0 : p_c  * log(p_c);
      parc2 += p_kc <= 0 ? 0 : p_kc * log(p_kc / p_k);
      parc3 += p_nkc <= 0 ? 0 : p_nkc * log(p_nkc / p_nk);
      
      ++it_c;
    }
    ig = - parc1 + parc2 + parc3;

    it_v->second->metric_value = ig;
    if(ig > best_metric){
      best_metric = ig;
      best_variable_return = it_v->first;
    }
    ++it_sv;
  }
  if (best_metric == 0 || best_metric == -0)
    return -1;
  return best_variable_return;
  
}
int get_number_of_instances(NodeITI * nd){
  using namespace std;
  if(!nd) return -1;
  int i = 0;
  i = nd->instances.size();
  return i;
}

int NodeITI::height(int level=0){
  static int h = 0;
  h = h > level ? h : level;
  if(left)
    left->height(level+1);
  if(right)
    right->height(level+1);
  if(level == 0)
    return h;
}

#endif
