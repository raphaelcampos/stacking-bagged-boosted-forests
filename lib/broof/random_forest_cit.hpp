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
  //count_object
  //  * best_value,
  //  * new_best_value;
  //std::map<double, count_object* >  value_counts;
  double 
    cutpoint,
    new_cutpoint,
    metric_value;
  int count;
};


class Node{
  public:
  Node();
  Node(Node*&, Node*&);
  ~Node();
  void add_variable_info(instance_object* );
  void remove_all_variable_info();
  void add_instance(instance_object* );
  void batch(instance_object*);
  void RF_batch(instance_object*, std::vector<int> &, int, int, int);
  int pick_best_variable_by_indirect_metric();
  int RF_pick_best_variable(std::vector<int> &, int);
  void ensure_best_variable(Node*&);
  void pull_up(Node*&, std::string, double);
  void transpose(Node*&, std::string, double);
  void rebuild(Node*&, Node*, Node*, std::string, double);
  int height(int level);
  void do_clean(Node *&);
  void destroy_tree();
  std::list< instance_object* > instances;
  std::map<std::string , int> class_counts;
  int count;
  std::map<int , variable_object* > variables;
  Node
    * right,
    * left;
  int best_variable;
  bool stale;
  static long how_many;
  static long deleted;
  //static std::map<std::string, double> TF;
  //static std::map<std::string, double> sumTF;
};
long Node::how_many = 0;
long Node::deleted = 0;
//std::map<std::string, double> Node::TF;
//std::map<std::string, double >Node::sumTF;

//Random generator as suggested by Knuth
class Random{
  public:
  static void init(long seed){
    x_n = seed;
  }
  static long rand(long from, long to){
    x_n = (69069 * x_n) % 2147483648;
    return int(floor(from + x_n/2147483648.0 * (to-from) + 0.5));
  }
  static long rand(){
    x_n = (69069 * x_n) % 2147483648;
    return x_n;
  }
  Random(){
    Random::x_n = time(NULL);
  }
  private:
  static long x_n;
};
long Random::x_n;


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
  Node * tree;
  virtual void add_instance(instance_object*) =0;
  virtual bool parse_train_line(const std::string&);
  void parse_test_line(const std::string&);
  void reset_model();
  Scores<double> classify(instance_object*, Node*);

  iti(){
    tree = new Node();
  }
  iti(unsigned int r) : SupervisedClassifier(r) {
    tree = new Node();
  }
};

// ***if calling this function, uncomment line tree->ensure_best_variable***
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
    vocabulary_add(tokens[i].data());
    instance->add_term(term_id, static_cast<double>(tf));
  }
  add_instance(instance);
  //tree->ensure_best_variable(tree);
  return true;
}

Scores<double> iti::classify(instance_object* inst, Node * node){
  if(node->left || node->right){
    std::map<int, double>::const_iterator it_c = inst->value.find(node->best_variable);
    std::map<int, variable_object*>::const_iterator it_v = node->variables.find(node->best_variable);
    if(it_c == inst->value.end())
      return classify(inst, node->right);
    else// if(it_c->second <= it_v->second->cutpoint){
      return classify(inst, node->left);
    //}
    //else{
    //  return classify(inst, node->right);
    //}
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
  //classes_add(doc_class);
  instance->set_class(doc_class);
  for (size_t i = 2; i < tokens.size()-1; i+=2) {
    unsigned int tf = atoi(tokens[i+1].data());
    int term_id = atoi(tokens[i].data());
    //vocabulary_add(term_id);
    instance->add_term(term_id, static_cast<double>(tf));
  }

  Scores<double> similarities = classify(instance, tree);
  
  get_outputer()->output(similarities);

}
void Node::do_clean(Node*& caller){
  if(left || right){
    remove_all_variable_info();
    if(this != caller)
      class_counts.clear();
    if(left)
      left->do_clean(caller);
    if(right)
      right->do_clean(caller);
  }
  //else{
  //  instances.clear();
  //}
}
void Node::destroy_tree(){
  //std::list<instance_object *>::iterator it_i = instances.begin();
  //while(it_i != instances.end()){
  //  delete *it_i;
  //  ++it_i;
  //}
  instances.clear();
  //remove_all_variable_info();

  if(left){ left->destroy_tree(); delete left; Node::deleted++;}
  if(right){ right->destroy_tree(); delete right; Node::deleted++;}
  
}

void iti::reset_model(){
  tree->destroy_tree();
  tree = new Node();
}

void count_object::add(count_object* c){

  count += c->count;
  metric_value = metric_value > c->metric_value ? metric_value : c->metric_value;
  
  std::map<std::string, int>::iterator it_cc = c->class_counts.begin();

  while(it_cc != c->class_counts.end()){
    std::map<std::string, int>::iterator it_cc_current = class_counts.find(it_cc->first);
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
  //best_value = NULL;
  //new_best_value = NULL;
}

variable_object::~variable_object(){
  class_counts.clear();
  //std::map<double, count_object* >::iterator it = value_counts.begin();
  //while(it != value_counts.end()){
  //  delete it->second;
  //  ++it;
  //}
  //value_counts.clear();

  //delete best_value;
  //delete new_best_value;
}


void variable_object::increment_value_count(std::string class_name, double value, int freq){
   /*std::map<double, count_object* >::iterator it_vc = value_counts.find(value);
  if(it_vc == value_counts.end()){
    count_object * c = new count_object();
    c->class_counts.insert(std::pair<std::string, int>(class_name, freq));
    c->count += freq;
    value_counts.insert(std::pair<double, count_object* >(value, c));
  }
  else{
     std::map<std::string, int>::iterator it_cc = it_vc->second->class_counts.find(class_name);
    if(it_cc == it_vc->second->class_counts.end()){
      it_vc->second->class_counts.insert(std::pair<std::string, int>(class_name, freq));
    }
    else{
      it_cc->second += freq;
    }
    it_vc->second->count += freq;
  }*/
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

  std::map<std::string, int>::const_iterator it_cc = v->class_counts.begin(); //changed to const iterator (maybe)
  
  while(it_cc != v->class_counts.end()){
    std::map<std::string, int>::iterator it_cc_current = class_counts.find(it_cc->first);
    if(it_cc_current == class_counts.end()){
      class_counts.insert(std::pair<std::string, int>(it_cc->first, it_cc->second));
    }
    else{
      it_cc_current->second += it_cc->second;
    }
    ++it_cc;
  }

  //it_cc = class_counts.begin(); //makes no sense

  /*std::map<double, count_object*>::iterator it_vc = v->value_counts.begin();
  while(it_vc != v->value_counts.end()){
    std::map<double, count_object*>::iterator it_vc_current = value_counts.find(it_vc->first);
    if(it_vc_current == value_counts.end()){
      count_object * co = new count_object();
      it_vc_current = value_counts.insert(std::pair<double, count_object*>(it_vc->first, co)).first;
    }

    it_vc_current->second->add(it_vc->second);
    
    ++it_vc;
  }*/
  
}

Node::Node(){
  right = NULL;
  left = NULL;
  stale = false;
  count = 0;
  best_variable = -1;
  //Node::how_many++;
}

//Node info is sum of info at nodes l, r; node left is l and right is r
Node::Node(Node*& l, Node*& r){
  left = l;
  right = r;
  count = 0;

  if(!l && !r) return;

  else if (r){
    std::map<std::string, int>::const_iterator it_cc = r->class_counts.begin(); //changed to const iterator (maybe)
  
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
    std::map<int, variable_object*>::iterator it_v = r->variables.begin();
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

  std::map<std::string, int>::const_iterator it_cc = l->class_counts.begin(); //changed to const iterator (maybe)
  
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

  std::map<int, variable_object*>::iterator it_v = l->variables.begin();
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

Node::~Node(){
  remove_all_variable_info();
  variables.clear();

  instances.clear();

  class_counts.clear();
}

void Node::add_variable_info(instance_object* inst){
  std::map<int, double>::iterator it;
  it = inst->value.begin();
  while(it != inst->value.end()){
     std::map<int, variable_object* >::iterator it_v = variables.find(it->first);
    if(it_v == variables.end()){
      variable_object * v = new variable_object();
      v->increment_value_count(inst->class_name, it->second, 1);
      v->count++;
      variables.insert(std::pair<int, variable_object* >(it->first, v)); 

          }
    else{
      it_v->second->increment_value_count(inst->class_name, it->second, 1);
      it_v->second->count++;
   
      //Node::TF[Utils::get_index(it_v->first, inst->class_name)] += it->second;
      //Node::sumTF[inst->class_name]+=it->second;

    }
    ++it;
  }
}

void Node::remove_all_variable_info(){
  std::map<int, variable_object* >::iterator it = variables.begin();
  while(it != variables.end()){
    delete it->second;
    ++it;
  }

  variables.clear();

}

void Node::batch(instance_object* inst){
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
      /*if(it_v->second->value_counts.size() < 2){
        bool missing = false;
        std::list<instance_object*>::iterator it_i = instances.begin();
        while(it_i != instances.end()){
          if((*it_i)->value.find(best_variable) == (*it_i)->value.end()){
            missing = true;
            break;
          }
          ++it_i;
        }
        if(!missing){
          std::cout << it_v->second->metric_value << std::endl;
          remove_all_variable_info();
          return;
        }
      }*/

      it_v->second->cutpoint = it_v->second->new_cutpoint;
      stale = false;
      std::list<instance_object*>::iterator it_i = instances.begin();
      while(it_i != instances.end()){
        instance_object * new_inst = *it_i;
        std::map<int , double>::iterator it_val = new_inst->value.find(best_variable);
        if(it_val == new_inst->value.end()){
          if(!right) right = new Node();
          right->batch(new_inst);
        }
        else{
          //if(it_val->second <= it_v->second->cutpoint){
            if(!left) left = new Node();
            left->batch(new_inst);
          //}
         //else{
         //   if(!right) right = new Node();
         //   right->batch(new_inst);
         // }
        }
        ++it_i;
      }
      instances.clear();
      if(left) left->batch(NULL);
      if(right) right->batch(NULL);
   }
}

void Node::RF_batch(instance_object* inst, std::vector<int> &vars, int m, int max_heigth = 5, int depth = 0){
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
      stale = false;
      std::list<instance_object*>::iterator it_i = instances.begin();
      while(it_i != instances.end()){
        instance_object * new_inst = *it_i;
        std::map<int , double>::const_iterator it_val = new_inst->value.find(best_variable); //changed iterator to const iterator HERE
        if(it_val == new_inst->value.end()){
          if(!right) right = new Node();
          right->RF_batch(new_inst, vars, m);
        }
        else{
          if(!left) left = new Node();
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

void Node::add_instance(instance_object* inst){
  std::map<std::string, int>::iterator it_cc = class_counts.find(inst->class_name);
  if(it_cc == class_counts.end()){
    it_cc = class_counts.insert(std::pair<std::string, int>(inst->class_name, 0)).first;
  }
  (it_cc->second)++;
  count++;
  if(left || right){
    add_variable_info(inst);
    stale = true;
    std::map<int, variable_object* >::iterator it_v = variables.find(best_variable);
    if(it_v == variables.end()){
      std::cerr << "best_variable value not defined" << std::endl;
      exit(1);
    }
    std::map<int, double>::const_iterator it_val = inst->value.find(best_variable); //changed iterator to const iterator HERE
    if(it_val == inst->value.end()){
      if(!right) right = new Node();
      right->add_instance(inst);
    }
    else{
      if(it_val->second <= it_v->second->cutpoint){
        if(!left) left = new Node();
        left->add_instance(inst);
      }
      else{
        if(!right) right = new Node();
        right->add_instance(inst);
      }
    }
  }
  else{
    instances.push_back(inst);
    if(class_counts.size() > 1){
       std::list<instance_object* >::iterator it_i = instances.begin();
      while(it_i != instances.end()){
        add_variable_info(*it_i);
        ++it_i;
      }
      stale = true;
      best_variable = pick_best_variable_by_indirect_metric();
      std::map<int, variable_object* >::iterator it_v = variables.find(best_variable);
      /*if(it_v->second->value_counts.size() < 2){
        bool missing = false;
        it_i = instances.begin();
        while(it_i != instances.end()){
          if((*it_i)->value.find(best_variable) == (*it_i)->value.end()){
            missing = true;
            break;
          }
          ++it_i;
        }
        if(!missing){
          remove_all_variable_info();
          return;
        }
      }*/
      
      it_v->second->cutpoint = it_v->second->new_cutpoint;
      stale = false;
      it_i = instances.begin();
      while(it_i != instances.end()){
        instance_object * new_inst = *it_i;
        std::map<int , double>::const_iterator it_val = new_inst->value.find(best_variable); //changed iterator to const iterator HERE
        if(it_val == new_inst->value.end()){
          if(!right) right = new Node();
          right->add_instance(new_inst);
        }
        else{
          if(it_val->second <= it_v->second->cutpoint){
            if(!left) left = new Node();
            left->add_instance(new_inst);
          }
         else{
            if(!right) right = new Node();
            right->add_instance(new_inst);
          }
        }
        ++it_i;
      }
      instances.clear();
    }
  }
}

/*std::string Node::pick_best_variable_by_indirect_metric(){
  std::string best_variable_return = "";
  double best_metric = 0.0;
  double p_tc = 0.0, tf_c = 0.0, sum_tf = 0.0;
  std::map<std::string , variable_object* >::iterator it_v = variables.begin();
  while(it_v != variables.end()){
    p_tc = 0.0;
    tf_c = 0.0;
    sum_tf = 0.0;
    
    std::map<std::string, int>::iterator it_cl = class_counts.begin();
    while(it_cl != class_counts.end()){
      
      ++it_cl;
    }
    
    
    if(ig > best_metric){
      best_metric = ig;
      best_variable_return = it_v->first;
    }
    std::map<double, count_object *>::iterator it_cc = it_v->second->value_counts.begin();
    while(it_cc != it_v->second->value_counts.end()){
      cutpoint += it_cc->first * it_cc->second->count;
      ++it_cc;
    }
    it_v->second->new_cutpoint=cutpoint/it_v->second->count;
    ++it_v;
  }
  return best_variable_return;
}
*/

int Node::pick_best_variable_by_indirect_metric(){
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

      //double cur_pt_c = Node::TF[(it_v->first)+"-"+(it_c->first)] / Node::sumTF[it_c->first];
      //p_tc = p_tc > cur_pt_c ? p_tc : cur_pt_c;
      
      ++it_c;
    }
    ig = - parc1 + parc2 + parc3;

    it_v->second->metric_value = ig;
    if(ig > best_metric){
      //if(ig == best_metric){
        //if(p_tc > p_tc_best_var){
          best_metric = ig;
          best_variable_return = it_v->first;
          //p_tc_best_var = p_tc;
          
        //}
      //}
      //else{
        //best_metric = ig;
        //best_variable_return = it_v->first;
        //p_tc_best_var = p_tc;
      //}
    }
    /*std::map<double, count_object* >::iterator it_cc = it_v->second->value_counts.begin();
    while(it_cc != it_v->second->value_counts.end()){
      cutpoint+=it_cc->first * it_cc->second->count;
      ++it_cc;
    }*/
    it_v->second->new_cutpoint=cutpoint/it_v->second->count;
    ++it_v;
  }
  if (best_metric == 0 || best_metric == -0)
    return -1;
  return best_variable_return;
}

int Node::RF_pick_best_variable(std::vector<int> &vars, int m){
  std::set<int> m_variables;
  while(m_variables.size() < m){
    #pragma omp critical (bagupdate)
    {
      //std::string r_var = vars[Random::rand(0, vars.size()-1)];
      int r_var = vars[rand() % vars.size()];
      m_variables.insert(r_var);
    }
  }
  
  int best_variable_return = -1;
  double best_metric = 0.0;
  double p_tc_best_var = 0.0;
  double ig, parc1, parc2, parc3, cutpoint, p_tc;
  std::set<int>::iterator it_sv = m_variables.begin();
  while(it_sv != m_variables.end()){
    ig = 0.0; cutpoint = 0.0; p_tc = 0.0;
    parc1 = parc2 = parc3 = 0.0;

    std::map<int, variable_object*>::iterator it_v = variables.find(*it_sv);

    if(it_v == variables.end()){
      ++it_sv;
      continue;
    }

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

      //double cur_pt_c = Node::TF[(it_v->first)+"-"+(it_c->first)] / Node::sumTF[it_c->first];
      //p_tc = p_tc > cur_pt_c ? p_tc : cur_pt_c;
      
      ++it_c;
    }
    ig = - parc1 + parc2 + parc3;

    it_v->second->metric_value = ig;
    if(ig > best_metric){
      //if(ig == best_metric){
        //if(p_tc > p_tc_best_var){
          best_metric = ig;
          best_variable_return = it_v->first;
          //p_tc_best_var = p_tc;
          
        //}
      //}
      //else{
        //best_metric = ig;
        //best_variable_return = it_v->first;
        //p_tc_best_var = p_tc;
      //}
    }
    /*std::map<double, count_object* >::iterator it_cc = it_v->second->value_counts.begin();
    while(it_cc != it_v->second->value_counts.end()){
      cutpoint+=it_cc->first * it_cc->second->count;
      ++it_cc;
    }*/
    //it_v->second->new_cutpoint=cutpoint/it_v->second->count;
    ++it_sv;
  }
  if (best_metric == 0 || best_metric == -0)
    return -1;
  return best_variable_return;
  
}

/*std::string Node::pick_best_variable_by_indirect_metric(){
  std::string best_variable_return = "";
  double best_metric = 0.0;
  double ig, parc1, parc2, parc3, cutpoint;
  std::map<std::string , variable_object* >::iterator it_v = variables.begin();
  while(it_v != variables.end()){
    ig = 0.0; cutpoint = 0.0;
    parc1 = parc2 = parc3 = 0.0;
    std::map<std::string , int>::iterator it_c = class_counts.begin();
    while(it_c != class_counts.end()){
      double n = count + 4.0;
      double n_c = it_c->second + 2.0;
      double n_k = it_v->second->count + 2.0;
      double n_kc = it_v->second->class_counts.find(it_c->first)->second + 1.0;

      double p_c  = n_c / n;
      double p_nc = (n - n_c) / n;

      double p_k  = n_k / n;
      double p_nk = (n - n_k) / n;

      double p_kc   = n_kc / n;
      double p_nkc  = (n_c - n_kc) / n;

      parc1 += p_c  * log(p_c);
      parc2 += p_kc * log(p_kc / p_k);
      parc3 += p_nkc * log(p_nkc / p_nk);
      
      ++it_c;
    }
    ig = - parc1 + parc2 + parc3;
    it_v->second->metric_value = ig;
    if(ig > best_metric){
      best_metric = ig;
      best_variable_return = it_v->first;
    }
    std::map<double, count_object* >::iterator it_cc = it_v->second->value_counts.begin();
    while(it_cc != it_v->second->value_counts.end()){
      cutpoint+=it_cc->first * it_cc->second->count;
      ++it_cc;
    }
    it_v->second->new_cutpoint=cutpoint/it_v->second->count;
    ++it_v;
  }
  return best_variable_return;
}*/

int get_number_of_instances(Node * nd){
  using namespace std;
  if(!nd) return -1;
  int i = 0;
  /*list<instance_object*>::iterator it_i = nd->instances.begin();
      while(it_i != nd->instances.end()){
        i++;
        ++it_i;
      }*/
  i = nd->instances.size();
  return i;
}
/*
void Node::ensure_best_variable(Node*& this_node){
  if(!left && !right || !stale)
    return;

  if(this != this_node) std::cerr << "this != this_node on ensure_best_variable" << std::endl << std::flush;
  int new_best_variable;
  new_best_variable = pick_best_variable_by_indirect_metric();
  std::map<int, variable_object*>::iterator it_v = variables.find(new_best_variable);
  if(best_variable != new_best_variable || it_v->second->new_cutpoint != it_v->second->cutpoint){
    pull_up(this_node, new_best_variable, it_v->second->new_cutpoint);
  }
  this_node->stale = false;

  if(this_node->left)
    this_node->left->ensure_best_variable(this_node->left);
  if(this_node->right)
    this_node->right->ensure_best_variable(this_node->right);
}

void Node::pull_up(Node*& this_node, std::string var, double new_cut){
  std::map<std::string, variable_object* >::iterator it_v = variables.find(var);
  if(it_v == variables.end()){
    variable_object * a_var = new variable_object();
    a_var->cutpoint = new_cut;
    variables.insert(std::pair<std::string, variable_object*>(var, a_var));
  }
  if(best_variable.compare(var) || new_cut != it_v->second->cutpoint){
    if(left){
      if(left->left || left->right){
        it_v = left->variables.find(var);
        if(left->best_variable.compare(var) || it_v->second->cutpoint != new_cut){
          left->pull_up(left, var, new_cut);
        }
      }
    }
    if(right){
      if(right->left || right->right){
        it_v = right->variables.find(var);
        if(right->best_variable.compare(var) || it_v->second->cutpoint != new_cut){
          right->pull_up(right, var, new_cut);
        }
      }
    }
    transpose(this_node, var, new_cut);    
  }
}
*/

int Node::height(int level=0){
  static int h = 0;
  h = h > level ? h : level;
  if(left)
    left->height(level+1);
  if(right)
    right->height(level+1);
  if(level == 0)
    return h;
}

/*
void Node::transpose(Node*& this_node, std::string var, double new_cut){
  if(left){
    if(right){
      Node * ll = left->left,
           * lr = left->right,
           * rl = right->left,
           * rr = right->right;
      if(left->left || left->right){
        if(right->left || right->right){
          left->rebuild(left, ll, rl, var, new_cut);
          right->rebuild(right, lr, rr, var, new_cut);
          
          best_variable = var;
          std::map<std::string, variable_object* >::iterator it_v = variables.find(best_variable);
          if(it_v == variables.end()){
            variable_object * a_var = new variable_object();
            a_var->cutpoint = new_cut;
            variables.insert(std::pair<std::string, variable_object*>(best_variable, a_var));
          }
          else{
            it_v->second->cutpoint = new_cut;
          }
        }
        else{
          this_node = left;

          std::list<instance_object *>::iterator it_i = right->instances.begin();
          while(it_i != right->instances.end()){
            this_node->add_instance(*it_i);
            ++it_i;
          }

          delete right;
          right = NULL;

          delete this;
        }
      }
      else{
        if(right->right || right->left){
          this_node = right;

          std::list<instance_object *>::iterator it_i = left->instances.begin();
          while(it_i != left->instances.end()){
            this_node->add_instance(*it_i);
            ++it_i;
          }

          delete left;
          left = NULL;

          delete this;
        }
        else{
          best_variable = var;
          std::map<std::string, variable_object* >::iterator it_v = variables.find(best_variable);
          if(it_v == variables.end()){
            variable_object * a_var = new variable_object();
            a_var->cutpoint = new_cut;
            variables.insert(std::pair<std::string, variable_object*>(best_variable, a_var));
          }
          else{
            it_v->second->cutpoint = new_cut;
          }
          
          std::list<instance_object*> insts = left->instances;
          insts.splice(insts.begin(), right->instances);

          delete left;
          left = NULL;

          delete right;
          right = NULL;


          std::list<instance_object*>::iterator it_i = insts.begin();
          
          bool go_left=false, go_right=false;
          
          while(it_i != insts.end()){
            std::map<std::string, double>::iterator it_val = (*it_i)->value.find(best_variable);
            if(it_val == (*it_i)->value.end()){
              go_right = true;
            }
            else{
              if(it_val->second <= it_v->second->cutpoint){
                go_left = true;
              }
              else{
        	go_right = true;
	      }
            }
            if(go_left && go_right)
              break;
            ++it_i;
          }
          if(!go_left || !go_right){
            remove_all_variable_info();
            instances = insts;
            return;
          }
          
          it_i = insts.begin();
          
          while(it_i != insts.end()){
            std::map<std::string, double>::iterator it_val = (*it_i)->value.find(best_variable);
            if(it_val == (*it_i)->value.end()){
	      if(!right) right = new Node();
	      right->add_instance(*it_i);
	    }
	    else{
	      if(it_val->second <= it_v->second->cutpoint){
		if(!left) left = new Node();
		left->add_instance(*it_i);
	      }
	      else{
		if(!right) right = new Node();
		right->add_instance(*it_i);
	      }
	    }
            ++it_i;
          }

          insts.clear();
        }
        
      }
    }
    else{
      remove_all_variable_info();
      this_node = left;
      delete this;
    }
  }
  else if(right){
    remove_all_variable_info();
    this_node = right;
    delete this;
  }
  this_node->stale = true;
}

void Node::rebuild(Node*& this_node, Node * l, Node * r, std::string var, double new_cut){
  if(l){
    if(r){
      if(l->left || l->right){
        if(r->left || r->right){
          this_node = new Node(l, r);
          this_node->stale = true;
          this_node->best_variable = var;
          std::map<std::string, variable_object*>::iterator it_v = this_node->variables.find(var);
          if(it_v == this_node->variables.end()){
            variable_object * v = new variable_object();
            v->cutpoint = new_cut;
            this_node->variables.insert(std::pair<std::string, variable_object*>(var, v));
          }
          else{
            it_v->second->cutpoint = new_cut;
          }
        }
        else{
          this_node = l;
          std::list<instance_object *>::iterator it_i = r->instances.begin();
          while(it_i != r->instances.end()){
            this_node->add_instance(*it_i);
            ++it_i;
          }
          delete r;
        }
      }
      else{
        this_node = r;
        std::list<instance_object *>::iterator it_i = l->instances.begin();
        while(it_i != l->instances.end()){
          this_node->add_instance(*it_i);
          ++it_i;
        }
        delete l;
      }
    }
    else{
      this_node = l;
    }
  }
  else{
    this_node = r;
  }
  delete this;
}
*/
#endif
