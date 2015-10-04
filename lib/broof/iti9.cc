// TODO:
// * store pointers to objects instead of the objects themselves
// * maybe index all strings in a hashtable, using std::string as int, making operations faster

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

long  line_num=0;

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
  std::map<std::string, double> value;
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
  count_object
    * best_value,
    * new_best_value;
  std::map<double, count_object* >  value_counts;
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
  std::string pick_best_variable_by_indirect_metric();
  void ensure_best_variable(Node*&);
  void pull_up(Node*&, std::string, double);
  void transpose(Node*&, std::string, double);
  void rebuild(Node*&, Node*, Node*, std::string, double);
  std::list< instance_object* > instances;
  std::map<std::string , int> class_counts;
  int count;
  std::map<std::string , variable_object* > variables;
  Node
    * right,
    * left,
    * next;
  std::string best_variable;
  bool stale;
};

void print_node(Node * nd);

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


class ITI {
  Node * tree;
};

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
  best_value = NULL;
  new_best_value = NULL;
}

variable_object::~variable_object(){
  class_counts.clear();
  std::map<double, count_object* >::iterator it = value_counts.begin();
  while(it != value_counts.end()){
    delete it->second;
    ++it;
  }
  value_counts.clear();

  delete best_value;
  delete new_best_value;
}


void variable_object::increment_value_count(std::string class_name, double value, int freq){
   std::map<double, count_object* >::iterator it_vc = value_counts.find(value);
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
  }
  if(class_counts.find(class_name) == class_counts.end()){
    class_counts.insert(std::pair<std::string, int>(class_name, 1));
  }
  else{
    class_counts[class_name]++;
  }
  //value_counts[value].class_count[class_name]+=freq;
  //value_counts[value].count+=freq;
}

void variable_object::add(variable_object* v){
  count += v->count;
  metric_value = metric_value > v->metric_value ? metric_value : v->metric_value;

  std::map<std::string, int>::iterator it_cc = v->class_counts.begin();
  
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

  it_cc = class_counts.begin();

  std::map<double, count_object*>::iterator it_vc = v->value_counts.begin();
  while(it_vc != v->value_counts.end()){
    std::map<double, count_object*>::iterator it_vc_current = value_counts.find(it_vc->first);
    if(it_vc_current == value_counts.end()){
      count_object * co = new count_object();
      it_vc_current = value_counts.insert(std::pair<double, count_object*>(it_vc->first, co)).first;
    }

    it_vc_current->second->add(it_vc->second);
    
    ++it_vc;
  }
  
}

Node::Node(){
  right = NULL;
  left = NULL;
  next = NULL;
  stale = false;
  count = 0;
  best_variable = "";
}

//Node info is sum of info at nodes l, r; node left is l and right is r
Node::Node(Node*& l, Node*& r){
  left = l;
  right = r;
  count = 0;

  if(!l && !r) return;

  else if (r){
    std::map<std::string, int>::iterator it_cc = r->class_counts.begin();
  
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
    std::map<std::string, variable_object*>::iterator it_v = r->variables.begin();
    while(it_v != r->variables.end()){
    std::map<std::string, variable_object*>::iterator it_v_current = variables.find(it_v->first);
      if(it_v_current == variables.end()){
        variable_object * var = new variable_object();
        it_v_current = variables.insert(std::pair<std::string, variable_object*>(it_v->first, var)).first;
      }
      it_v_current->second->add(it_v->second);
 
      ++it_v;
    }
    count += r->count;
  }
  
  if(!l) return;

  std::map<std::string, int>::iterator it_cc = l->class_counts.begin();
  
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

  std::map<std::string, variable_object*>::iterator it_v = l->variables.begin();
  while(it_v != l->variables.end()){
    std::map<std::string, variable_object*>::iterator it_v_current = variables.find(it_v->first);
    if(it_v_current == variables.end()){
      variable_object * var = new variable_object();
      it_v_current = variables.insert(std::pair<std::string, variable_object*>(it_v->first, var)).first;
    }

    it_v_current->second->add(it_v->second);
    
    ++it_v;
  }
  count += l->count;
}

Node::~Node(){
  remove_all_variable_info();
  variables.clear();
  
  //nao eh usado. somente os ponteiros devem ser removidos. os instance_objects, em si, nao
  /*std::list<instance_object *>::iterator it_i = instances.begin();
  while(it_i != instances.end()){
    delete *it_i;
    ++it_i;
  }*/

  instances.clear();

  class_counts.clear();
}

void Node::add_variable_info(instance_object* inst){
  stale = true;
   std::map<std::string, double>::iterator it;
  it = inst->value.begin();
  while(it != inst->value.end()){
     std::map<std::string, variable_object* >::iterator it_v = variables.find(it->first);
    if(it_v == variables.end()){
      variable_object * v = new variable_object();
      v->increment_value_count(inst->class_name, it->second, 1);
      v->count++;
      variables.insert(std::pair<std::string, variable_object* >(it->first, v)); 
    }
    else{
      it_v->second->increment_value_count(inst->class_name, it->second, 1);
      it_v->second->count++;
    }
    ++it;
  }
}

void Node::remove_all_variable_info(){
  std::map<std::string, variable_object* >::iterator it = variables.begin();
  while(it != variables.end()){
    delete it->second;
    ++it;
  }

  variables.clear();

}

void Node::add_instance(instance_object* inst){
  //class_counts[inst->class_name]++;
  std::map<std::string, int>::iterator it_cc = class_counts.find(inst->class_name);
  if(it_cc == class_counts.end()){
    it_cc = class_counts.insert(std::pair<std::string, int>(inst->class_name, 0)).first;
  }
  (it_cc->second)++;
  count++;
  if(left || right){
    add_variable_info(inst);
     std::map<std::string, variable_object* >::iterator it_v = variables.find(best_variable);
    if(it_v == variables.end()){
      std::cerr << "best_variable value not defined" << std::endl;
      exit(1);
    }
    std::map<std::string, double>::iterator it_val = inst->value.find(best_variable);
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
      //implement this one
      best_variable = pick_best_variable_by_indirect_metric();
      std::map<std::string , variable_object* >::iterator it_v = variables.find(best_variable);
      if(it_v->second->value_counts.size() < 2){
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
      }
      //if(it_v == variables.end() || (it_v->second->value_counts.size() < 2 && inst->value.find(best_variable)!=inst->value.end())){
      //  std::cerr << "no best variable returned. consider \"remove_all_variable_info_from_tree_node\"." << std::endl;
      //  variables.clear();
      //  return;
      //}
      it_v->second->cutpoint = it_v->second->new_cutpoint;
      stale = false;
      it_i = instances.begin();
      while(it_i != instances.end()){
        instance_object * new_inst = *it_i;
        std::map<std::string , double>::iterator it_val = new_inst->value.find(best_variable);
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


std::string Node::pick_best_variable_by_indirect_metric(){
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
}

int get_number_of_instances(Node * nd){
  using namespace std;
  if(!nd) return -1;
  int i = 0;
  list<instance_object*>::iterator it_i = nd->instances.begin();
      while(it_i != nd->instances.end()){
        i++;
        ++it_i;
      }
  return i;
}
void Node::ensure_best_variable(Node*& this_node){
  //if(line_num == 188) std::cerr << "Entering ensure_best_variable " << this->left << " " << get_number_of_instances(this->left) << "  " << this << " " << get_number_of_instances(this)<< "  " << this->right << " " << get_number_of_instances(this->right) << std::endl;
  if(!left && !right || !stale)
    return;
  if(this != this_node) std::cerr << "this != this_node on ensure_best_variable" << std::endl << std::flush;
  std::string new_best_variable;
  new_best_variable = pick_best_variable_by_indirect_metric();
  std::map<std::string, variable_object*>::iterator it_v = variables.find(new_best_variable);
  if(best_variable.compare(new_best_variable) || it_v->second->new_cutpoint != it_v->second->cutpoint){
    pull_up(this_node, new_best_variable, it_v->second->new_cutpoint);
  }
  this_node->stale = false;

  if(this_node->left)
    this_node->left->ensure_best_variable(this_node->left);
  if(this_node->right)
    this_node->right->ensure_best_variable(this_node->right);
  //if(line_num == 188) std::cerr << "Exiting ensure_best_variable" << std::endl << std::flush;
  
}

void Node::pull_up(Node*& this_node, std::string var, double new_cut){
  #ifdef DEBUG
  if(this != this_node) std::cerr << "this != this_node on ensure_best_variable" << std::endl << std::flush;
  std::cerr << "Pulling Up node " << this << std::endl << std::flush;
  #endif
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

void print_tree(Node * nd, int level=0);

void Node::transpose(Node*& this_node, std::string var, double new_cut){
  #ifdef DEBUG
  std::cerr << "Entering transpose(" << this << ", " << var << ", " << new_cut << ") ... test=" << best_variable << " : " << variables.find(best_variable)->second->cutpoint <<  std::endl;
  if(left)
    std::cerr << "left=" << left << "; left->left=" << left->left << "; left->right=" << left->right << std::endl;
  if(right)
    std::cerr << "right="<< right << "; right->left=" << right->left << "; right->right=" <<  right->right << std::endl << std::flush;

  print_tree(this);

  #endif
  if(this != this_node) std::cerr << "This != this_node on transpose!!!!!!!!" << std::endl << std::flush;
  if(left){
    if(right){
      Node * ll = left->left,
           * lr = left->right,
           * rl = right->left,
           * rr = right->right;
      if(left->left || left->right){
        if(right->left || right->right){
    //      if(line_num == 188) std::cerr << "Pre Enter Rebuild Left" << std::endl << std::flush;
          left->rebuild(left, ll, rl, var, new_cut);
    //      if(line_num == 188) std::cerr << "Exited Rebuild Left / Entering Right" << std::endl << std::flush;
          right->rebuild(right, lr, rr, var, new_cut);
    //      if(line_num == 188) std::cerr << "Exited Rebuild Right" << std::endl << std::flush;
          
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
     //     if(line_num == 188) std::cerr << "this_node = left" << std::endl << std::flush;
          this_node = left;

          std::list<instance_object *>::iterator it_i = right->instances.begin();
          while(it_i != right->instances.end()){
            this_node->add_instance(*it_i);
            ++it_i;
          }

          //right->instances.clear();
          delete right;
          right = NULL;

          delete this;
        }
      }
      else{
        if(right->right || right->left){
     //     if(line_num == 188) std::cerr << "this_node = right" << std::endl << std::flush;
          this_node = right;

          std::list<instance_object *>::iterator it_i = left->instances.begin();
          while(it_i != left->instances.end()){
            this_node->add_instance(*it_i);
            ++it_i;
          }

          //left->instances.clear();
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
          
          //std::list<instance_object*> left_instances = left->instances;
          //std::list<instance_object*> right_instances = right->instances;
          std::list<instance_object*> insts = left->instances;
          insts.splice(insts.begin(), right->instances);

          delete left;
          left = NULL;

          delete right;
          right = NULL;

      //    if(line_num == 188) std::cerr << "re-adding instances" << std::endl << std::flush;

          std::list<instance_object*>::iterator it_i = insts.begin();
          
          //if(it_v->second->value_counts.size() < 2){
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
          //}
          
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

          /*while(it_il != left_instances.end()){
            std::map<std::string, double>::iterator it_val = (*it_il)->value.find(best_variable);
	    if(it_val == (*it_il)->value.end()){
	      if(!right) right = new Node();
	      right->add_instance(*it_il);
	    }
	    else{
	      if(it_val->second <= it_v->second->cutpoint){
		if(!left) left = new Node();
		left->add_instance(*it_il);
	      }
	      else{
		if(!right) right = new Node();
		right->add_instance(*it_il);
	      }
	    }
            ++it_il;
          }
          
          std::list<instance_object*>::iterator it_ir = right_instances.begin();
          while(it_ir != right_instances.end()){
            std::map<std::string, double>::iterator it_val = (*it_ir)->value.find(best_variable);
	    if(it_val == (*it_ir)->value.end()){
	      if(!right) right = new Node();
	      right->add_instance(*it_ir);
	    }
	    else{
	      if(it_val->second <= it_v->second->cutpoint){
		if(!left) left = new Node();
		left->add_instance(*it_ir);
	      }
	      else{
		if(!right) right = new Node();
		right->add_instance(*it_ir);
	      }
	    }
            ++it_ir;
          }

          left_instances.clear();
          right_instances.clear();*/
        }
        
      }
    }
    else{
  //    if(line_num == 188) std::cerr << "Extern this_node = left" << std::endl << std::flush;
      remove_all_variable_info();
      this_node = left;
      delete this;
    }
  }
  else if(right){
 //   if(line_num == 188) std::cerr << "Extern this_node = right" << std::endl << std::flush;
    remove_all_variable_info();
    this_node = right;
    delete this;
  }
  this_node->stale = true;
  #ifdef DEBUG
  print_tree(this_node);
  std::cerr << "\n--------------------------------------------------------------------------" << std::endl << std::flush;

  #ifdef STEP_DEBUG
  getchar();
  #endif

  #endif
}

void Node::rebuild(Node*& this_node, Node * l, Node * r, std::string var, double new_cut){
  #ifdef DEBUG
  std::cout << "Entering rebuild(" << this_node << "," << l << "," << r << "," << var << "," << new_cut << ")..." << std::endl;
  #endif
  if(l){
    if(r){
      if(l->left || l->right){
        if(r->left || r->right){
          #ifdef DEBUG
          std::cerr << "l not null, r not null, l->left || l->right not null, r->left || r->right not null" << std::endl;
          #endif
          //this_node = NULL;
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
          #ifdef DEBUG
          std::cerr << "l not null, r not null, l->left || l->right not null, r->left && r->right NULL" << std::endl;
          #endif
          this_node = l;
          std::list<instance_object *>::iterator it_i = r->instances.begin();
          while(it_i != r->instances.end()){
            this_node->add_instance(*it_i);
            ++it_i;
          }
          delete r;
          //r->instances.clear();
        }
      }
      else{
        #ifdef DEBUG
        std::cerr << "l not null, r not null, l->left && l->right NULL, r->left && r->right NULL" << std::endl;
        #endif
        this_node = r;
        std::list<instance_object *>::iterator it_i = l->instances.begin();
        while(it_i != l->instances.end()){
          this_node->add_instance(*it_i);
          ++it_i;
        }
        delete l;
        //l->instances.clear();
      }
    }
    else{
      #ifdef DEBUG
      std::cerr << "l not null, r NULL, l->left && l->right NULL" << std::endl;
      #endif
      this_node = l;
    }
  }
  else{
    #ifdef DEBUG
    std::cerr << "l NULL" << std::endl;
    #endif
    this_node = r;
  }
  delete this;
  #ifdef DEBUG
  std::cerr << "Exiting rebuild..." << std::endl;
  #endif
}

void print_node(Node * nd){
  if(nd){
    {using namespace std;
      cout << "This Node: " << nd << endl;
      cout << "Node Left: " << nd->left << endl;
      cout << "Node Right: " << nd->right << endl;
      cout << "Node Info:" << endl;
      cout << "  Best Variable: " << nd->best_variable << endl;
      map<string, variable_object*>::iterator it_cutpnt = nd->variables.find(nd->best_variable);
      if(it_cutpnt != nd->variables.end())
        cout << "  Cutpoint: " << it_cutpnt->second->cutpoint << endl;
      cout << "  Instances:" << endl;
      list<instance_object*>::iterator it_i = nd->instances.begin();
      while(it_i != nd->instances.end()){
        cout << "    " << (*it_i)->class_name << " ";
        map<std::string, double>::iterator it_vals = (*it_i)->value.begin();
        for(;it_vals != (*it_i)->value.end();++it_vals)
          cout << "(" << it_vals->first << " , "<< it_vals->second << ") ";
        cout << endl;
        ++it_i;
      }
      cout << "  Class Counts:" << endl;
      map<string, int>::iterator it_cc = nd->class_counts.begin();
      for(;it_cc != nd->class_counts.end();++it_cc)
        cout << "    " << it_cc->first << " " << it_cc->second << endl;
      map<string, variable_object*>::iterator it_v = nd->variables.begin();
      cout << "  Variables:" << endl;
      for(;it_v != nd->variables.end(); ++it_v){
        cout << "    " << it_v->first << " : " << it_v->second->count << endl;
        cout << "      Class Counts:" << endl;
        map<string, int>::iterator it_vcc = it_v->second->class_counts.begin();
        for(;it_vcc != it_v->second->class_counts.end();++it_vcc)
          cout << "         " << it_vcc->first << " " << it_vcc->second << endl;
        map<double, count_object*>::iterator it_vc = it_v->second->value_counts.begin();
        for(;it_vc != it_v->second->value_counts.end();++it_vc){
          cout << "      " << it_vc->first << " : " << it_vc ->second->count << endl;
          map<string, int>::iterator it_cc2 = it_vc->second->class_counts.begin();
          for(; it_cc2 != it_vc->second->class_counts.end();++it_cc2)
            cout << "        " << it_cc2->first << " " << it_cc2->second << endl;
        }
      }
    }
  }
  //if(nd->left)
    //print_node(nd->left);
  //if(nd->right)
    //print_node(nd->right);
}

void print_tree(Node * nd, int level){
  if(!nd) return;
  for(int i = 0; i < level; i++) std::cout << "|";
  if(nd->left || nd->right){
    std::cout << "node" << " " << nd->best_variable << " : " << nd->variables.find(nd->best_variable)->second->cutpoint << std::endl << std::flush;
  }
  else{
    std::cout << "node" << " " << nd->class_counts.begin()->first << std::endl << std::flush;
  }

  print_tree(nd->left, level+1);
  print_tree(nd->right, level+1);

}

class IDT{
  public:
  Node * tree;
  IDT(){tree = new Node();}
  std::string classify(Node* &node, instance_object* & inst);
  
};

std::string IDT::classify(Node* &node, instance_object* & inst){
  if(node->left || node->right){
    std::map<std::string, double>::iterator it_c = inst->value.find(node->best_variable);
    std::map<std::string, variable_object*>::iterator it_v = node->variables.find(node->best_variable);
    if(it_c == inst->value.end())
      return classify(node->right, inst);
    else if(it_c->second <= it_v->second->cutpoint){
      return classify(node->left, inst);
    }
    else{
      return classify(node->right, inst);
    }
  }
  else{
    std::stringstream str;
    std::map<std::string, int>::iterator it = node->class_counts.begin();
    while(it != node->class_counts.end()){
      str << it->first << " " << it->second << "; ";
      ++it;
    }
    return str.str();
  }
}

int main(int argc, char* argv[]){
  time_t start_time = time(NULL);
  Node * nd = new Node();
  IDT * idt = new IDT();
  instance_object * instance=NULL;
  std::string line;
  std::vector<std::string> tokens;
  std::ifstream inputFile(argv[1]);
  //long line_num=0;
  while(inputFile >> line){
    std::cerr << "Instance: " << line_num++ << "\r";
    fflush(stderr);
    string_tokenize(line, tokens, ";");
    instance=NULL;
    instance = new instance_object();
    if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) continue;
    instance->class_name=tokens[1].data();
    for (long i = 2; i < tokens.size()-1; i+=2) {
      unsigned int tf = atoi(tokens[i+1].data());
      instance->value.insert(std::pair<std::string, double>(tokens[i].data(), static_cast<double>(tf)));
    }
    idt->tree->add_instance(instance);
    #ifndef NOTR
    idt->tree->ensure_best_variable(idt->tree);
    #endif
    tokens.clear();

  }
  inputFile.close();
  std::cerr << std::endl;
  print_tree(idt->tree);
  
  
  
  std::ifstream cla(argv[2]);
  while(cla >> line){
    string_tokenize(line, tokens, ";");
    instance=NULL;
    instance = new instance_object();
    if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) continue;
    instance->class_name=tokens[1].data();
    for (long i = 2; i < tokens.size()-1; i+=2) {
      unsigned int tf = atoi(tokens[i+1].data());
      instance->value.insert(std::pair<std::string, double>(tokens[i].data(), static_cast<double>(tf)));
    }
    std::cout << instance->class_name << " : " << idt->classify(idt->tree, instance) << std::endl;
    fflush(stdout);
    tokens.clear();

  }

  cla.close();
  
  
  /*
  instance_object* obj = new instance_object();
  obj->class_name="tal";
  obj->value.insert(std::pair<std::string, double>("1", -2.0));
  obj->value.insert(std::pair<std::string, double>("2", -3.0));
  obj->value.insert(std::pair<std::string, double>("3", -4.0));

  instance_object* obj2 = new instance_object();
  obj2->class_name="tal";
  obj2->value.insert(std::pair<std::string, double>("1", -2.0));
  obj2->value.insert(std::pair<std::string, double>("2", -4.0));

  instance_object* obj3 = new instance_object();
  obj3->class_name="tal2";
  obj3->value.insert(std::pair<std::string, double>("1", -3.0));
  obj3->value.insert(std::pair<std::string, double>("2", -1.0));
  obj3->value.insert(std::pair<std::string, double>("3", -2.0));
  nd->add_instance(obj);
  nd->add_instance(obj2);
  nd->add_instance(obj3);
  */
  /*Node * open = idt->tree;
  Node * current;
  open->next = NULL;
  while (open){ 
    current = open;
    open = open->next;
    if (current->left){
      current->left->next = open;
      open = current->left;
    }
    if (current->right){
      current->right->next = open;
      open = current->right;
    }
    print_node(current);
  }*/
  time_t end_time = time(NULL);
  std::cerr << "\nTime: " <<difftime(end_time, start_time) << std::endl;
  return 0;
}
#endif
