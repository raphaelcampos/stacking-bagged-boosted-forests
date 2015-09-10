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

#include "iti.hpp"

#include "iti_incremental.hpp"

long line_num=0;

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


std::string classify(Node* &node, instance_object* & inst){
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
  iti * idt = new iti_incremental();
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
    idt->add_instance(instance);
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
    std::cout << instance->class_name << " : " << classify(idt->tree, instance) << std::endl;
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

