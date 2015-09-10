#ifndef ITI_BATCH_H__
#define ITI_BATCH_H__

#include "iti.hpp"

class iti_batch : public iti{
  public:
  iti_batch(){}
  iti_batch(unsigned int r) : iti(r) {}
  void add_instance(instance_object* inst) {tree->batch(inst);}
  bool parse_train_line(const std::string& line);
  void train(const std::string&);
};

bool iti_batch::parse_train_line(const std::string &line){
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
  return true;
}

void iti_batch::train(const std::string& train_fn){
   iti::train(train_fn);
   add_instance(NULL);
   std::cerr << "[SUPERVISED CLASSIFIER] Tree height " << tree->height() << std::endl;
}


#endif
