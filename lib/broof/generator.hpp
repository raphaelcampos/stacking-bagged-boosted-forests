#ifndef GENERATOR_HPP__
#define GENERATOR_HPP__

#include "selector.hpp"

class Generator {
 public:
  Generator() : parsed_(false), next_id_(0) {srand(123);}
  // copy constructor
  Generator (const Generator &rhs) {
    std::map<std::string,
      std::map<unsigned int,
        std::vector<std::string> > >::const_iterator c_it = rhs.instances_.begin();
    while (c_it != rhs.instances_.end()) {
      std::map<unsigned int,
        std::vector<std::string> >::const_iterator p_it = c_it->second.begin();
      while (p_it != c_it->second.end()) {
        for (unsigned int i = 0; i < p_it->second.size(); i++) {
          instances_[c_it->first][p_it->first].push_back(p_it->second.at(i));
        }
        ++p_it;
      }
      ++c_it;
    }
    parsed_ = true;
  }

  virtual void update_model(const std::string &c,
                            const unsigned int p,
                            const std::string &ln) {
    instances_[c][p].push_back(ln);
  }

  virtual std::string generate_sample(const unsigned int doc_id,
                                      const std::string &c,
                                      const unsigned int p_source,
                                      const unsigned int p_target) = 0;

  unsigned int size(const std::string &c, const unsigned int p) {
    return instances_[c][p].size();
  }

  virtual Generator *clone() = 0;

  virtual ~Generator() {}

 protected:

  std::map<std::string,
    std::map<unsigned int,
      std::vector<std::string> > > instances_;

  virtual void parse(const std::string &fn) {
    if (parsed_) return;
    std::ifstream file(fn.data());
    if (file) {
      std::string line;
      while (std::getline(file, line)) {
        std::vector<std::string> tokens;
        Utils::string_tokenize(line, tokens, ";");
        unsigned int id = atoi(tokens[0].data());
        if (next_id_ < id) next_id_ = id;
        unsigned int tp = atoi(tokens[1].data());
        std::string cl = tokens[2];
        update_model(cl, tp, line);
      }
      file.close();
      parsed_ = true;
      ++next_id_;
    }
    else {
      std::cerr << "Unable to open data file." << std::endl;
      exit(1);
    }
  }

 protected:
  unsigned int next_id_;

 private:
  bool parsed_;
};

// data structure for SMOTE
class Smote : public Generator {
 public:
  Smote(const std::string &i, const std::string &o, unsigned int k) :
    knn_(i, o), k_(k), input_(i), output_(o) {
//     parse(i); // FIXME
  }

  Smote (const Smote &rhs) : knn_(rhs.knn_), Generator(rhs){
    
  }

  virtual ~Smote() {}

  virtual std::string generate_sample(const unsigned int doc_id,
                                      const std::string &c,
                                      const unsigned int p_source,
                                      const unsigned int p_target) {
    std::vector<std::string> syn_docs;
    // sort a sample from <c.p> and smote it !
    if (instances_[c].find(p_source) != instances_[c].end()) {
      unsigned int sz = instances_[c][p_source].size();
      unsigned int rnd_idx = static_cast<unsigned int>(rand() % sz);

      std::string d = instances_[c][p_source][rnd_idx];

      knn_.select_nn(d, k_, syn_docs);

      // now, we smote each document based on d:
      bool smoted = false;
      for (unsigned int i = 0; i < syn_docs.size(); i++) {
        std::string syn = smote(doc_id, p_target, d, syn_docs[i]);
        if (syn.compare(std::string("###")) != 0) {
          return syn;
        }
      }
    }
    return std::string("");
  }

  // FIXME: STUB
  std::string generate_sample(const std::string &c,
                              const unsigned int p_source,
                              const unsigned int p_target) {
    std::vector<std::string> syn_docs;
    // sort a sample from <c.p> and smote it !
    std::string res("");
    if (instances_[c].find(p_source) != instances_[c].end()) {
      unsigned int sz = instances_[c][p_source].size();
      unsigned int rnd_idx = static_cast<unsigned int>(rand() % sz);

      std::string d = instances_[c][p_source][rnd_idx];

      knn_.select_nn(d, k_, syn_docs);

      // now, we smote each document based on d:
      bool smoted = false;
      for (unsigned int i = 0; i < syn_docs.size(); i++) {
        std::string syn = smote(next_id_++, p_target, d, syn_docs[i]);
        if (syn.compare(std::string("###")) != 0) {
          res = syn;
          smoted = true; break;
        }
      }
      if (!smoted) {
        std::cerr << d << std::endl;
        res = d;
      }
    }
    else {
      std::cerr << "[SMOTE] PROBLEM: Zero examples in class " << c << " at tp=" << p_source << std::endl;
    }
    return res;
  }

  virtual void update_model(const std::string &c,
                            const unsigned int p,
                            const std::string &ln) {
    Generator::update_model(c, p, ln);
    knn_.update(ln);
  }

  virtual Generator *clone() {
    return new Smote(*this);
  }

 private:
  KNN_Selector knn_;
  unsigned int k_;
  std::string input_;
  std::string output_;

  std::string smote(const unsigned int n_id, const unsigned int p_target,
                    const std::string &a, const std::string &b) {
    std::string id = Utils::toString(n_id);
    std::string tp = Utils::toString(p_target);

    std::vector<std::string> tokens;
    Utils::string_tokenize(a, tokens, ";");
    std::string cl = tokens[2];
//    double gap = rand() / static_cast<double>(RAND_MAX);
    std::set<std::string> terms;
    std::map<std::string, double> tf_a;
    for (unsigned int i = 3; i < tokens.size()-1; i+=2) {
      std::string t = tokens[i];
      terms.insert(t);
      double tf = atof(tokens[i+1].data());
      tf_a[t] = tf;
    }

    tokens.clear();
    Utils::string_tokenize(b, tokens, ";");
    std::map<std::string, double> tf_b;
    for (unsigned int i = 3; i < tokens.size()-1; i+=2) {
      std::string t = tokens[i];
      terms.insert(t);
      double tf = atof(tokens[i+1].data());
      tf_b[t] = tf;
    }

    std::string nl = id + ";" + tp + ";" + cl;
    std::set<std::string>::const_iterator it = terms.begin();
    bool print = false;
    while (it != terms.end()) {
      std::map<std::string, double>::const_iterator ita = tf_a.find(*it);
      std::map<std::string, double>::const_iterator itb = tf_b.find(*it);
      double wa = (ita != tf_a.end() ? ita->second : 0.0);
      double wb = (itb != tf_b.end() ? itb->second : 0.0);
      double gap = rand() / static_cast<double>(RAND_MAX);
      unsigned int tf = ceil(wa + fabs(wa - wb)*gap);
      if (tf != 0) {
        nl = nl + ";" + *it + ";" + Utils::toString(tf);
        print = true;
      }
      ++it;
    }
    return (print ? nl : std::string("###"));
  }
};

// data structure for managing the simple data replicator
class Replicator : public Generator {
 public:
  Replicator() {}
  virtual ~Replicator() {}

  virtual std::string generate_sample(const unsigned int doc_id,
                                      const std::string &c,
                                      const unsigned int p_source,
                                      const unsigned int p_target) {
    if (instances_[c].find(p_source) != instances_[c].end()) {
      unsigned int sz = instances_[c][p_source].size();
      unsigned int rnd_idx = static_cast<unsigned int>(rand() % sz);

      std::string r = instances_[c][p_source][rnd_idx];

      // change creation point in time
      std::vector<std::string> tokens;
      Utils::string_tokenize(r, tokens, ";");

      std::string id = Utils::toString(doc_id);
      std::string tp = Utils::toString(p_target);
      std::string cl = tokens[2];

      std::string nl = id + ";" + tp + ";" + cl;
      for (unsigned int i = 3; i < tokens.size(); i++) {
        nl = nl + ";" + std::string(tokens[i]);
      }
      tokens.clear();
      return nl;
    }
    else {
      std::cerr << "THERE ISN'T ANY EXEMPLE FOR CLASS " << c << " AT P_SOURCE=" << p_source << std::endl;
      exit(1);
    }
  }

  virtual Generator *clone() {
    return new Replicator(*this);
  }
};

// data structure for managing the multinomial probabilistic model
class Multinomial : public Generator {
 public:
  virtual ~Multinomial() {}

  virtual void update_model(const std::string &c,
                            const unsigned int p,
                            const std::string &ln) { return; } // FIXME STUB
  
  // FIXME: STUB
  virtual std::string generate_sample(const unsigned int doc_id,
                                      const std::string &c,
                                      const unsigned int p_source,
                                      const unsigned int p_target) {
    return std::string();
  }

  virtual Generator *clone() {
    return new Multinomial(*this);
  }

//  void merge_model(const Multinomial *gn);

 private:

};

// data structure for managing the gaussian probabilistic model
class Gaussian : public Generator {
 public:
  virtual ~Gaussian() {}
  virtual void update_model(const std::string &c,
                            const unsigned int p,
                            const std::string &ln) { return; } // FIXME: STUB

  // FIXME: STUB
  virtual std::string generate_sample(const unsigned int doc_id,
                                      const std::string &c,
                                      const unsigned int p_source,
                                      const unsigned int p_target) {
    return std::string();
  }

  virtual Generator *clone() {
    return new Gaussian(*this);
  }

//  void merge_model(const Gaussian *gn);

 private:

};


#endif
