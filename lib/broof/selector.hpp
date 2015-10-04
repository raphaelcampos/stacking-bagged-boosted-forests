#ifndef SELECTOR_HPP__
#define SELECTOR_HPP__

#include <map>
#include <set>

#include <iostream>
#include <fstream>
#include <queue>

#include <cmath>
#include <cstdlib>

#include "utils.hpp"

class Selector {
 public:
  Selector(const std::string &i_fn, const std::string &o_fn)
    : parsed_(false), input_(i_fn), output_(o_fn), total_docs_(0) { }

  Selector() : parsed_(false), total_docs_(0) { }

  Selector(const Selector &rhs) :
    parsed_(rhs.parsed_), input_(rhs.input_), output_(rhs.output_),
    total_docs_(rhs.total_docs_), cl_dist_(rhs.cl_dist_) {}

  void select(unsigned int b, bool rev=false) {
    if (!parsed_) parse_data();
    select_examples(b, rev);
  }

  void select(const std::string &ln,
              const std::vector<std::string> &data,
              unsigned int num_docs,
              std::vector<std::string> &out) {
    if (!parsed_) parse_data(data);
    select_examples(ln, num_docs, out);
  }

  virtual void update(const std::string &ln) = 0;

  void output(const std::vector<std::string> &docs) {
    // TODO: IMPLEMENTAR ESCRITA EM OUTPUT_
    for (unsigned int i = 0; i < docs.size(); i++) {
      std::cout << docs[i] << std::endl;
    }    
  }

  void output(const std::vector<std::string> &docs,
              std::vector<std::string> &out) {
    for (unsigned int i = 0; i < docs.size(); i++) {
      out.push_back(docs[i]);
    }    
  }

 protected:
  bool parsed_;
  std::string input_;
  std::string output_;
  unsigned int total_docs_;
  std::map<std::string, unsigned int> cl_dist_;

  virtual void parse_data(const std::vector<std::string> &data) {
    for (unsigned int i = 0; i < data.size(); i++) {
      update(data[i]);
      total_docs_++;
    }
  }

  virtual void parse_data() {
    if (parsed_) return;
    std::ifstream file(input_.data());
    if (file) {
      std::string line;
      unsigned int total = 0;
      while (std::getline(file, line)) {
        update(line); 
        total++;
      }
      file.close();
      parsed_ = true;
      std::cerr << "[SELECTOR] Parsed " << total << " documents." << std::endl;

//      std::map<std::string, unsigned int>::iterator it = cl_dist_.begin();
//      while (it != cl_dist_.end()) {
//        std::cerr << "CL=" << it->first << " -> " << it->second << std::endl;
//        ++it;
//      }
    }
    else {
      std::cerr << "Unable to open data file." << std::endl;
      exit(1);
    }
  }

  virtual void select_examples(unsigned int b, bool rev=false) = 0;
  virtual void select_examples(const std::string &ln,
                               unsigned int n, 
                               std::vector<std::string> &out) = 0;
};

class KNN_Selector : public Selector {
 public:
  KNN_Selector(const std::string &i, const std::string &o, unsigned int min = 0)
    : Selector(i, o), max_idf_(-9999.99), min_docs_(min) {}

  KNN_Selector(const KNN_Selector &rhs) : Selector(rhs) {
    index_ = rhs.index_;
    centroids_ = rhs.centroids_;
    dataset_ = rhs.dataset_;
    doc_sizes_ = rhs.doc_sizes_;
    max_idf_ = rhs.max_idf_;
    min_docs_ = rhs.min_docs_;
  }

  void select_nn(const std::string &ln, unsigned int k, std::vector<std::string> &out) {
    if (!parsed_) parse_data();
    select_examples(ln, k, out);
  }

  virtual void update(const std::string &ln) {
    std::vector<std::string> tokens;
    Utils::string_tokenize(ln, tokens, ";");
    unsigned int id = static_cast<unsigned int>(atoi(tokens[0].data()));
    std::string yr = tokens[1];
    std::string cl = tokens[2];
    Document cur_doc(id, Utils::get_index(cl, yr));
    cl_dist_[Utils::get_index(cl, yr)]++;

    dataset_[id] = ln;

    for (size_t i = 3; i < tokens.size()-1; i+=2) {
      double tf = 1.0 + log(atof(tokens[i+1].data()));
      std::string term_id = tokens[i];
      ExpNetHead head;
      head.term_id = static_cast<unsigned int>(atoi(term_id.data()));
      head.idf = 0.0; // we still need to update IDF, after everything !!!

      double w = tf;
      ExpNetDoc expNetDoc (cur_doc, w);

      exp_net::iterator it = index_.find(head);
      // term is not at index, so add it!
      if (it == index_.end()) {
        std::set<ExpNetDoc> postings;
        postings.insert(expNetDoc);
        index_.insert(std::pair<ExpNetHead,
                                std::set<ExpNetDoc> >(head, postings));
      }
      else { // term is already there ! just update (insert new post).
//      it->second.insert(expNetDoc);
        std::set<ExpNetDoc> *postings = &it->second;
        postings->insert(expNetDoc);
      }
      // updating centroids
      centroids_[Utils::get_index(cl, yr)].terms[term_id] += w;
    }
    total_docs_++;
  }


 private:
  // SOME CLASS/TYPE DEFINITIONS
  class Centroid {
   public:
    Centroid() : size(0.0) {}
    double size;
    std::map<std::string, double> terms;
  };

  class ExpNetDoc;
  class Document {
   public:
    friend class ExpNetDoc;
    friend class KNN_Selector;
    Document(const unsigned int i, const std::string &c)
      : id(i), doc_class(c), size(0.0) {}
    bool operator<(const Document &rhs) const {return id > rhs.id;}
    unsigned int get_id() { return id; }
    std::string get_class() { return doc_class; }
    double get_size() { return size; }
    void set_size(const double s) { size = s;}
   private:
    unsigned int id;
    std::string doc_class;
    double size;
  };

  class ExpNetDoc {
   public:
    ExpNetDoc(const Document &d, double w) : weight(w), doc(d) {}
    bool operator<(const ExpNetDoc &rhs) const {return doc.id < rhs.doc.id;}
    double weight;
    Document doc;
  };

  class ExpNetHead {
   public:
    bool operator<(const ExpNetHead &rhs) const {return term_id < rhs.term_id;}
    unsigned int term_id;
    double idf;
  };

  typedef std::map<ExpNetHead, std::set<ExpNetDoc> > exp_net;

  class similarity_t {
   public:
    similarity_t(const Document &d, double s) : doc(d), similarity(s) {}
    bool operator<(const similarity_t &rhs) const {
      return similarity < rhs.similarity;
    }
    Document doc;
    double similarity;
  };

  // DATA MEMBERS
  exp_net index_;
  std::map<std::string, Centroid> centroids_;
  std::map<unsigned int, std::string> dataset_;
  std::map<unsigned int, double> doc_sizes_;
  double max_idf_;
  unsigned int min_docs_;

  // MEMBER FUNCTIONS
  void updateIDF() {
    exp_net::iterator it = index_.begin();
    while (it != index_.end()) {
      // Pre-condition: we already have created our inverted index !
      double idf = log10((static_cast<double>(total_docs_) + 1.0) /
                         (static_cast<double>((it->second).size() + 1.0)));
      if (max_idf_ < idf) max_idf_ = idf;
      (const_cast<ExpNetHead *> (&(it->first)))->idf = idf;
      ++it;
    }
  }

  virtual void parse_data() {
    Selector::parse_data();
    updateIDF();
    update_document_sizes();
  }

  void update_document_sizes() {
    std::ifstream file(input_.data());
    std::string line;
    if (file) {
      while (std::getline(file, line)) {
        std::vector<std::string> tokens;
        Utils::string_tokenize(line, tokens, ";");
        unsigned int id = static_cast<unsigned int>(atoi(tokens[0].data()));

        double size = 0.0;

        if (tokens.size() < 5) {
          doc_sizes_[id] = size; return;
        }

        for (size_t i = 3; i < tokens.size()-1; i+=2) {
          double tf = 1.0 + log(atof(tokens[i+1].data()));
          ExpNetHead head;
          head.term_id = static_cast<unsigned int>(atoi(tokens[i].data()));
          head.idf = 0.0;
          exp_net::iterator it = index_.find(head);
          if (it != index_.end()) {
            if ((it->first).idf != 0.0) {
              tf *= ((it->first).idf / max_idf_);
            }
          }
          size += tf * tf;
        }
        doc_sizes_[id] = size;
      }
      file.close();
    }
    else {
      std::cerr << "Unable to open input file for size computation." << std::endl;
      exit(1);
    }
  }

  virtual void select_examples(const std::string &ln, unsigned int k, std::vector<std::string> &out) {
    std::vector<std::string> tokens;
    Utils::string_tokenize(ln, tokens, ";");
    unsigned int id = static_cast<unsigned int>(atoi(tokens[0].data()));
    std::string yr = tokens[1];
    std::string cl = tokens[2];
    Centroid doc;

    for (size_t i = 3; i < tokens.size()-1; i+=2) {
      double tf = 1.0 + log(atof(tokens[i+1].data()));
      std::string term_id = tokens[i];
      doc.terms[term_id] += tf;
    }

    double cent_size = 0.0;
    std::map<std::string, double>::const_iterator t_it = doc.terms.begin();
    while (t_it != doc.terms.end()) {
      std::string term_id = t_it->first;
      double tf = 1.0 + log(t_it->second);
      ExpNetHead head;
      head.term_id = static_cast<unsigned int>(atoi(term_id.data()));
      head.idf = 0.0;
      exp_net::iterator exp_it = index_.find(head);
      if (exp_it != index_.end()) {
        if ((exp_it->first).idf != 0.0) {
          tf *= ((exp_it->first).idf / max_idf_);
        }
      }
      cent_size += tf * tf;
      ++t_it;
    }
    cent_size = sqrt(cent_size);

    std::map<Document, double> similarities;
    t_it = doc.terms.begin();
    while (t_it != doc.terms.end()) {
      std::string term_id = t_it->first;
      double cent_weight = 1.0 + log(t_it->second);
      ExpNetHead head;
      head.term_id = static_cast<unsigned int>(atoi(term_id.data()));
      head.idf = 0.0;
      exp_net::iterator exp_it = index_.find(head);
      if (exp_it != index_.end()) {
        cent_weight *= ((exp_it->first).idf / max_idf_);
        for (std::set<ExpNetDoc>::iterator itt = (exp_it->second).begin();
             itt != (exp_it->second).end(); ++itt) {
          double train_size = sqrt(doc_sizes_[itt->doc.id]);
          double train_weight = itt->weight * ((exp_it->first).idf / max_idf_);
          similarities[itt->doc] += (((train_weight) / train_size) *
                                      ((cent_weight) / cent_size));
        }
      }
      ++t_it;
    }
    get_k_neighbors(id, similarities, k, out);
  }

  virtual void select_examples(unsigned int b, bool rev=false) {
    std::map<std::string, Centroid>::const_iterator it = centroids_.begin();
    while (it != centroids_.end()) {
      double cent_size = 0.0;
      std::map<std::string, double>::const_iterator t_it = it->second.terms.begin();
      while (t_it != it->second.terms.end()) {
        std::string term_id = t_it->first;
        double tf = 1.0 + log(t_it->second);
        ExpNetHead head;
        head.term_id = static_cast<unsigned int>(atoi(term_id.data()));
        head.idf = 0.0;

        exp_net::iterator exp_it = index_.find(head);
        if (exp_it != index_.end()) {
          if ((exp_it->first).idf != 0.0) {
            tf *= ((exp_it->first).idf / max_idf_);
          }
        }
        cent_size += tf * tf;
        ++t_it;
      }
      cent_size = sqrt(cent_size);

      std::map<Document, double> similarities;
      t_it = it->second.terms.begin();
      while (t_it != it->second.terms.end()) {
        std::string term_id = t_it->first;
        double cent_weight = 1.0 + log(t_it->second);
        ExpNetHead head;
        head.term_id = static_cast<unsigned int>(atoi(term_id.data()));
        head.idf = 0.0;
        exp_net::iterator exp_it = index_.find(head);
        if (exp_it != index_.end()) {
          cent_weight *= ((exp_it->first).idf / max_idf_);
          for (std::set<ExpNetDoc>::iterator itt = (exp_it->second).begin();
               itt != (exp_it->second).end(); ++itt) {
            double train_size = sqrt(doc_sizes_[itt->doc.id]);
            double train_weight = itt->weight * ((exp_it->first).idf / max_idf_);
            similarities[itt->doc] += (((train_weight) / train_size) *
                                        ((cent_weight) / cent_size));
          }
        }
        ++t_it;
      }

      std::vector<std::string> knn;
      unsigned int num = cl_dist_[it->first];
      num = (num <= b ? num : b);
      get_k_neighbors(it->first, similarities, num, knn, rev);
      output(knn);
      ++it;
    }
  }

  void get_k_neighbors(const std::string &cl,
                       const std::map<Document, double> &sim,
                       const unsigned int k,
                       std::vector<std::string> &out, bool rev=false) {

    std::priority_queue<similarity_t, std::vector<similarity_t> > ord_sim;
    std::map<Document, double>::const_iterator it = sim.begin();
    while (it != sim.end()) {
      if (it->first.doc_class.compare(cl) == 0) {        
        similarity_t simil(it->first, it->second);
        ord_sim.push(simil);
      }
      ++it;
    }

    if (rev) {
      // get k less similar:
      if (ord_sim.size() < k) {
        std::cerr << "[KNN] UPS ! There are less (" << ord_sim.size() << ",CL=" << cl << ") examples than k=" << k << std::endl;
        exit(1);
      }
      unsigned int skip = ord_sim.size() - k;
      for (unsigned int i = 0; i < skip; i++) ord_sim.pop();
    }

    unsigned int cur = 0;
    while((cur < k) && !ord_sim.empty()) {
      similarity_t simil = ord_sim.top();
      out.push_back(dataset_[simil.doc.id]);
      ord_sim.pop();
      cur++;
    }
  }

  void get_k_neighbors(const unsigned int id,
                       const std::map<Document, double> &sim,
                       const unsigned int k,
                       std::vector<std::string> &out) {
    std::priority_queue<similarity_t, std::vector<similarity_t> > ord_sim;
    std::map<Document, double>::const_iterator it = sim.begin();
    while (it != sim.end()) {
      if (it->first.id != id) {
        similarity_t simil(it->first, it->second);
        ord_sim.push(simil);
      }
      ++it;
    }
    unsigned int cur = 0;
    while(cur < k && !ord_sim.empty()) {
      similarity_t simil = ord_sim.top();
      out.push_back(dataset_[simil.doc.id]);
      ord_sim.pop();
      cur++;
    }
  }

};

#endif
