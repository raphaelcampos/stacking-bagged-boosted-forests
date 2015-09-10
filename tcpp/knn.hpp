#ifndef KNN_HPP__
#define KNN_HPP__

#include <fstream>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <vector>
#include <set>
#include <queue>
#include <utility>
#include <fstream>
#include <cstdlib>

#include "supervised_classifier.hpp"
#include "scores.hpp"

class ExpNetDoc;
class knn;

class KNN_Document {
 friend class ExpNetDoc;
 friend class knn;
 public:
  KNN_Document(const unsigned int i, const std::string &c)
    : id(i), doc_class(c), size(0.0) {}
  bool operator<(const KNN_Document &rhs) const {return id > rhs.id;}
  unsigned int get_id() { return id; }
  std::string get_class() { return doc_class; }
  double get_size() { return size; }
  void set_size(const double s) { size = s;}

 protected:
  unsigned int id;
  std::string doc_class;
  double size;
};

// link connectiong term to document. It's weighted in face to
// compute cosine similarity in a fast manner:
//   weight initialized to trainTermWeight / |doc|^2
// For cosine, the test words are also normalized and multiplied with
// those train weights previously computed (each term value are then summed).
class ExpNetDoc {
 public:
  ExpNetDoc(const KNN_Document &d, double w) : weight(w), doc(d) {}
  bool operator<(const ExpNetDoc &rhs) const {return doc.id < rhs.doc.id;}
  double weight;
  KNN_Document doc;
};

class ExpNetHead {
 public:
  bool operator<(const ExpNetHead &rhs) const {return term_id < rhs.term_id;}
  unsigned int term_id;
  double idf;
};

class similarity_t {
 public:
  similarity_t(const KNN_Document &d, double s) : doc(d), similarity(s) {}
  bool operator<(const similarity_t &rhs) const {
    return similarity < rhs.similarity;
  }
  KNN_Document doc;
  double similarity;
};

// Node from the inverted index (term -> docs)
typedef std::map<ExpNetHead, std::set<ExpNetDoc> > exp_net;

class knn : public virtual SupervisedClassifier {
 public:
  knn(unsigned int k=30, unsigned int round=0) :
    SupervisedClassifier(round),
    k(k), maxIDF(-9999.99) {
      if (k <= 0) {
        std::cerr << "[KNN] Invalid K=0. Using default: K=30" << std::endl;
        k=30;
      }
    }

  virtual bool parse_train_line(const std::string &l);
  virtual void parse_test_line(const std::string &l);
  virtual bool check_train_line(const std::string &l) const;

  virtual void reset_model() {
    maxIDF = -9999.99;
    index.clear();
    doc_sizes.clear();
  }

  virtual void train(const std::string &trn); // overloaded (to perform idf and size updates)

  virtual ~knn() {}

 protected:
  virtual void updateIDF();
  virtual bool updateDocumentSize(const std::string &train_fn);
  virtual void getKNearest(unsigned int id, std::map<KNN_Document, double> &s,
                           Scores<double> &o, double &n);

  unsigned int k;
  exp_net index;
  double maxIDF;
  std::map<int, double> doc_sizes;
};

bool knn::check_train_line(const std::string &line) const {
  std::vector<std::string> tokens;  tokens.reserve(100);
  Utils::string_tokenize(line, tokens, ";");
  // input format: doc_id;CLASS=class_name;{term_id;tf}+
  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return false;
  return true;
}

bool knn::parse_train_line(const std::string &line) {
  std::vector<std::string> tokens;  tokens.reserve(100);
  Utils::string_tokenize(line, tokens, ";");
  // input format: doc_id;CLASS=class_name;{term_id;tf}+
  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return false;

  KNN_Document cur_doc(atoi(tokens[0].data()), tokens[1]);

  double maxTF = 1.0;

/*
  double maxTF = 1.0 + log(atof(tokens[2].data()));
  for (unsigned int i = 4; i < tokens.size()-1; i+=2) {
    double tf = (raw) ? atof(tokens[i+1].data()) : 1.0 + log(atof(tokens[i+1].data()));
    maxTF = (tf > maxTF) ? tf : maxTF;
  }
*/
  // for each term, insert a post for this doc in the posting lists
  for (size_t i = 2; i < tokens.size()-1; i+=2) {
    double tf = (raw) ? atof(tokens[i+1].data()) : 1.0 + log(atof(tokens[i+1].data()));
    unsigned int term_id = atoi(tokens[i].data());
    ExpNetHead head;
    head.term_id = term_id;
    head.idf = (raw) ? 1.0 : 0.0; // we still need to update IDF, after everything !!!

    double w = tf / maxTF;
    ExpNetDoc expNetDoc (cur_doc, w);

    exp_net::iterator it = index.find(head);
    // term is not at index, so add it!
    if (it == index.end()) {
      std::set<ExpNetDoc> postings;
      postings.insert(expNetDoc);
      index.insert(std::pair<ExpNetHead,
                             std::set<ExpNetDoc> >(head, postings));
    }
    else { // term is already there ! just update (insert new post).
//      it->second.insert(expNetDoc);
      std::set<ExpNetDoc> *postings = &it->second;
      postings->insert(expNetDoc);
    }
  }

  return true;
}

void knn::updateIDF() {
  if (raw) return;

  exp_net::iterator it = index.begin();
  while (it != index.end()) {
    // Pre-condition: we already have created our inverted index !
    double idf = log10((static_cast<double>(get_total_docs()) + 1.0) /
                       (static_cast<double>((it->second).size() + 1.0)));
    if (maxIDF < idf) maxIDF = idf;
    (const_cast<ExpNetHead *> (&(it->first)))->idf = idf;
    ++it;
  }
  //maxIDF = 1.0;
}

bool knn::updateDocumentSize(const std::string &train_fn) {
  std::ifstream file(train_fn.data(), std::ios::in);
  std::string line;

  if (file) {
    file.seekg(0, std::ios::beg);
    while (std::getline(file, line)) {
      std::vector<std::string> tokens; tokens.reserve(100);
      Utils::string_tokenize(line, tokens, ";");

      int id = atoi(tokens[0].data());

      double size = 0.0;

      if (tokens.size() < 4) {
        doc_sizes[id] = size;
        continue;
      }

      double maxTF = 1.0;
/*
      double maxTF = 1.0 + log(atof(tokens[2].data()));
      for (unsigned int i = 4; i < tokens.size()-1; i+=2) {
        double tf = 1.0 + log(atof(tokens[i+1].data()));
        maxTF = (tf > maxTF) ? tf : maxTF;
      }
*/

      for (size_t i = 2; i < tokens.size()-1; i+=2) {
        double tf = (raw) ? atof(tokens[i+1].data()) : 1.0 + log(atof(tokens[i+1].data()));
        tf /= maxTF;
        ExpNetHead head;
        head.term_id = atoi(tokens[i].data());
        head.idf = (raw) ? 1.0 : 0.0;

        if (!raw) {
          exp_net::iterator it = index.find(head);
          if (it != index.end()) {
            if ((it->first).idf == 0) {
              std::cerr << "[TREINO] > TERMO " << (it->first).term_id
                        << " COM IDF ZERO !!!!" << std::endl;
            }
            else {
              tf *= ((it->first).idf / maxIDF);
            }
          }
        }
        size += tf * tf;
      }
      doc_sizes[id] = size;
    }
    file.close();
  }
  return true;
}

void knn::train(const std::string &trn) {
  SupervisedClassifier::train(trn);
  updateIDF();
  updateDocumentSize(trn);
}

void knn::getKNearest(unsigned int test_id,
                      std::map<KNN_Document, double> &similarities,
                      Scores<double> &ordered_sim_class, double &norm) {
  std::priority_queue<similarity_t, std::vector<similarity_t> > sim;
  std::map<KNN_Document, double>::iterator it = similarities.begin();
  while (it != similarities.end()) {
    double s = it->second;
    switch(dist_type) {
      case L2:
        s = 1.0 - sqrt(s);
        break;
      case L1:
        s = 1.0 - s;
        break;
    }

    similarity_t simil(it->first, s);
    sim.push(simil);
    ++it;
  }

  norm = -9999.99;
  std::map<std::string, double> sim_classes;
  unsigned int cur = 0;
  while(cur < k && !sim.empty()) {
    similarity_t simil = sim.top();
    sim_classes[simil.doc.doc_class] += (simil.similarity);
    if (sim_classes[simil.doc.doc_class] > norm) norm = sim_classes[simil.doc.doc_class];
    sim.pop();
    cur++;
  }

  std::map<std::string, double>::iterator sIt = sim_classes.begin();
  while (sIt != sim_classes.end()) {
    ordered_sim_class.add(sIt->first, sIt->second);
    ++sIt;
  }
}

void knn::parse_test_line(const std::string &line) {
  std::vector<std::string> tokens;  tokens.reserve(100);
  Utils::string_tokenize(line, tokens, ";");
  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return;

  double maxTF = 1.0;
/*
  maxTF = atoi(tokens[3].data());
  for (unsigned int i = 4; i < tokens.size()-1; i+=2) {
    double tf = (raw) ? atof(tokens[i+1].data()) : 1.0 + log(atof(tokens[i+1].data()));
    maxTF = (tf > maxTF) ? tf : maxTF;
  }
*/

  double test_size = 0.0;

  for (int i = 2; i < static_cast<int>(tokens.size())-1; i+=2) {
    unsigned int term_id = atoi(tokens[i].data());
    double tf = (raw) ? atof(tokens[i+1].data()) : 1.0 + log(atof(tokens[i+1].data()));
    tf /= maxTF;
    ExpNetHead head;
    head.term_id = term_id;
    head.idf = (raw) ? 1.0 : 0.0;

    if (!raw) {
      exp_net::iterator it = index.find(head);
      if (it != index.end()) {
        if ((it->first).idf == 0)
          std::cerr << "[TESTE] > TERMO " << (it->first).term_id
                    << " COM IDF ZERO !!!!" << std::endl;
        else tf *= ((it->first).idf / maxIDF);
      }
    }
    test_size += tf * tf;
  }
  test_size = sqrt(test_size);

  std::map<KNN_Document, double> similarities;
  for (int i = 2; i < static_cast<int>(tokens.size())-1; i+=2) {
    double tf = (raw) ? atof(tokens[i+1].data()) : 1.0 + log(atof(tokens[i+1].data()));
    int term_id = atoi(tokens[i].data());
    ExpNetHead head;
    head.term_id = term_id;
    head.idf = (raw) ? 1.0 : 0.0;
    double test_weight = tf / maxTF;
    exp_net::iterator it = index.find(head);
    if (it != index.end()) {
      double idfw = (raw) ? 1.0 :  ((it->first).idf / maxIDF);
      if (!raw) test_weight *= idfw;
      for (std::set<ExpNetDoc>::iterator itt = (it->second).begin();
           itt != (it->second).end(); ++itt) {

        double dsz = 0.0;
        std::map<int, double>::const_iterator dsz_it = doc_sizes.find(itt->doc.id);
        if (dsz_it != doc_sizes.end()) dsz = dsz_it->second;

        double train_size = sqrt(dsz);
        double train_weight = itt->weight * idfw;

        train_size = test_size = 1.0;

        switch(dist_type) {
          case L2:
            similarities[itt->doc] += pow((((train_weight) / train_size) -
                                           ((test_weight) / test_size)), 2.0);
            break;

          case L1:
            similarities[itt->doc] += abs(((train_weight) / train_size) -
                                          ((test_weight) / test_size));
            break;
          case COSINE:
          default:
            similarities[itt->doc] += (((train_weight) / train_size) *
                                       ((test_weight) / test_size));
            break;
        }
      }
    }
  }

  std::string real_class = tokens[1];
  KNN_Document cur_doc(atoi(tokens[0].data()), real_class);

  double norm=1.0;
  Scores<double> scores(tokens[0], real_class);
  getKNearest(cur_doc.id, similarities, scores, norm);

  get_outputer()->output(scores, norm);
}
#endif
