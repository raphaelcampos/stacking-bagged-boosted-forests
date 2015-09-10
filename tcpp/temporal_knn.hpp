#ifndef TEMPORAL_KNN_HPP__
#define TEMPORAL_KNN_HPP__

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

#include "temporal_classifier.hpp"
#include "knn.hpp"

class temporal_knn;
class TempExpNetDoc;

class TempKNN_Document : public KNN_Document {
 friend class TempExpNetDoc;
 friend class temporal_knn;
 public:
  TempKNN_Document(const unsigned int i, const std::string &c,
                   const std::string &y) : KNN_Document(i, c), doc_year(y) {}
  bool operator<(const TempKNN_Document &rhs) const {return id > rhs.id;}

 private:
  std::string doc_year;
};

class temp_similarity_t {
 public:
  temp_similarity_t(const TempKNN_Document &d, double s) :
    doc(d), similarity(s) {}
  bool operator<(const temp_similarity_t &rhs) const {
    return similarity < rhs.similarity;
  }
  TempKNN_Document doc;
  double similarity;
};

// link connectiong term to document. It's weighted in face to
// compute cosine similarity in a fast manner:
//   weight initialized to trainTermWeight / |doc|^2
// For cosine, the test words are also normalized and multiplied with
// those train weights previously computed (each term value are then summed).
class TempExpNetDoc {
 public:
  TempExpNetDoc(const TempKNN_Document &d, double w) : weight(w), doc(d) {}
  bool operator<(const TempExpNetDoc &rhs) const {return doc.id < rhs.doc.id;}
  double weight;
  TempKNN_Document doc;
};

typedef std::map<ExpNetHead, std::set<TempExpNetDoc> > temp_exp_net;

class temporal_knn : public knn, public TemporalClassifier {
 public:
  temporal_knn(const std::string &twf, unsigned int k=30, unsigned int r=0, double b=1.0,
      bool w = false, bool gw = false, unsigned int w_sz=0) :
        knn(k,r), TemporalClassifier(twf,r,b,w,gw,w_sz)
          { std::cerr << "temporal_knn (k=" << k << ")" << std::endl; }

  virtual bool parse_train_line(const std::string &l);
  virtual void parse_test_line(const std::string &l);
  virtual bool check_train_line(const std::string &l) const;

  virtual void reset_model() {
    TemporalClassifier::reset_model();
    knn::reset_model();
  }

  virtual double twf(unsigned int p_r, unsigned int p) {
    return TemporalClassifier::twf(p_r, p);
  }

  virtual ~temporal_knn() {}

 private:
  temp_exp_net index;

  virtual void updateIDF();
  virtual bool updateDocumentSize(const std::string &trn);
  virtual void getKNearest(unsigned int id, const std::string &ref_yr,
                           std::map<TempKNN_Document, double> &s, Scores<double> &o, double &n);
};

bool temporal_knn::check_train_line(const std::string &line) const {
  std::vector<std::string> tokens;
  Utils::string_tokenize(line, tokens, ";");
  // input format: doc_id;year;CLASS=class_name;{term_id;tf}+
  if ((tokens.size() < 5) || (tokens.size() % 2 == 0)) return false;
  return true;
}

bool temporal_knn::parse_train_line(const std::string &line) {
  std::vector<std::string> tokens;
  Utils::string_tokenize(line, tokens, ";");
  // input format: doc_id;year;CLASS=class_name;{term_id;tf}+
  if ((tokens.size() < 5) || (tokens.size() % 2 == 0)) return false;

  TempKNN_Document cur_doc(atoi(tokens[0].data()),
                           tokens[2], tokens[1]);

  double maxTF = 1.0;

/*
  double maxTF = 1.0 + log(atof(tokens[2].data()));
  for (unsigned int i = 4; i < tokens.size()-1; i+=2) {
    double tf = 1.0 + log(atof(tokens[i+1].data()));
    maxTF = (tf > maxTF) ? tf : maxTF;
  }
*/

  // for each term, insert a post for this doc in the posting lists
  for (size_t i = 3; i < tokens.size()-1; i+=2) {
    double tf = 1.0 + log(atof(tokens[i+1].data()));
    unsigned int term_id = atoi(tokens[i].data());
    ExpNetHead head;
    head.term_id = term_id;
    head.idf = 0.0; // we still need to update IDF, after everything !!!

    double w = tf / maxTF;
    TempExpNetDoc expNetDoc (cur_doc, w);

    temp_exp_net::iterator it = index.find(head);
    // term is not at index, so add it!
    if (it == index.end()) {
      std::set<TempExpNetDoc> postings;
      postings.insert(expNetDoc);
      index.insert(std::pair<ExpNetHead,
                             std::set<TempExpNetDoc> >(head, postings));
    }
    else { // term is already there ! just update (insert new post).
//      it->second.insert(expNetDoc);
      std::set<TempExpNetDoc> *postings = &it->second;
      postings->insert(expNetDoc);
    }
  }

  return true;
}

void temporal_knn::updateIDF() {
  temp_exp_net::iterator it = index.begin();
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

bool temporal_knn::updateDocumentSize(const std::string &train_fn) {
  std::ifstream file(train_fn.data());
  std::string line;

  if (file) {
    while (file >> line) {
      std::vector<std::string> tokens;
      Utils::string_tokenize(line, tokens, ";");
      if ((tokens.size() < 5) || (tokens.size() % 2 == 0)) return false;

      int id = atoi(tokens[0].data());
      std::string year = tokens[1];

      double size = 0.0;

      double maxTF = 1.0;

/*
      double maxTF = 1.0 + log(atof(tokens[4].data()));
      for (unsigned int i = 5; i < tokens.size()-1; i+=2) {
        double tf = 1.0 + log(atof(tokens[i+1].data()));
        maxTF = (tf > maxTF) ? tf : maxTF;
      }
*/

      for (size_t i = 3; i < tokens.size()-1; i+=2) {
        double tf = 1.0 + log(atof(tokens[i+1].data()));
        tf /= maxTF;
        ExpNetHead head;
        head.term_id = atoi(tokens[i].data());
        head.idf = 0.0;

        temp_exp_net::iterator it = index.find(head);
        if (it != index.end()) {
          if ((it->first).idf == 0) {
            std::cerr << "[TREINO] > TERMO " << (it->first).term_id
                      << " COM IDF ZERO !!!!" << std::endl;
          }
          else {
            tf *= ((it->first).idf / maxIDF);
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

void temporal_knn::getKNearest(unsigned int test_id,
                              const std::string &ref_year,
                              std::map<TempKNN_Document, double> &similarities,
                              Scores<double> &ordered_sim_class, double &norm) {
  std::priority_queue<temp_similarity_t, std::vector<temp_similarity_t> > sim;
  std::map<TempKNN_Document, double>::iterator it = similarities.begin();
  while (it != similarities.end()) {
    double w = it->second * twf(atoi(ref_year.data()),
                       atoi(it->first.doc_year.data()));
    temp_similarity_t simil(it->first, w);
    sim.push(simil);
    ++it;
  }
  std::map<std::string, double> sim_classes;
  unsigned int cur = 0;
  norm = -9999.99;
  while(cur < k && !sim.empty()) {
    temp_similarity_t simil = sim.top();
    sim_classes[simil.doc.doc_class] += (simil.similarity);
    if (norm < sim_classes[simil.doc.doc_class]) norm = sim_classes[simil.doc.doc_class];
    sim.pop();
    cur++;
  }
  std::map<std::string, double>::iterator sIt = sim_classes.begin();
  while (sIt != sim_classes.end()) {
    ordered_sim_class.add(sIt->first, sIt->second);
    ++sIt;
  }
}

void temporal_knn::parse_test_line(const std::string &line) {
  std::vector<std::string> tokens;
  Utils::string_tokenize(line, tokens, ";");
  if ((tokens.size() < 5) || (tokens.size() % 2 == 0)) return;

  double maxTF = 1.0;

  maxTF = atoi(tokens[4].data());
  for (unsigned int i = 5; i < tokens.size()-1; i+=2) {
    double tf = (atof(tokens[i+1].data()));
    maxTF = (tf > maxTF) ? tf : maxTF;
  }

  double test_size = 0.0;

  for (int i = 3; i < static_cast<int>(tokens.size())-1; i+=2) {
    unsigned int term_id = atoi(tokens[i].data());
    double tf = 1.0 + log(atof(tokens[i+1].data()));
    tf /= maxTF;
    ExpNetHead head;
    head.term_id = term_id;
    head.idf = 0.0;

    temp_exp_net::iterator it = index.find(head);
    if (it != index.end()) {
      if ((it->first).idf == 0)
        std::cerr << "[TESTE] > TERMO " << (it->first).term_id
                  << " COM IDF ZERO !!!!" << std::endl;
      else tf *= ((it->first).idf / maxIDF);
    }
    test_size += tf * tf;
  }
  test_size = sqrt(test_size);

  std::map<TempKNN_Document, double> similarities;
  for (int i = 3; i < static_cast<int>(tokens.size())-1; i+=2) {
    double tf = 1 + log(atof(tokens[i+1].data()));
    int term_id = atoi(tokens[i].data());
    ExpNetHead head;
    head.term_id = term_id;
    head.idf = 0.0;
    double test_weight = tf / maxTF;
    temp_exp_net::iterator it;
    it = index.find(head);
    if (it != index.end()) {
      test_weight *= ((it->first).idf / maxIDF);
      for (std::set<TempExpNetDoc>::iterator itt = (it->second).begin();
           itt != (it->second).end(); ++itt) {
        double dsz = 0.0;
        std::map<int, double>::const_iterator dsz_it = doc_sizes.find(itt->doc.id);
        if (dsz_it != doc_sizes.end()) dsz = dsz_it->second;

        double train_size = sqrt(dsz);
        double train_weight = itt->weight * ((it->first).idf / maxIDF);
        similarities[itt->doc] += (((train_weight) / train_size) *
                                   ((test_weight) / test_size));
      }
    }
  }

  TempKNN_Document cur_doc(atoi(tokens[0].data()),
                           tokens[2], tokens[1]);

  Scores<double> scores(tokens[0], cur_doc.doc_class);
  double norm = 1.0;
  getKNearest(cur_doc.id, cur_doc.doc_year, similarities, scores, norm);

  get_outputer()->output(scores);
}
#endif
