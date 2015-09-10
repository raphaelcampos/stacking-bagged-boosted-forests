#ifndef ROCCHIO_HPP_H__
#define ROCCHIO_HPP_H__

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
#include <cstdio>

#include "supervised_classifier.hpp"
#include "utils.hpp"

class Rocchio;

class Rocchio_Document {
 public:
  Rocchio_Document() : size(0.0) {}
  double size;
  std::map<int, double> terms;
};

class Rocchio : public virtual SupervisedClassifier {
 public:
  Rocchio(unsigned int round=0) :
    SupervisedClassifier(round),
    maxIDF(-9999.99) {}

  virtual bool parse_train_line(const std::string &l);
  virtual void parse_test_line(const std::string &l);
  virtual bool check_train_line(const std::string &l) const;

  virtual void reset_model() {
    centroids.clear();
    sumDFperClass.clear();
    DFperTerm.clear();
    maxIDF = -9999.99;
  }

  virtual void train(const std::string &new_train);

  virtual ~Rocchio() {}

 protected:
  void updateMaxIDF();
  void updateCentroidsSize();

  std::map<std::string, Rocchio_Document> centroids;
  std::map<std::string, unsigned int> sumDFperClass;
  std::map<int, unsigned int> DFperTerm;

  double maxIDF;
};

bool Rocchio::check_train_line(const std::string &line) const {
  std::vector<std::string> tokens; tokens.reserve(100);
  Utils::string_tokenize(line, tokens, ";");
  // input format: doc_id;CLASS=class_name;{term_id;tf}+
  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return false;
  return true;
}

bool Rocchio::parse_train_line(const std::string &line) {
  std::vector<std::string> tokens; tokens.reserve(100);
  Utils::string_tokenize(line, tokens, ";");
  // input format: doc_id;CLASS=class_name;{term_id;tf}+
  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return false;

  std::string id = tokens[0];

  // remove CLASS= mark
  std::string doc_class = tokens[1];

  sumDFperClass[doc_class]++;

  for (unsigned int i = 2; i < tokens.size()-1; i+=2) {
    double weight = (raw) ? atof(tokens[i+1].data()) :
                    1.0 + log10(atof(tokens[i+1].data()));
    int term_id = atoi(tokens[i].data());
    DFperTerm[term_id]++;
    vocabulary_add(term_id);
    centroids[doc_class].terms[term_id] += weight;
  }

  return true;
}

void Rocchio::updateMaxIDF() {
  std::set<int>::const_iterator vIt = vocabulary_begin();
  while (vIt != vocabulary_end()) {
    double idf = 
      log10(static_cast<double>(get_total_docs() + 1.0)) /
            static_cast<double>(Utils::get_value(DFperTerm,*vIt) + 1.0);
    if (maxIDF < idf) maxIDF = idf;
    ++vIt;
  }
}

void Rocchio::updateCentroidsSize() {
  std::map<std::string, Rocchio_Document>::iterator cIt = centroids.begin();
  while(cIt != centroids.end()) {
    double centSize = 0.0;
    std::map<int, double>::iterator vIt = (cIt->second).terms.begin();
    while(vIt != (cIt->second).terms.end()) {
      double idf = (raw) ? 1.0 :
       log10(static_cast<double>(get_total_docs() + 1.0) /
             static_cast<double>(Utils::get_value(DFperTerm,vIt->first)+1.0))
               / maxIDF;
      double weight =
        vIt->second*idf/static_cast<double>(Utils::get_value(sumDFperClass, cIt->first));
      vIt->second = weight;
      centSize += pow(weight, 2);
      ++vIt;
    }
    cIt->second.size = sqrt(centSize);
    ++cIt;
  }
}

void Rocchio::train(const std::string &trn) {
  SupervisedClassifier::train(trn);
  if (!raw) updateMaxIDF();
  updateCentroidsSize();
}

void Rocchio::parse_test_line(const std::string &line) {
  std::vector<std::string> tokens; tokens.reserve(100);
  Utils::string_tokenize(line, tokens, ";");
  if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return;

  Scores<double> similarities(tokens[0], tokens[1]);
  double normalizer = 0.0;

  std::map<std::string, Rocchio_Document>::const_iterator cIt = centroids.begin();
  while (cIt != centroids.end()) {
    std::string cur_class = cIt->first;
    double sim = 0.0;

    double cent_size = cIt->second.size;
    if (cent_size > 0.0) {
      double doc_size = 0.0;

      for (size_t i = 2; i < tokens.size()-1; i+=2) {
        int term_id = atoi(tokens[i].data());
        double tf = (raw) ? atof(tokens[i+1].data()) :
          1.0 + log10(atof(tokens[i+1].data()));
        double idf = (raw) ? 1.0 :
          log10(static_cast<double>(get_total_docs() + 1.0) /
                static_cast<double>(Utils::get_value(DFperTerm, term_id)+1.0)) 
                  / maxIDF;

        double train_tfidf = tf * idf;
        double cent_tfidf = Utils::get_value(cIt->second.terms, term_id);
        sim += (train_tfidf * cent_tfidf);
        doc_size += pow(train_tfidf, 2);
      }
      sim /= (cent_size * sqrt(doc_size)); // cosine similarity
    }
    similarities.add(cur_class, sim);
    normalizer += sim;
    ++cIt;
  }

  get_outputer()->output(similarities, normalizer);
}
#endif
