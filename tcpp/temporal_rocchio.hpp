#ifndef TEMPORAL_ROCCHIO_HPP_H__
#define TEMPORAL_ROCCHIO_HPP_H__

#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <vector>
#include <set>
#include <queue>
#include <utility>
#include <fstream>
#include <stdlib.h>

#include "supervised_classifier.hpp"
#include "temporal_classifier.hpp"
#include "rocchio.hpp"

class TemporalRocchio : public Rocchio, public TemporalClassifier {
 public:
  TemporalRocchio(const std::string &twf, unsigned int r=0, double b=1.0,
      bool w = false, bool gw = false, unsigned int w_sz=0) :
    Rocchio(r), TemporalClassifier(twf, r, b, w, gw, w_sz) { read_twf(twf); }

  virtual bool parse_train_line(const std::string &l);
  virtual void parse_test_line(const std::string &l);
  virtual bool check_train_line(const std::string &l) const;

  virtual void reset_model() {
    TemporalClassifier::reset_model();
    Rocchio::reset_model();

    class_vocab.clear();
    cent_sizes.clear();
    sumTF.clear();
  }

  virtual void train(const std::string &trn); // overriding to compute just the idf

  virtual ~TemporalRocchio() {}

 private:
  std::map<std::string, std::set<int> > class_vocab;
  std::map<std::string, double> cent_sizes;

  std::map<std::string, std::map<int,
                        std::map<std::string, double > > > sumTF;

};

bool TemporalRocchio::check_train_line(const std::string &line) const {
  std::vector<std::string> tokens;
  Utils::string_tokenize(line, tokens, ";");
  // input format: doc_id;year;CLASS=class_name;{term_id;tf}+
  if ((tokens.size() < 5) || (tokens.size() % 2 == 0)) return false;
  return true;
}

bool TemporalRocchio::parse_train_line(const std::string &line) {
  std::vector<std::string> tokens;
  Utils::string_tokenize(line, tokens, ";");
  // input format: doc_id;CLASS=class_name;{term_id;tf}+
  if ((tokens.size() < 5) || (tokens.size() % 2 == 0)) return false;

  std::string id = tokens[0];
  std::string year = tokens[1];
  years_add(year);

  // remove CLASS= mark
  std::string doc_class = tokens[2];
  classes_add(doc_class);

  sumDFperClass[doc_class]++;

  std::string cl_yr_idx = Utils::get_index(doc_class, year);
  for (unsigned int i = 3; i < tokens.size()-1; i+=2) {
    double weight = (raw) ? atof(tokens[i+1].data()) :
                    1.0 + log10(atof(tokens[i+1].data()));
    int term_id = atoi(tokens[i].data());
    vocabulary_add(term_id);
    DFperTerm[term_id]++;
    sumTF[doc_class][term_id][year] += weight;
    class_vocab[doc_class].insert(term_id);
  }

  return true;
}

void TemporalRocchio::train(const std::string &trn) {
  Rocchio::train(trn);
  if (!raw) updateMaxIDF();
}

void TemporalRocchio::parse_test_line(const std::string &line) {
  std::vector<std::string> tokens;
  Utils::string_tokenize(line, tokens, ";");
  if ((tokens.size() < 5) || (tokens.size() % 2 == 0)) return;

  Scores<double> similarities(tokens[0], tokens[2]);
  std::string year = tokens[1];

  double normalizer = 0.0;

  std::set<std::string>::const_iterator cIt = classes_begin();
  while (cIt != classes_end()) {
    std::string cur_class = *cIt;

    double cent_size = 0.0;
    std::set<int> remain;
    bool has_cent_size = false;

    #pragma omp critical (cent_size_query)
    {
      #pragma omp flush (cent_sizes)
      std::map<std::string, double>::const_iterator cent_sz_it = 
        cent_sizes.find(Utils::get_index(cur_class, year));
      if (cent_sz_it == cent_sizes.end()) {
        remain.insert(class_vocab[cur_class].begin(),
                      class_vocab[cur_class].end());
      }
      else {
        cent_size = cent_sz_it->second;
        has_cent_size = true;
      }
    }

    double sim = 0.0;
    double doc_size = 0.0;

    for (size_t i = 3; i < tokens.size()-1; i+=2) {
      int term_id = atoi(tokens[i].data());
      double tf = (raw) ? atof(tokens[i+1].data()) :
        1.0 + log10(atof(tokens[i+1].data()));
      double idf = (raw) ? 1.0 :
        log10(static_cast<double>(get_total_docs() + 1.0) /
              static_cast<double>(Utils::get_value(DFperTerm, term_id)+1.0)) 
                / maxIDF;

      double train_tfidf = tf * idf;
      double cent_tfidf = 0.0;

      std::map<std::string, std::map<int,
                            std::map<std::string, double> > >::const_iterator
        fst_it = sumTF.find(cur_class);
      if (fst_it != sumTF.end()) {
        std::map<int, std::map<std::string, double> >::const_iterator
          snd_it = fst_it->second.find(term_id);
        if (snd_it != fst_it->second.end()) {
          std::map<std::string, double>::const_iterator yIt = snd_it->second.begin();
          while(yIt != snd_it->second.end()) {
            double t = twf(atoi(year.data()), atoi((yIt->first).data()));
            double val = yIt->second * t; 
            cent_tfidf += val;
            ++yIt;
          }
        }
      }

      cent_tfidf *= (idf / Utils::get_value(sumDFperClass, cur_class));
      if (!has_cent_size) {
        remain.erase(term_id);
        cent_size += pow(cent_tfidf, 2);
      }

      sim += (train_tfidf * cent_tfidf);
      doc_size += pow(train_tfidf, 2);
    }

    if (!has_cent_size) {
      std::set<int>::const_iterator rIt = remain.begin();
      while(rIt != remain.end()) {
        int term_id = *rIt;
        double val = 0.0;
        std::map<std::string, std::map<int,
                              std::map<std::string, double> > >::const_iterator
          fst_it = sumTF.find(cur_class);
        if (fst_it != sumTF.end()) {
          std::map<int, std::map<std::string, double> >::const_iterator
            snd_it = fst_it->second.find(term_id);
          if (snd_it != fst_it->second.end()) {
            std::map<std::string, double>::const_iterator yIt = snd_it->second.begin();
            while(yIt != snd_it->second.end()) {
              double w = yIt->second * twf(atoi(year.data()),
                                       atoi((yIt->first).data()));
              val += w; 
              ++yIt;
            }
          }
        }
        double idf = (raw) ? 1.0 :
          log10(static_cast<double>(get_total_docs() + 1.0) /
            static_cast<double>(Utils::get_value(DFperTerm,term_id)+1.0)) 
              / maxIDF;
        val *= (idf / Utils::get_value(sumDFperClass, cur_class));
        cent_size += pow(val, 2);
        ++rIt;
      }

      #pragma omp critical (cent_size_update)
      {
        #pragma omp flush (cent_sizes)
        cent_sizes[Utils::get_index(cur_class, year)] = cent_size;
      }
    }

    sim /= (sqrt(cent_size) * sqrt(doc_size)); // cosine similarity
    similarities.add(cur_class, sim);
    normalizer += sim;
    ++cIt;
  }
  get_outputer()->output(similarities, normalizer);
}
#endif
