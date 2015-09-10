#ifndef EXT_ONSCORES_CLASSIFIER_HPP_
#define EXT_ONSCORES_CLASSIFIER_HPP__

#include <typeinfo>

#include <unistd.h>
#include <fstream>
#include <iostream>
#include <string>
#include <set>
#include <queue>
#include <cstdlib>

#include "supervised_classifier.hpp"
#include "temporal_classifier.hpp"

class ExtOnScoresClassifier : public TemporalClassifier {
 public:
  ExtOnScoresClassifier(const std::string &twf, SupervisedClassifier *c, unsigned int r=0,
      double b=1.0, bool w = false, bool gw = false, unsigned int w_sz=0) :
    SupervisedClassifier(r), TemporalClassifier(twf,r,b,w,gw,w_sz), cc(c)
      { cc->reset(); read_twf(twf); std::cerr << "beta=" << b << std::endl; }
  virtual ~ExtOnScoresClassifier() {};

  virtual double twf(unsigned int p_r, unsigned int p) {
    return TemporalClassifier::twf(p_r, p);
  }

  // retrieve training examples' offsets dividing by time point
  virtual void train(const std::string &train_fn) {
    std::ifstream file(train_fn.data());
    std::map<std::string, std::vector<unsigned long> > trn_offsets;

    if (file) {
      std::string line;
      unsigned long offset = file.tellg();
      while (std::getline(file, line)) {
        std::vector<std::string> tokens;
        Utils::string_tokenize(line, tokens, ";");
        std::string year = tokens[1];
        trn_offsets[year].push_back(offset);
        offset = file.tellg();
      }
      file.close();
    }
    else {
      std::cerr << "Error while opening training file." << std::endl;
      exit(1);
    }
    file.open(train_fn.data());
    if (file) {
      // writing temporary training files (one for each time point)
      std::map<std::string, std::vector<unsigned long> >::iterator it = trn_offsets.begin();
      while(it != trn_offsets.end()) {
        std::stringstream tmp_fn; tmp_fn << ".trn" << it->first;
        std::ofstream tmp_file(tmp_fn.str().data());
        if (tmp_file) {
          for (size_t i = 0; i < it->second.size(); i++) {
            file.seekg(it->second[i]);
            std::string line;
            if (std::getline(file, line)) tmp_file << change_line(line) << std::endl;
            else std::cerr << "Getline FAILED at " << it->first << " offset:" << it->second[i] << std::endl;
          }
          tmp_file.close();
        }
        ++it;
      }
      file.close();
    }
    else {
      std::cerr << "Error while opening training file." << std::endl;
      exit(1);
    }
  }

  virtual void test(const std::string &test_fn) {
    std::ifstream file(test_fn.data());
    std::map<std::string, std::string> actual_time_points;
   
    std::string tmp_tst(".tst"); 
    std::ofstream tmp_file(tmp_tst.data());
    if (file && tmp_file) {
      std::string line;
      while (std::getline(file, line)) {
        std::vector<std::string> tokens;
        Utils::string_tokenize(line, tokens, ";");
        std::string doc_id = tokens[0];
        std::string year = tokens[1];
        years_add(year);
        actual_time_points[doc_id] = year;
        tmp_file << change_line(line) << std::endl;
      }
      file.close();
    }
    else {
      std::cerr << "Error while opening test file." << std::endl;
      exit(1);
    }
    
    std::map<std::string, std::map<std::string, double> > agg_scores;
    std::map<std::string, std::string> actual_class;
    std::set<std::string>::const_iterator yIt = years_begin();
    while (yIt != years_end()) {
      std::stringstream trn; trn << ".trn" << *yIt;
      std::stringstream inter_fn; inter_fn << ".inter" << *yIt;
      cc->reset();
      cc->train(trn.str());
      Outputer *tmp = cc->get_outputer();
      cc->set_outputer(new FileOutputer(inter_fn.str()));
      cc->test(tmp_tst.data());
      cc->set_outputer(tmp); tmp = NULL;
      unlink(trn.str().data());
      // accumulate weighted scores per test example
      file.open(inter_fn.str().data());
      if (file) {
        std::string line;
                std::map<std::string, std::string> actual_classes;
        while (std::getline(file, line)) {
          if (line[0] == '#') continue;

          std::map<std::string, double> scores;
          // format: doc_id true_class {pred_class:value}+
          std::vector<std::string> tokens;
          Utils::string_tokenize(line, tokens, " ");
          std::string id_doc = tokens[0];
          std::string real = tokens[1];
          actual_class[id_doc] = real;
          for (size_t i = 2; i < tokens.size(); i++) {
            std::vector<std::string> tmp;
            Utils::string_tokenize(tokens[i], tmp, ":");
            std::string pred_class = tmp[0];
            double sco = atof(tmp[1].data());
            double w = twf(atoi(actual_time_points[id_doc].data()), atoi(yIt->data()));

            std::map<std::string, double>::iterator it = agg_scores[id_doc].find(pred_class);
            if (it == agg_scores[id_doc].end()) agg_scores[id_doc][pred_class] = sco * w;
            else it->second += sco * w;
          }
        }
        file.close();
      }
      unlink(inter_fn.str().data());
      ++yIt;
    }

    // output final scores in agg_scores
    std::map<std::string, std::map<std::string, double> >::iterator it = agg_scores.begin();
    while (it != agg_scores.end()) {
      std::string doc_id = it->first;
      std::string real_cl = actual_class[doc_id];
      Scores<double> sco(doc_id, real_cl);
      std::map<std::string, double>::const_iterator sIt = agg_scores[doc_id].begin();
      while (sIt != agg_scores[doc_id].end()) {
        sco.add(sIt->first, sIt->second);
        ++sIt;
      }
      if (!get_outputer()) set_outputer(new Outputer());
      get_outputer()->output(sco);
      ++it;
    }
    unlink(tmp_tst.data());
  }

  // dummy definitions (this class delegates work)
  virtual bool parse_train_line(const std::string &l) { return true; };
  virtual void parse_test_line(const std::string &l) { return; };

  virtual void reset_model() {
    cc->reset();
  }

 protected:
  std::string change_line(const std::string &l) {
    std::vector<std::string> tokens;
    Utils::string_tokenize(l, tokens, ";");
    // input format: doc_id;year;class_name;{term_id;tf}+
    if ((tokens.size() < 5) || (tokens.size() % 2 == 0)) {
      std::cerr << "[FAILED tkns=" << tokens.size() << "] " << l << std::endl;
      return std::string("");
    }

    std::string id = tokens[0];
    std::string year = tokens[1];
    std::string doc_class = tokens[2];

    std::string nl = id + ";" + doc_class;
    for (unsigned int i = 3; i < tokens.size(); i++) {
      nl = nl + ";" + tokens[i];
    }
    return nl;
  }

  SupervisedClassifier *cc;
};
#endif
