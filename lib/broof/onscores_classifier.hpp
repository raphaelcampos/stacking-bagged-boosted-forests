#ifndef ONSCORES_CLASSIFIER_HPP__
#define ONSCORES_CLASSIFIER_HPP__

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

class OnScoresClassifier : public TemporalClassifier {
 public:
  OnScoresClassifier(const std::string &twf, SupervisedClassifier *c, unsigned int r=0, double b=1.0,
    bool w = false, bool gw = false, unsigned int w_sz=0) :
    SupervisedClassifier(r), TemporalClassifier(twf,r,b,w,gw,w_sz), cc(c)
      { cc->reset(); read_twf(twf); }
  virtual ~OnScoresClassifier() {};

  virtual void train(const std::string &train_fn) {
    std::string train_fn_tmp = ".trn.Onsco_intermediate";
    std::ifstream file(train_fn.data());
    std::ofstream of(train_fn_tmp.data());
    if (file && of) {
      std::string line;
      while (file >> line) of << change_line(line) << std::endl;
      of.close();
      file.close();
      cc->train(train_fn_tmp);
    }
    else {
      std::cerr << "Error while opening training file." << std::endl;
      exit(1);
    }
    of.close(); file.close();
    unlink(train_fn_tmp.data());
  }

  virtual void test(const std::string &test_fn) {
    std::string test_fn_tmp = ".tst.Onsco_intermediate";
    std::string inter_res_fn = ".res_onsco";

    std::ifstream file(test_fn.data());

    std::ofstream of(test_fn_tmp.data());
    if (file && of) {
      std::string line;
      while (file >> line) of << change_line(line) << std::endl;
      of.close();
      file.close();
      Outputer *tmp = cc->get_outputer();
      cc->set_outputer(new FileOutputer(inter_res_fn));
      cc->test(test_fn_tmp);
      cc->set_outputer(tmp); tmp = NULL;
    }
    else {
      std::cerr << "Error while opening training file." << std::endl;
      exit(1);
    }
    of.close(); file.close();
    unlink(test_fn_tmp.data());

    file.open(inter_res_fn.data());
    if (file) {
      // aggregate scores
      std::string line;
      while (std::getline(file, line)) {
        if (line[0] == '#') continue;

        std::map<std::string, double> scores;
        // format: doc_id true_class-true_year class-yr:value
        std::vector<std::string> tokens;
        Utils::string_tokenize(line, tokens, " ");
        std::string id_doc = tokens[0];

        std::string cmb_real = tokens[1];
        std::vector<std::string> tmp;
        Utils::string_tokenize(cmb_real, tmp, "-");
        std::string real_class = tmp[0];
        std::string ref_year = tmp[1];
        tmp.clear();

        for (size_t i = 2; i < tokens.size(); i++) {
          Utils::string_tokenize(tokens[i], tmp, ":");
          std::string cmb = tmp[0];
          double sco = atof(tmp[1].data());
          tmp.clear();
          Utils::string_tokenize(cmb, tmp, "-");
          std::string cl  = tmp[0];
          std::string yr = tmp[1];
          tmp.clear();

          double w = twf(atoi(ref_year.data()), atoi(yr.data()));

          std::map < std::string, double >::iterator it = scores.find(cl);
          if (it == scores.end()) scores[cl] = sco * w;
          else it->second += sco * w;
        }
        Scores<double> sco(id_doc, real_class);
        std::map<std::string, double>::const_iterator it = scores.begin();
        while (it != scores.end()) {
          sco.add(it->first, it->second);
          ++it;
        }
        if (!get_outputer()) set_outputer(new Outputer());
        get_outputer()->output(sco);
      }
      file.close();
    }
    unlink(inter_res_fn.data());
  }

  // dummy definitions (this class delegates work)
  virtual bool parse_train_line(const std::string &l) { return true; };
  virtual void parse_test_line(const std::string &l) { return; };

  virtual void reset_model() { cc->reset_model(); }

 protected:
  std::string change_line(const std::string &l) {
    std::vector<std::string> tokens;
    Utils::string_tokenize(l, tokens, ";");
    // input format: doc_id;year;class_name;{term_id;tf}+
    if ((tokens.size() < 5) || (tokens.size() % 2 == 0)) return std::string("");

    std::string id = tokens[0];
    std::string year = tokens[1];
    std::string doc_class = tokens[2];

    std::string nl = id + ";" + Utils::get_index(doc_class, year);
    for (unsigned int i = 3; i < tokens.size(); i++) {
      nl = nl + ";" + tokens[i];
    }
    return nl;
  }

  SupervisedClassifier *cc;
};
#endif
