#include <map>
#include <set>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

#include "functs.hpp"

class Score {
  public:
    Score() {}
    Score(const std::string &cl, const long double sc) : class_name(cl), score(sc) {}
    bool operator<(const Score &rhs) const { return score > rhs.score; }

    std::string class_name;
    long double score;
};

void parse_test_file(const char *fn, std::map<unsigned int, std::string> &true_class) {
  std::ifstream file(fn);
  if (file) {
    std::string line;
    unsigned int doc_id = 0;
    while (std::getline(file, line)) {
      std::vector<std::string> tokens;
      stringTokenize(line, tokens, ";");
      if (tokens.size() > 3) {
        std::string cl = tokens[2]; cl.replace(0, 6, "");
        true_class[doc_id] = cl;
        ++doc_id;
      }
    }
    file.close();
  }
  else {
    std::cout << "Error while opening test file." << std::endl;
    exit(1);
  }
}

void parse_predictions(const char *fn, const std::map<unsigned int, std::string> &true_class) {
  std::ifstream file(fn);
  if (file) {
    std::vector<std::string> cl_ordered;
    unsigned int doc_id=0;

    std::string header; std::getline(file, header);
    std::vector<std::string> tks_h;
    stringTokenize(header, tks_h, " ");
    cl_ordered.resize(tks_h.size()-1);
    for (unsigned int i = 1; i < tks_h.size(); i++) {
      std::cerr << "cl_ordered[" << (i-1) << "] = " << tks_h[i] << std::endl;
      cl_ordered[i-1] = tks_h[i];
    }

    std::string line;
    while (std::getline(file, line)) {
      std::vector<std::string> tokens;
      stringTokenize(line, tokens, " ");
      std::set<Score> scores;
      if (tokens.size() > 2) {
        for (unsigned int i = 1; i < tokens.size(); i++)
          scores.insert(Score(cl_ordered[i-1], atof(tokens[i].data())));

        std::map<unsigned int, std::string>::const_iterator d_it = true_class.find(doc_id);
        if (d_it != true_class.end()) {
          std::cout << doc_id << " " << d_it->second;     
          std::set<Score>::const_iterator it = scores.begin();
          while (it != scores.end()) {
            std::cout << " " << it->class_name << ":" << it->score;
            ++it;
          }
          std::cout << std::endl;
        }
        else {
          std::cerr << " -> doc_id " << doc_id << " mismatch !! aborting." << std::endl; exit(1);
        }
        ++doc_id;
      }
    }
    file.close();
  }
  else {
    std::cout << "Error while opening svm predictions file." << std::endl;
    exit(1);
  }
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "USAGE: " << argv[0] << " [SVM_PROB_OUTPUT] [TEST]" << std::endl;
    return 1;
  }
  std::map<unsigned int, std::string> true_class;
  parse_test_file(argv[2], true_class);
  parse_predictions(argv[1], true_class);
}
