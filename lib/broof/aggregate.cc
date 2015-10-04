#include <fstream>
#include <iostream>
#include <vector>
#include <queue>

#include "utils.hpp"

class Score {
 public:
  Score(const std::string &cl, const double s) : cl_(cl), score_(s) {}
  bool operator< (const Score &rhs) const { return rhs.score_ > score_; }
  std::string cl_;
  double score_;
};

void process_result(const char *inputH, const char *inputF,
                    double lambda, int round) {
  std::ifstream file_h(inputH);
  std::ifstream file_f(inputF);

  if (file_h && file_f) {
    std::cout << "#" << round << std::endl;

    std::string line_h, line_f;
    while (std::getline(file_h, line_h) &&
           std::getline(file_f, line_f)) {
      std::vector<std::string> tokens_h;
      std::vector<std::string> tokens_f;
      Utils::string_tokenize(line_h, tokens_h, " ");
      Utils::string_tokenize(line_f, tokens_f, " ");
      if (tokens_h.size() < 3) continue;

      if (tokens_h[0].compare(tokens_f[0]) != 0) {
        std::cerr << "Both results should have documents in same order !" << std::endl;
        exit(1);
      }
      std::string id = tokens_h[0];
      std::string cl = tokens_h[1];

      std::map<std::string, double> res;
      for (unsigned int i = 2; i < tokens_h.size(); i++) {
        std::vector<std::string> tmp;
        Utils::string_tokenize(tokens_h[i], tmp, ":");
        res[tmp[0]] = lambda * atof(tmp[1].data());
      }
      for (unsigned int i = 2; i < tokens_f.size(); i++) {
        std::vector<std::string> tmp;
        Utils::string_tokenize(tokens_f[i], tmp, ":");
        res[tmp[0]] += (1-lambda) * atof(tmp[1].data());
      }     

      // ordering the results
      std::priority_queue<Score, std::vector<Score> > ordered;
      std::map<std::string, double>::iterator it = res.begin();
      while (it != res.end()) {
        ordered.push(Score(it->first, it->second));
        ++it;
      }

      std::cout << id << " " << cl;
      while (!ordered.empty()) {
        Score s = ordered.top();
        std::cout << " " << s.cl_ << ":" << s.score_;
        ordered.pop();
      }
      std::cout << std::endl;
    }
    file_h.close();
    file_f.close();
  }
  else {
    std::cerr << "Unable to open affinity matrix files (" << inputH << " and/or " << inputF << ")" << std::endl;
    exit(1);
  }
}

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " [affinity_high] [affinity_low] [lambda] [round]" << std::endl;
    exit(1);
  }
  process_result(argv[1], argv[2], atof(argv[3]), atoi(argv[4]));
  return 0;
}
