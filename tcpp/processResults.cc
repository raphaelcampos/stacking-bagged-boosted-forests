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
#include "ttmath/ttmath.h"

double minPenality = 9999.99;

class Score {
 public:
  bool operator<=(const Score &rhs) const {
    return (this->sc > rhs.sc);
  }
  bool operator<(const Score &rhs) const {
    return (this->sc >= rhs.sc);
  }

  Score(const std::string &c, double &s) : tc(c), sc(s) {}
  std::string target_class() {return tc;}
  double score() {return sc;}
 private:
  std::string tc;
  double sc;
};

void stringTokenize(const std::string &str, std::vector<std::string> &tokens,
                    const std::string &delimiters = " ") {
  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  std::string::size_type pos = str.find_first_of(delimiters, lastPos);
  while (std::string::npos != pos || std::string::npos != lastPos) {
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    lastPos = str.find_first_not_of(delimiters, pos);
    pos = str.find_first_of(delimiters, lastPos);
  }
}

double timePenalty(int testYear, int trainYear, std::map<int, double> penalities) {
  if (penalities.find(testYear-trainYear) != penalities.end())
    return (double)penalities[testYear-trainYear] * 10.0;
  else return (double)minPenality;
}

void readPenalitiesFile(const char *filename, std::map<int, double> &penal) {
  std::ifstream file(filename);
  std::string line;
  while (file >> line) {
    std::vector<std::string> tokens;
    stringTokenize(line, tokens, ";");
    int dist = atoi(tokens[0].c_str());
    double penality = atof(tokens[1].c_str());
    penal[dist] = penality;
    if (penality < minPenality) minPenality = penality;
  }
  file.close();
}

void computeScores(const char *s, std::map<int, double> &penal, int round) {
  std::cout << "#" << round << std::endl;
  std::ifstream file(s);
  std::string line;

  std::map<std::string, std::string> classes;

  //           docId                  class      score
  std::map< std::string, std::map < std::string, double > >  scores;
  while (std::getline(file, line)) {
    if (line[0] == '#') continue;

    std::vector<std::string> tokens;
    stringTokenize(line, tokens, " ");
    std::string idDoc = tokens[0];

    std::string cmbReal = tokens[1];
    std::vector<std::string> tmp;

    stringTokenize(cmbReal, tmp, "-");
    std::string classeReal = tmp[0];
    classes[idDoc] = classeReal;

    std::string anoReal = tmp[1];
    tmp.clear();

    for (unsigned int i = 2; i < tokens.size(); i++) {
      stringTokenize(tokens[i], tmp, ":");
      std::string cmb = tmp[0];
      double valor(atof(tmp[1].c_str()));
      tmp.clear();
      stringTokenize(cmb, tmp, "-");
      std::string cl  = tmp[0];
      std::string ano = tmp[1];
      tmp.clear();

      double ajuste = timePenalty(atoi(anoReal.c_str()), atoi(ano.c_str()), penal);

      std::map < std::string, double >::iterator it = scores[idDoc].find(cl);
      if (it == scores[idDoc].end()) scores[idDoc][cl] = valor * ajuste;
      else scores[idDoc][cl] += valor * ajuste;
    }

  }

  file.close();

  std::cerr << "Scores computed for " << scores.size() << " test examples." << std::endl;

  std::map < std::string, std::map< std::string, double > >::iterator it = scores.begin();
  while (it != scores.end()) {
    std::string idDoc = it->first;
    std::string classeReal = classes[idDoc];

    std::cout << idDoc << " " << classeReal;

    std::set<Score> ordered_scores;

    std::map<std::string, double>::iterator cIt = it->second.begin();
    while (cIt != it->second.end()) {
      Score s(cIt->first, cIt->second);
      ordered_scores.insert(s);
      ++cIt;
    }

    std::set<Score>::iterator sIt = ordered_scores.begin();
    while (sIt != ordered_scores.end()) {
      Score s = *sIt;
      std::cout << " " << s.target_class() << ":" << s.score();
      ++sIt;
    }
    std::cout << std::endl;
    ++it;
  }

}

void processResults(const char *p, const char *s, int round) {
  std::map<int, double> penal;
  readPenalitiesFile(p, penal);
  computeScores(s, penal, round);
}


int main(int argc, char** argv) {
  int opt; int round = 0;
  char *penalties = NULL;
  char *scores = NULL;
  while ((opt = getopt (argc, argv, "p:s:r:")) != -1) {
    switch (opt) {
      case 'p': penalties = optarg; break;
      case 's': scores = optarg; break;
      case 'r': round = atoi(optarg); break;
      case '?':
        if ((optopt == 'p') || (optopt == 's'))
          fprintf (stderr, "Option -%c requires an argument.\n", optopt);
        else if (isprint (optopt))
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf (stderr, "Unknown option character `-%c'.\n", optopt);
        return 1;
      default:
        abort ();
    }
  }
  processResults(penalties, scores, round);
}


