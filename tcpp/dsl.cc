#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <vector>
#include <set>
#include <queue>
#include <fstream>
#include <sstream>
#include <time.h>
#include <cstdlib>

#include "utils.hpp"

class DocEL {
  public:
      DocEL(std::string d, double e) : docId(d), el(e) {}
      std::string docId;
      double el;
};

struct ComparatorDocEL {
      bool operator()(const DocEL a, const DocEL b) {return a.el > b.el;}
};

struct term_model {
  std::map<std::string, int> domNum;
  std::map<std::string, int> domDen;

  std::map<std::string, std::string> dataset_;

  std::set<std::string> years;
  std::set<std::string> classes;

  std::map<std::string, double> stabilityLevel;

  double minDom; // minimum dominance (e.g. 0.5)
};

std::istream& operator >> (std::istream& is, term_model& model) {
  std::string line;
  is >> line;
  if (line == "") return is;

  std::vector<std::string> tokens;
  Utils::string_tokenize(line, tokens, ";");

  std::string docId = tokens[0];
  model.dataset_[docId] = line;
  std::string year = tokens[1];
  model.years.insert(year);
  std::string docClass = tokens[2];
  model.classes.insert(docClass);

  for (unsigned int i = 3; i < tokens.size()-1; i+=2) {
    std::string termId = tokens[i];
    std::string index = Utils::get_index(termId, Utils::get_index(docClass, year));
    model.domNum[index]++;
    model.domDen[Utils::get_index(termId, year)]++;
  }

  return is;
}

bool termDominated(term_model &model, std::string termId, std::string year) {
  std::set<std::string>::iterator clIt = model.classes.begin();
  while (clIt != model.classes.end()) {
    std::string index = Utils::get_index(termId, Utils::get_index(*clIt, year));
    std::string indexDen = Utils::get_index(termId, year);
    double dominance = 0.0;
    if (model.domNum.find(index) != model.domNum.end() &&
        model.domDen.find(indexDen) != model.domDen.end()) {
      double num = model.domNum[index], den = model.domDen[indexDen];
      dominance = ((num == 0) || (den == 0)) ? 0.0 : ( num / den );
    }
    index.clear(); indexDen.clear();
    if (dominance > model.minDom) return true;

    ++clIt;
  }
  return false;
}

unsigned int retrieveStabilityPeriodSize(term_model &model, std::string termId, std::string yr) {
  unsigned int size = 0; bool lookPast = true, lookFuture = true;
  std::set<std::string>::iterator refYear = model.years.find(yr);
  if (refYear != model.years.end()) {

    // disconsider present
    if (termDominated(model, termId, *refYear)) {
      size++;
    }

    // verify on past and future
    std::set<std::string>::iterator pastIt = std::set<std::string>::iterator(refYear);
    if (pastIt == refYear) lookPast = false;
    else pastIt--;

    std::set<std::string>::iterator futureIt = std::set<std::string>::iterator(refYear);
    if (*futureIt == *(model.years.rbegin())) lookFuture = false;
    else futureIt++;

    // future (model.years.end() eh o flag de fim)
    while (futureIt != model.years.end() && lookFuture) {
      if (termDominated(model, termId, *futureIt)) {
        size++;
      }
      else {
        break;
      }
      ++futureIt;
    }

    // past
    bool verifyFirstYear = true;
    while(pastIt != model.years.begin() && lookPast) {
      if (termDominated(model, termId, *pastIt)) {
        size++;
      }
      else {
        verifyFirstYear = false;
        break;
      }
      --pastIt;
    }
    // verify first year here...
    if (verifyFirstYear && model.years.begin() != refYear) {
      if (termDominated(model, termId, *(model.years.begin()))) {
        size++;
      }
    }
  }
  else {
    std::cerr << "Failed to fetch reference year on model (" << yr << "). Aborting." << std::endl;
    std::set<std::string>::iterator y = model.years.begin();
    while(y != model.years.end()) {
      std::cerr << *y << " ";
      ++y;
    }
    std::cerr << std::endl;
    exit(1);
  }
  return size;
}

double computeTemporalStabilityIndex(term_model &model, std::string termId, std::string refYr) {;
    unsigned int size = retrieveStabilityPeriodSize(model, termId, refYr);
    return (static_cast<double>(size) / static_cast<double>(model.years.size()));

}

void computeStabilityLevel(std::istream &is, term_model &model) {
  std::string line;
  while(is >> line) {
    if (line == "") continue;

    std::vector<std::string> tokens;
    Utils::string_tokenize(line, tokens, ";");

    std::string docId = tokens[0];
    std::string year = tokens[1];

    unsigned int numTerms = 0;
    for (unsigned int i = 3; i < tokens.size()-1; i+=2) {
      std::string termId = tokens[i].c_str();
      model.stabilityLevel[docId] += computeTemporalStabilityIndex(model, termId, year);
      termId.clear();
      numTerms++;
    }
    model.stabilityLevel[docId] /= static_cast<double>(numTerms);
  }
}

void verifyTermDistribution(const char* dataFile,
                            term_model &model) {
  std::ifstream file(dataFile);
  // first pass: term dominance computation
  if (file) {
    while (file >> model);
    file.close();
  }
  else {
    std::cerr << "Error while opening dataset file." << std::endl;
    exit(1);
  }

  file.open(dataFile);
  // third pass: stability level for each document
  if (file) {
    computeStabilityLevel(file, model);
    file.close();
  }
  else {
    std::cerr << "Error while opening target file." << std::endl;
    exit(1);
  }
}

void summary(term_model &model, double b) {
  std::priority_queue<DocEL, std::vector<DocEL>, ComparatorDocEL> ordered;
  std::map<std::string, double>::iterator dIt = model.stabilityLevel.begin();
  while(dIt != model.stabilityLevel.end()) {
    ordered.push(DocEL(dIt->first, dIt->second));
    ++dIt;
  }

  while(!ordered.empty()) {
    DocEL el = ordered.top();
    if (el.el <= b) std::cout << /*el.el*/ model.dataset_[el.docId] << std::endl;
    else std::cerr << /*el.el*/ model.dataset_[el.docId] << std::endl;
    ordered.pop();
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " [data_file] [threshold]" << std::endl;
    exit(1);
  }
  term_model model;
  verifyTermDistribution(argv[1], model);
  summary(model, atof(argv[2]));
  return 0;
}
