#include <iostream>
#include <cstring>
#include <vector>
#include <map>
#include <climits>
#include <fstream>
#include <cstdlib>

#include "utils.h"
#include "interval.h"

void stringTokenize(const std::string& str,
    std::vector<std::string>& tokens, const std::string& delimiters = " ") {
  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  std::string::size_type pos = str.find_first_of(delimiters, lastPos);
  while (std::string::npos != pos || std::string::npos != lastPos) {
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    lastPos = str.find_first_not_of(delimiters, pos);
    pos = str.find_first_of(delimiters, lastPos);
  }
}

int readDataset(const char* filename, std::map<unsigned int, std::string> &docs,
                std::map< std::string, std::vector < unsigned int > > &classMap) {

  std::ifstream file(filename);

  if (file) {
    std::string line;
    unsigned int docId = 0;
    while (std::getline(file, line)) {
      std::vector<std::string> tokens;
      stringTokenize(line, tokens, ";");
      unsigned int docId = docId++;
      std::string cl = tokens[2];
      docs[docId] = line;
      classMap[cl].push_back(docId);
    }
    file.close();
    return docs.size();
  }
  else {
    std::cout << "Error while opening training file." << std::endl;
    exit(1);
  }
}

void randomSampling(int seed, double percent, std::map<unsigned int, std::string> &docs,
                    std::map< std::string, std::vector <unsigned int > > &classMap) {

  Utils::srand(seed);

  unsigned int numDocs = static_cast<unsigned int>(percent * docs.size());

  // select ramdomly a document with probability based on class distribution
  for(unsigned i = 0; i < numDocs; i++) {
    // first, we construct the roulette wheel:
    unsigned int init = 0, finish = 0;
    std::map< Interval < unsigned int >, std::string> roulette;
    std::map< std::string, std::vector <unsigned int > >::iterator mapIt = classMap.begin();
    while (mapIt != classMap.end()) {
      unsigned int size = mapIt->second.size();
      finish = init + size;
      Interval<unsigned int> interval(init, finish);
      roulette[interval] = mapIt->first;
      init = finish;
      ++mapIt;
    }

    unsigned int prob = Utils::rand(0, finish);
    Interval<unsigned int> target(prob, prob);
    std::map< Interval < unsigned int >, std::string>::iterator it = roulette.find(target);
    if (it == roulette.end()) {
      std::string msg = "Could not find probability on roulette: " + Utils::toString(prob) + ".";
      Utils::errorMessage(msg);
    }

    std::string targetClass = it->second;
    // print document from selected class and remove this doc.

    unsigned int pos = Utils::rand(0, classMap[targetClass].size());
    unsigned int idxToPrint = classMap[targetClass][pos];

    // print selected document to stdout and remove it from our indices
    std::cout << docs[idxToPrint] << std::endl;
    docs.erase(idxToPrint);
    classMap[targetClass].erase(classMap[targetClass].begin() + pos);
    if (classMap[targetClass].empty()) classMap.erase(targetClass);
  }
  // now we print the remaining files to stderr :)
   std::map< std::string, std::vector <unsigned int > >::iterator mapIt = classMap.begin();
   while (mapIt != classMap.end()) {
     unsigned int numDocs = mapIt->second.size();
     for (unsigned int i = 0; i < numDocs; i++) {
       std::cerr << docs[(mapIt->second)[i]] << std::endl;
     }
     ++mapIt;
   }

}

int main(int argc, char** argv) {
  int opt;
  char *datasetFile = NULL;
  double percent = 0;
  int seed = 0;

  while ((opt = getopt (argc, argv, "d:p:s:")) != -1) {
    switch (opt) {
      case 'd': datasetFile = optarg; break;
      case 'p': percent = atof(optarg); break;
      case 's': seed = atoi(optarg); break;
      case '?':
        if ((optopt == 'd') || (optopt == 'p') || (optopt == 's'))
          fprintf (stderr, "Option -%c requires an argument.\n", optopt);
        else if (isprint (optopt))
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
        return 1;
      default:
        abort ();
    }
  }

  std::map<unsigned int, std::string> docs;
  std::map< std::string, std::vector < unsigned int > > classMap;
  readDataset(datasetFile, docs, classMap);

  randomSampling(seed, percent, docs, classMap);
}
