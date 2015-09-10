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

using namespace std;

double minPenality = 9999.99;

struct Document {
  double size;
  map<string, double> terms;
};

struct rc_model {
  map<string, int> sumDFperClass;
  map<string, int> DFperTerm;
  map<string, map< string, map< string, double > > > sumTF;

  map<string, double> centSizes; // class-refYear -> size
  int totalDocs;
  set<string> years;
  set<string> classes;
  set<string> vocabulary;
  map< string, set<string> > classVocab;
  double maxIDF;
};

struct Similarity {
  string className;
  double similarity;
};

struct cmpSimilarity {
  bool operator()(const Similarity a, const Similarity b) {return a.similarity < b.similarity;}
};

void stringTokenize(const string& str, vector<string>& tokens, const string& delimiters = " ")
{
  // Skip delimiters at beginning.
  string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  // Find first "non-delimiter".
  string::size_type pos = str.find_first_of(delimiters, lastPos);
  while (string::npos != pos || string::npos != lastPos) {
    // Found a token, add it to the vector.
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    // Skip delimiters.  Note the "not_of"
    lastPos = str.find_first_not_of(delimiters, pos);
    // Find next "non-delimiter"
    pos = str.find_first_of(delimiters, lastPos);
  }
}

string getCompIndex(string docClass, string year) {
  return docClass + "-" + year;
}

// Train docs parser: update term frequencies for priori and cond.
istream& operator >> (istream& is, rc_model& model) {
  string line;
  is >> line;
  if (line == "") return is;

  vector<string> tokens;
  stringTokenize(line, tokens, ";");

  string id = tokens[0];
  string year = tokens[1];
  model.years.insert(year);

  model.totalDocs++;

  // remove CLASS= mark
  string docClass = tokens[2];
  model.classes.insert(docClass);

  model.sumDFperClass[docClass]++;

  string compClass = getCompIndex(docClass, year);

  // retrieve each term frequency and update occurrencies
  for (unsigned int i = 3; i < tokens.size()-1; i+=2) {
    double weight = 1.0 + log10(atof(tokens[i+1].c_str()));
    string termId = tokens[i];
    model.vocabulary.insert(termId);
    model.DFperTerm[termId]++;
    model.sumTF[docClass][termId][year] += weight;
    model.classVocab[docClass].insert(termId);
  }

  return is;
}

void updateMaxIDF (rc_model &model) {
  set<string>::iterator vocIt = model.vocabulary.begin();
  double maxIDF = -9999.99;
  while (vocIt != model.vocabulary.end()) {
    double idf =  log10((((double)model.totalDocs + 1) / ((double)model.DFperTerm[*vocIt] + 1 )));
    if (maxIDF < idf) maxIDF = idf;
    ++vocIt;
  }
  model.maxIDF = maxIDF;
}

int trainRocchio(const char* filename, rc_model &model) {
  ifstream file(filename);
  model.totalDocs = 0;

  if (file) {
    while (file >> model) ;
    file.close();
  }
  else {
    cout << "Error while opening training file." << endl;
    exit(1);
  }
  updateMaxIDF(model);
  model.vocabulary.clear();
  return 0;
}

double timePenalty(int testYear, int trainYear, map<int, double> penalities, bool isBaseLine) {
  if (isBaseLine) return 1.0;
  if (penalities.find(testYear-trainYear) != penalities.end())
    return penalities[testYear-trainYear];
  else return minPenality;
}

/**************************************************************************************************************************
 * TEMPORAL CLASSIFIER  FUNCTIONS
 **************************************************************************************************************************/
double readPenalitiesFile(const char *filename, map<int, double> &penal) {
  ifstream file(filename);
  string line;
  double minPenalty = 9999.99;
  while (file >> line) {
    vector<string> tokens;
    stringTokenize(line, tokens, ";");
    int dist = atoi(tokens[0].c_str());
    double penality = atof(tokens[1].c_str());
    penal[dist] = penality;
    if (penality < minPenality) minPenality = penality;
  }
  return minPenalty;
}

void rocchio(const char* filename, rc_model &model, const char* penal, bool isBaseLine, int round) {
  ifstream file(filename);
  string line;

  if(file) {
    // read penalities file
    map<int, double> penalities;
    double minPenalty = readPenalitiesFile(penal, penalities);

    // read each test doc and parse it
    cout << "#" << round << endl;

    while (file >> line) {
      vector<string> tokens;
      stringTokenize(line, tokens, ";");

      priority_queue<Similarity, vector<Similarity>, cmpSimilarity> similarities;
      string id = tokens[0];
      string year = tokens[1];

      set<string>::iterator classIterator = model.classes.begin();
      while (classIterator != model.classes.end()) {
        string curClass = *(classIterator);

        // aggregate our final centroid, using temporal information (credibility
        // of each sub-centroid)
        double centSize = 0.0;

        set<string> remain; bool hasCentSize = false;
        if (model.centSizes.find(getCompIndex(curClass, year)) == model.centSizes.end()) {
            remain.insert(model.classVocab[curClass].begin(), model.classVocab[curClass].end());
        }
        else {
          centSize = model.centSizes[getCompIndex(curClass, year)]; hasCentSize = true;
        }

        double docSize = 0.0, sim = 0.0;
        for (unsigned int i = 3; i < tokens.size()-1; i+=2) {
          string termId = tokens[i];

          // test document (query)
          double tf = 1.0 + log10(atof(tokens[i+1].c_str()));
          double idf = (log10((((double)model.totalDocs + 1) / ((double)model.DFperTerm[termId] + 1 ))) / model.maxIDF);
          if (tf != tf || idf != idf) cerr << "Problem with tf/idf when calculating tfidf for docs." << endl; // chack for NaN
          double docTfidf = tf * idf;

          double centTfidf = 0.0;
          // retrieving centroid's corresponding value
          map<string, double>::iterator yearIt = model.sumTF[curClass][termId].begin();
          while(yearIt != model.sumTF[curClass][termId].end()) {
            centTfidf += yearIt->second * timePenalty(atoi(year.c_str()), atoi((yearIt->first).c_str()), penalities, isBaseLine);
            ++yearIt;
          }
          centTfidf *= (idf / model.sumDFperClass[curClass]); // idf and 'vectorial mean'

          if (!hasCentSize) {
              remain.erase(termId);
              centSize += pow(centTfidf, 2);
          }

          sim += (docTfidf * centTfidf);
          docSize += pow(docTfidf, 2);
        }

        if (!hasCentSize) {
          set<string>::iterator rIt = remain.begin();
          while(rIt != remain.end()) {
            string termId = *rIt;
            double val = 0.0;
            map<string, double>::iterator yearIt = model.sumTF[curClass][termId].begin();
            while(yearIt != model.sumTF[curClass][termId].end()) {
              val += yearIt->second * timePenalty(atoi(year.c_str()), atoi((yearIt->first).c_str()), penalities, isBaseLine);
              ++yearIt;
            }
            double idf = (log10((((double)model.totalDocs + 1) / ((double)model.DFperTerm[termId] + 1 ))) / model.maxIDF);
            val *= (idf / model.sumDFperClass[curClass]); // idf and 'vectorial mean'
            centSize += pow(val, 2);
            ++rIt;
          }
          model.centSizes[getCompIndex(curClass, year)] = centSize;
        }

        sim /= (sqrt(centSize) * sqrt(docSize));

        Similarity similar;
        similar.className = curClass;
        similar.similarity = sim;
        similarities.push(similar);
        ++classIterator;
      }

      // remove CLASS= mark
      string documentClass = tokens[2];

      // print scores
      cout << id << " " << documentClass;
      while(!similarities.empty()) {
        Similarity clSim = similarities.top();
        cout << " " << clSim.className << ":" << clSim.similarity;
        similarities.pop();
      }
      cout << endl;
    }
    file.close();
  }
  else {
    cout << "Error while opening testing file." << endl;
    exit(1);
  }
}

int main(int argc, char** argv) {

  rc_model model;

  int opt;
  char *testFile = NULL;
  char *trainFile = NULL;
  char *credFile = NULL;
  int round = 0;
  bool isBaseLine = true;
  while ((opt = getopt (argc, argv, "c:r:d:t:p")) != -1) {
    switch (opt) {
      case 't': testFile = optarg; break;
      case 'd': trainFile = optarg; break;
      case 'c': credFile = optarg; break;
      case 'p': isBaseLine = false; break;
      case 'r': round = atoi(optarg); break;
      case '?':
        if ((optopt == 'd') || (optopt == 't') || (optopt == 'r') || (optopt == 'c'))
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

  trainRocchio(trainFile, model);
  rocchio(testFile, model, credFile, isBaseLine, round);
}
