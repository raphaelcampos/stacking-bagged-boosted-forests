#ifndef NN_PROJECT_HPP__
#define NN_PROJECT_HPP__

#include <fstream>
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

#include "knn.hpp"

class nn_project : public virtual knn {
 public:
  nn_project(unsigned int k=30, unsigned int round=0) : knn(k, round) {}

  virtual ~nn_project() {}

 protected:
  virtual void getKNearest(unsigned int id, std::map<KNN_Document, double> &s,
                           Scores<double> &o, double &n);

};

void nn_project::getKNearest(unsigned int test_id,
                      std::map<KNN_Document, double> &similarities,
                      Scores<double> &ordered_sim_class, double &norm) {
  std::priority_queue<similarity_t, std::vector<similarity_t> > sim;
  std::map<KNN_Document, double>::iterator it = similarities.begin();
  while (it != similarities.end()) {
    double s = it->second;
    switch(dist_type) {
      case L2:
        s = 1.0 - sqrt(s);
        break;
      case L1:
        s = 1.0 - s;
        break;
    }

    similarity_t simil(it->first, s);
    sim.push(simil);
    ++it;
  }

  norm = -9999.99;
  std::map<std::string, double> sim_classes;
  unsigned int cur = 0;
  while(cur < k && !sim.empty()) {
    similarity_t simil = sim.top();
    sim_classes[simil.doc.doc_class] += 1;
    sim.pop();
    cur++;
  }

  std::map<std::string, double>::iterator sIt = sim_classes.begin();
  while (sIt != sim_classes.end()) {
    ordered_sim_class.add(sIt->first, sIt->second);
    ++sIt;
  }
}

#endif
