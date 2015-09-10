#ifndef ABSTRACT_FEATURE_SELECTOR_HPP__
#define ABSTRACT_FEATURE_SELECTOR_HPP__

#include <string>
#include <set>
#include <queue>

#include "utils.hpp"

class AbstractFeatureSelector;

class Term {
 friend class AbstractFeatureSelector;
 public:
  Term(const int &id, double w) : term(id), weight(w) {}
  // Terms will be inserted in a priority_queue. Terms with higher
  // weights will be retrieved first. So, let us invert this:
  // terms with smaller weights will have greater priority (so as to
  // properly retrieve the worst ones and fill the filtered set)
  bool operator<(const Term &rhs) const { return weight > rhs.weight; }

 private:
  int term;
  double weight;
};

class AbstractFeatureSelector {
 public:
  AbstractFeatureSelector(const std::string &i, double p, bool r=false) : // r = round robin approach
    input(i), percentage(p), rr(r) {}

  virtual void select() = 0; // feature selection method

  void select (const std::string &in) {
    input = in;
    select();
  }

  virtual std::string filter(const std::string &l) = 0;

  virtual ~AbstractFeatureSelector() {}

 protected:
  void push_term(const Term &t) { ordered.push(t); }
  void push_term(const std::string &c, const Term &t)
    { ordered_class[c].push(t); }

  void fill_filtered() {
    if (rr) {
      std::map<std::string, std::priority_queue<Term,
                            std::vector<Term> > >::iterator it = ordered_class.begin();
      while(it != ordered_class.end()) {
        unsigned int n = static_cast<unsigned int >(
          static_cast<double>(it->second.size()) * percentage);
        std::cerr << "[ABSTRACT_FS] Removing " << n << " attributes from " << it->second.size() << " Class=" << it->first << std::endl;
        unsigned int cur = 0;
        while(!it->second.empty() && cur < n) {
          filtered_class[it->first].insert(it->second.top().term);
          if (cur > n-5) std::cerr << "     > t=" << it->second.top().term << ", w=" << it->second.top().weight << std::endl;
          it->second.pop();
          ++cur;
        }
        ++it;
      }
    }
    else {
      unsigned int n = static_cast<unsigned int >(
        static_cast<double>(ordered.size()) * percentage);
      std::cerr << "[ABSTRACT_FS] Removing " << n << " attributes from " << ordered.size() << std::endl;
      unsigned int cur = 0;
      while(!ordered.empty() && cur < n) {
        filtered.insert(ordered.top().term);
        if (cur > n-5) std::cerr << "     > t=" << ordered.top().term << ", w=" << ordered.top().weight << std::endl;
        ordered.pop();
        ++cur;
      }
    }
  }

  std::string input;
  double percentage;
  bool rr;
  std::set<int> filtered; // feature_ids to be removed
  std::map<std::string, std::set<int> > filtered_class;

  // contingency table
  std::map<std::string, unsigned int> n_kc_;
  std::map<std::string, unsigned int> n_c_;
  std::map<int, unsigned int> n_k_;
  unsigned int n_;

  std::set<std::string> classes;
  std::set<int> vocabulary;
  std::map<std::string, std::set<int> > vocabulary_class;

  std::map<std::string, unsigned int> tf_;
  std::map<std::string, unsigned int> tfc_;

  std::priority_queue<Term, std::vector<Term> > ordered;
  std::map<std::string, std::priority_queue<Term,
                        std::vector<Term> > > ordered_class;
 
};
#endif
