#ifndef CASCADED_TEMPORAL_OVERSAMPLING_HPP__
#define CASCADED_TEMPORAL_OVERSAMPLING_HPP__

#include <set>
#include <map>
#include <vector>

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>

#include "utils.hpp"
#include "generator.hpp"

class CTO {
 public:
  CTO(const std::string &t_fn, Generator *gn = new Replicator())
    : train_fn_(t_fn), train_parsed_(false), next_id_(0), generator_(gn) { }

  ~CTO() { delete generator_; }
  void oversampling(unsigned int min_docs);

 private:
  // CLASS ATTRIBUTES
  std::string train_fn_;
  bool train_parsed_;
  unsigned int next_id_;
  Generator *generator_;

  // known classes
  std::set<std::string> classes_;
  // known points in time
  std::set<unsigned int> t_pts_;
  // for each class c and point in time p,
  // store the number of documents
  std::map< std::string, std::map < unsigned int, unsigned int> > sizes_;

  // MEMBER FUNCTIONS
  void parse_train_file();

  int find_lowerbound(const std::string &c,
                      const unsigned int p,
                      const unsigned int m);
  int find_upperbound(const std::string &c,
                      const unsigned int p,
                      const unsigned int m);

  // generate n synthetic documents (that will be created at p_target),
  // based on data created at p_source assigned to class c, and store in output
//  void generate_samples(const unsigned int p_source, const unsigned int p_target,
//                        const unsigned int n, const std::string &c,
//                        std::vector<std::string> &output);
  void generate_samples(const unsigned int p_source, const unsigned int p_target,
                        const unsigned int n, const std::string &c,
                        const std::vector<std::string> &aux_data,
                        std::vector<std::string> &output, bool final=false);
};

/*
void CTO::generate_samples(const unsigned int p_source,
                           const unsigned int p_target,
                           const unsigned int n,
                           const std::string &c,
                           std::vector<std::string> &output) {
  for (unsigned int i = 0; i < n; i++) {
    std::string sample = generator_->generate_sample(next_id_++, c, p_source, p_target);
    output.push_back(sample);
  }
}
*/

void CTO::generate_samples(const unsigned int p_source,
                           const unsigned int p_target,
                           const unsigned int n,
                           const std::string &c,
                           const std::vector<std::string> &aux_data,
                           std::vector<std::string> &output, bool final) {

  std::set<std::string> all_samples;
  // first, merge auxiliary synthetic data with original training set
  Generator *new_gen = generator_->clone();
  for (unsigned int i = 0; i < aux_data.size(); i++) {
    std::string ln = aux_data[i];
    std::vector<std::string> tokens;
    Utils::string_tokenize(ln, tokens, ";");
    unsigned int tp = static_cast<unsigned int>(atoi(tokens[1].data()));
    std::string cl = tokens[2];
    new_gen->update_model(cl, tp, ln);
    if (final) all_samples.insert(ln);
  }
  if (!final) {
    std::cerr << " ** AUXILIARY UPDATING CL=" << c << " AND TP=" << p_target << " W/ " << n << " docs." << std::endl;
    for (unsigned int i = 0; i < n; i++) {
      std::string sample = new_gen->generate_sample(next_id_++, c, p_source, p_target);
      new_gen->update_model(c, p_source, sample);
      output.push_back(sample);
    }
  }
  else {
    // then, we generate the final synthetic samples
    // fill in aux_data with what is already included in syn_data
    for (unsigned int i = 0; i < output.size(); i++) {
      std::string ln = output[i];
      all_samples.insert(ln);
    }
    std::set<std::string>::const_iterator it = all_samples.begin();
    while(it != all_samples.end()) {
      std::vector<std::string> tokens;
      Utils::string_tokenize(*it, tokens, ";");
      unsigned int tp = static_cast<unsigned int>(atoi(tokens[1].data()));
      std::string cl = tokens[2];
      new_gen->update_model(cl, tp, *it);
      ++it;
    }

    std::cerr << "  > UPDATING FINAL MODEL FOR CL=" << c << " AND TP=" << p_target << " W/ " << n << " docs." << std::endl;
    for (unsigned int i = 0; i < n; i++) {
      std::string sample = new_gen->generate_sample(next_id_++, c, p_source, p_target);
      generator_->update_model(c, p_source, sample);
//      std::cout << "> " << sample << std::endl;
      output.push_back(sample);
    }
    sizes_[c][p_target] += n;
  }
  delete new_gen;
}

int CTO::find_lowerbound(const std::string &c,
                         const unsigned int p,
                         const unsigned int m) {
  std::cerr << "[LOWERBOUND c=" << c << ",p=" << p << ":";

  std::set<unsigned int> candidates;
  std::set<unsigned int>::const_iterator it = t_pts_.begin();
  while (*it < p) {
    if(sizes_[c][*it] >= m) {
      std::cerr << " " << *it;
      candidates.insert(*it);
    }
    ++it;
  }
  std::cerr << std::endl;

  if (candidates.empty()) {
    std::cerr << "[PROBLEM] NO CANDIDATES FOR cl=" << c << ", tp=" << p << " (m=" << m << ")" << std::endl;
    return -1;
  }

  unsigned int lb = *(candidates.rbegin());
  return lb; // return maximal point in time
}

int CTO::find_upperbound(const std::string &c,
                         const unsigned int p,
                         const unsigned int m) {
  std::cerr << "[UPPERBOUND c=" << c << ",p=" << p << ":";

  std::set<unsigned int> candidates;
  std::set<unsigned int>::reverse_iterator it = t_pts_.rbegin();
  while (*it > p) {
    if(sizes_[c][*it] >= m) {
      std::cerr << " " << *it;
      candidates.insert(*it);
    }
    ++it;
  }
  std::cerr << std::endl;

  if (candidates.empty()) {
    std::cerr << "[PROBLEM] NO CANDIDATES FOR cl=" << c << ", tp=" << p << " (m=" << m << ")" << std::endl;
    return -1;
  }

  unsigned int ub = *(candidates.begin());
  return ub; // return minimum point in time
}

void CTO::oversampling(unsigned int min_docs) {
  if (!train_parsed_) parse_train_file();
  if (classes_.empty() || t_pts_.empty()) return;

  std::vector<std::string> syn_data;
  std::set<std::string>::const_iterator c_it = classes_.begin();
  while (c_it != classes_.end()) {
    std::cerr << "cl = " << *c_it << std::endl;

    std::set<unsigned int>::const_iterator p_it = t_pts_.begin();
    while (p_it != t_pts_.end()) {
      std::cerr << "  tp = " << *p_it << std::endl;

      unsigned int n = sizes_[*c_it][*p_it];
      // number of synthetic documents to be generated
      unsigned int r = (n < min_docs ? min_docs - n : 0);
      std::cerr << "    n = " << n << " , r = " << r << std::endl;

      if (r > 0) { // i.e., there are not enough documents
        int l_b = find_lowerbound(*c_it, *p_it, min_docs); // -1: undefined
        int u_b = find_upperbound(*c_it, *p_it, min_docs); // -1: undefined
        std::vector<std::string> aux_data; // auxiliary synthetic data
        if (l_b < 0) {
          unsigned int n_l = min_docs - sizes_[*c_it][*(t_pts_.begin())];
          std::cerr << " [SELF_GEN] Generating " << n_l << " docs for " << *(t_pts_.begin()) << std::endl;
          generate_samples(static_cast<unsigned int>(*(t_pts_.begin())),
                           static_cast<unsigned int>(*(t_pts_.begin())),
                           n_l, *c_it, aux_data, aux_data); // fill first point in time
          l_b = *(t_pts_.begin());
        }
        if (u_b < 0) {
            unsigned int n_u = min_docs - sizes_[*c_it][*(t_pts_.rbegin())];
            std::cerr << " [SELF_GEN] Generating " << n_u << " docs for " << *(t_pts_.rbegin()) << std::endl;
            generate_samples(static_cast<unsigned int>(*(t_pts_.rbegin())),
                             static_cast<unsigned int>(*(t_pts_.begin())),
                             n_u, *c_it, aux_data, aux_data); // fill last point in time
            u_b = *(t_pts_.rbegin());
        }

        std::cerr << "[SELECTED INTERVAL] < " << l_b << " , " << *p_it << " , " << u_b << " >" << std::endl;

        while (static_cast<unsigned int>(l_b+1) < *p_it) {
          unsigned int n_p = min_docs - sizes_[*c_it][static_cast<unsigned int>(l_b)+1];
          std::cerr << "     [FROM PAST] Generating " << n_p << " docs. p_source=" << l_b << " , p_target=" << (l_b + 1) << std::endl;
          generate_samples(static_cast<unsigned int>(l_b),
                           static_cast<unsigned int>(l_b)+1,
                           n_p, *c_it, aux_data, aux_data); // fill in aux_data         
          ++l_b;
        }
        while (static_cast<unsigned int>(u_b-1) > *p_it) {
          unsigned int n_f = min_docs - sizes_[*c_it][static_cast<unsigned int>(u_b)-1];
          std::cerr << "     [FROM FUTURE] Generating " << n_f << " docs. p_source=" << u_b << " , p_target=" << (u_b - 1) << std::endl;
          generate_samples(static_cast<unsigned int>(u_b),
                           static_cast<unsigned int>(u_b)-1,
                           n_f, *c_it, aux_data, aux_data); // fill in aux_data
          --u_b;
        }

        // based on aux_data, generate synthetic data to the final training set
        // we must consider three cases:
        //  1 - If p is the first point in time, do forward cascading
        //  2 - If p is the last point in time, do backward cascading
        //  3 - Otherwise, do both forward and backward cascading
        std::cerr << "[FINAL SYNTHETIC SAMPLES] Generating " << r << " samples for " << *p_it << std::endl;
        if (*p_it != *(t_pts_.begin()) &&
            *p_it != *(t_pts_.rbegin())) { // case 3
          std::cerr << "Backward/Forward cascading" << std::endl;
          unsigned int den = ((l_b > 0 ? 1 : 0) + (u_b > 0 ? 1 : 0));
          double eff_r = static_cast<double>(r) / static_cast<double>(den);
          if (l_b > 0) {
            std::cerr << " > From " << (*p_it - 1) << " to " << *p_it << ": " << ceil(eff_r) << " examples." << std::endl;
            generate_samples(*p_it - 1, *p_it, ceil(eff_r), *c_it, aux_data, syn_data, true);
          }
          if (u_b > 0) {
            std::cerr << " > From " << (*p_it + 1) << " to " << *p_it << ": " << floor(eff_r) << " examples." << std::endl;
            generate_samples(*p_it + 1, *p_it, floor(eff_r), *c_it, aux_data, syn_data, true);
          }
          std::cerr << "     -> final_sz=" << sizes_[*c_it][*p_it] << " , added " << r << " examples." << std::endl;
//          sizes_[*c_it][*p_it] += r;
        }
        else if (*p_it == *(t_pts_.begin())) { // case 1
          std::cerr << "Forward cascading" << std::endl;
          std::cerr << " > From " << (*p_it + 1) << " to " << *p_it << ": " << r << " examples." << std::endl;
          generate_samples(*p_it + 1, *p_it, r, *c_it, aux_data, syn_data, true);
          std::cerr << "     -> final_sz=" << sizes_[*c_it][*p_it] << " , added " << r << " examples." << std::endl;
//          sizes_[*c_it][*p_it] += r;
        }
        else if (*p_it == *(t_pts_.rbegin())){ // case 2
          std::cerr << "Backward cascading" << std::endl;
          std::cerr << " > From " << (*p_it - 1) << " to " << *p_it << ": " << r << " examples." << std::endl;
          generate_samples(*p_it - 1, *p_it, r, *c_it, aux_data, syn_data, true);
          std::cerr << "     -> final_sz=" << sizes_[*c_it][*p_it] << " , added " << r << " examples." << std::endl;
//          sizes_[*c_it][*p_it] += r;
        }
        else {
          std::cerr << "ERROR: SHOULD NOT BE HERE !" << std::endl;
          exit(1);
        }
      }
      ++p_it;
    }
    ++c_it;
  }
  for (unsigned int i = 0; i < syn_data.size(); i++) {
    std::cout << syn_data[i] << std::endl;
  }
}

void CTO::parse_train_file() {
  if (train_parsed_) return;

  std::ifstream file(train_fn_.data());
  if (file) {
    std::string line("");
    while (std::getline(file, line)) {
      std::vector<std::string> tokens;
      Utils::string_tokenize(line, tokens, ";");
      unsigned int id = static_cast<unsigned int>(atoi(tokens[0].data()));
      if (next_id_ <= id) next_id_ = id;
      unsigned int tp = static_cast<unsigned int>(atoi(tokens[1].data()));
      std::string cl = tokens[2];
     
      sizes_[cl][tp]++;
      t_pts_.insert(tp);
      classes_.insert(cl);

      generator_->update_model(cl, tp, line);
    }
    line.clear();
    file.close();
    next_id_++;
    std::cerr << "TRAINING SUMMARY:" << std::endl;
    std::cerr << "  Num_cl: " << classes_.size() << std::endl;
    std::cerr << "  Num_tp: " << t_pts_.size() << std::endl;
    std::cerr << "  Nxt_id: " << next_id_ << std::endl;
    train_parsed_ = true;
  }
  else {
    std::cerr << "Error while opening training file." << std::endl;
    exit(1);
  }
}

#endif
