#include <map>
#include <limits>
#include <algorithm>

#include "inc_nb.hpp"
#include "functs.hpp"
#include "stats.hpp"

#define BUFFER_SZ 100
#define ALPHA    0.05

long double RationalApproximation(long double t) {
  long double c[] = {2.515517, 0.802853, 0.010328};
  long double d[] = {1.432788, 0.189269, 0.001308};
  return t - ((c[2]*t + c[1])*t + c[0]) /
             (((d[2]*t + d[1])*t + d[0])*t + 1.0);
}

long double NormalCDFInverse(double p) {
  if (p >= 1.0) p = 1.0; // precision adjustment

  if (p <= 0.0) {
    std::cerr << "Invalid input argument (" << p
              << "); must be larger than 0 but less than 1." << std::endl;
    exit(1);
  }
  if (p < 0.5) return -RationalApproximation( sqrt(-2.0*log(p)) );
  else return RationalApproximation( sqrt(-2.0*log(1-p)) );

  return 0.0; // never reaches here!
}

long long unsigned int get_value(const std::map<std::string, long long unsigned int> &h,
                                 const std::string &k) {
  std::map<std::string, long long unsigned int>::const_iterator it = h.find(k);
  if (it != h.end()) return it->second;
  return 0;
}

class TransferNB {
 public:
  TransferNB(const char *fn, long double alpha=0.05, unsigned int k=10)
    : init_(false), alpha_(alpha), total_docs_(0), k_(k) { srand(0); train(fn); }
  TransferNB(long double alpha=0.05, unsigned int k=10)
    : init_(false), alpha_(alpha), total_docs_(0), k_(k) { srand(0); }

  void train(const char *fn);
  void test(const char *fn);

 private:
  typedef std::map< std::string, std::map< std::string, long double > > Matrix;

  bool init_;
  long double alpha_;
  std::map<std::string, IncNB> classifiers_;
  std::map<std::string, std::set<std::string> > vocab_years_;
  std::map<std::string, std::set<std::string> > class_years_;

  std::set<std::string> classes_;
  std::set<std::string> years_;
  std::set<std::string> vocabulary_;
  std::map<std::string, unsigned long long int> TF_;
  std::map<std::string, unsigned long long int> sum_TF_p_;
  std::map<std::string, unsigned long long int> sum_TF_;
  std::map<std::string, unsigned long long int> DF_;
  std::map<std::string, unsigned long long int> DF_tc_;
  std::map<std::string, unsigned long long int> DF_t_;
  std::map<std::string, unsigned long long int> sum_DF_;

  Matrix KLm_;
  std::map< std::string, std::set<Term> > top_terms_;
  std::map<std::string, unsigned long long int> sum_TF_top_terms_;

  unsigned int total_docs_;
  unsigned int k_;
  void build_KLm();
  void update_top_terms();
  void update_top_terms_bns(const std::string &t, const std::string &tp);
  void update_top_terms_ig(const std::string &t, const std::string &tp);
  void update_top_terms_mi(const std::string &t, const std::string &tp);
  void push_top_k_term(const Term &t, const std::string &tp);

  void update_model(const std::string &ln);
  void update_model(const std::vector<std::string> &tks, bool upd_cl=true);

  Output classify(const std::string &ln, std::map<std::string, Output> &ret);
  Output classify(const std::vector<std::string> &tks, std::map<std::string, Output> &ret);
  void process_batch(const std::map< std::string,
                     std::map< std::string, std::vector< std::vector<std::string> > > > &to_update);
  void transfer_instances(const char *fn);
  void transfer_instances(const std::string &tp_tgt);
  void do_transfer(const std::string &tp, const std::vector<std::string> &tks);

  long double term_conditional(const std::string &t, const std::string &cl, const std::string &tp,
                               const std::set<std::string> &voc, const long long unsigned int v_sz);
  long double apriori(const std::string &cl, const std::string &tp, const long long int c_sz);

  void print_matrix() {
    Matrix::const_iterator it_i = KLm_.begin();
    while (it_i != KLm_.end()) {
      std::cerr << it_i->first;
      std::map<std::string, long double >::const_iterator it_j = it_i->second.begin();
      while (it_j != it_i->second.end()) {
        std::cerr << " " << it_j->second;
        ++it_j;
      }
      std::cerr << std::endl;
      ++it_i;
    }
  }
};

long double TransferNB::term_conditional(const std::string &t, const std::string &cl, const std::string &tp,
                                         const std::set<std::string> &voc, const long long unsigned int v_sz) {
  std::string idx_tc  = get_idx(t, cl);
  std::string idx_tcp = get_idx(idx_tc, tp);
  long double condNum = static_cast<long double>(get_value(TF_, idx_tcp) + ALPHA);

/*
  long double sum_tf = 0.0;
  std::set<std::string>::const_iterator vi = voc.begin();
  while (vi != voc.end()) {
    idx_tcp = get_idx(get_idx(*vi, cl), tp);
    sum_tf += TF_[idx_tcp];
    ++vi;
  }
  long double condDen = sum_tf + (ALPHA*static_cast<long double>(v_sz));
*/

  long double condDen = static_cast<long double>(get_value(sum_TF_p_, tp) + (ALPHA*static_cast<long double>(v_sz)));
  return (condDen == 0.0) ? 0.0 : (condNum / condDen);
}

long double TransferNB::apriori(const std::string &cl, const std::string &tp, const long long int c_sz) {
  long double aPrioriNum = static_cast<long double>(get_value(DF_, get_idx(cl, tp)) + ALPHA);
  long double aPrioriDen = static_cast<long double>(get_value(sum_DF_, tp) + ALPHA*static_cast<long double>(c_sz));
  return (aPrioriDen == 0.0) ? 0.0 : (aPrioriNum / aPrioriDen);
}

/*
void TransferNB::build_KLm() {
  update_top_terms();

  std::map<std::string, long double> min;
  std::map<std::string, long long unsigned int>::const_iterator it_i = sum_DF_.begin();
  while (it_i != sum_DF_.end()) {
    std::string tp_i = it_i->first;
    // vocabulary for tp_i:
    unsigned long long int sum_tf_i = 0;
    std::set<std::string> voc_i;
    std::set<Term>::const_iterator vi = top_terms_[tp_i].begin();
    while (vi != top_terms_[tp_i].end()) {
      voc_i.insert(vi->term);
      ++vi;
    }
    KLm_[tp_i][tp_i] = 1.0;
    std::map<std::string, long long unsigned int>::const_iterator it_j = it_i; ++it_j;
    while (it_j != sum_DF_.end()) {
      std::string tp_j = it_j->first;

      // vocabulary for tp_i:
      std::set<std::string> voc_j;
      std::set<Term>::const_iterator vj = top_terms_[tp_j].begin();
      while (vj != top_terms_[tp_j].end()) {
        voc_j.insert(vj->term);
        ++vj;
      }
      std::set<std::string> vocabulary;
      std::set_union(voc_i.begin(), voc_i.end(), voc_j.begin(), voc_j.end(),
                     std::insert_iterator< std::set<std::string> >(vocabulary, vocabulary.begin()));

      long double KLij = 0.0;
      long double KLji = 0.0;
      std::set<std::string>::const_iterator v_it = vocabulary.begin();
      while (v_it != vocabulary.end()) {
        std::set<std::string>::const_iterator c_it = classes_.begin();
        while (c_it != classes_.end()) {
          // P(w|p) = \sum_{c} P(w|c,p) * P(c|p)
          long double cond_i = term_conditional(*v_it, *c_it, tp_i, vocabulary, vocabulary_.size());
          long double cond_j = term_conditional(*v_it, *c_it, tp_j, vocabulary, vocabulary_.size());
          long double p_i = cond_i * apriori(*c_it, tp_i, classes_.size());
          long double p_j = cond_j * apriori(*c_it, tp_j, classes_.size());
          long double parcel = ( p_i == 0.0 || p_j == 0.0) ? 0.0 : p_i * log(p_i / p_j);

          KLij += parcel;

          parcel = ( p_i == 0.0 || p_j == 0.0) ? 0.0 : p_j * log(p_j / p_i);
          KLji += parcel;

          ++c_it;
        }
        ++v_it;
      }

      long double val_ij = 1.0 - KLij;
      long double val_ji = 1.0 - KLji;
      KLm_[tp_i][tp_j] = val_ij;
      KLm_[tp_j][tp_i] = val_ji;

      if (min.find(tp_i) == min.end() || min[tp_i] > val_ij) min[tp_i] = val_ij;
      if (min.find(tp_j) == min.end() || min[tp_j] > val_ji) min[tp_j] = val_ji;

      ++it_j;
    }

    ++it_i;
  }

  print_matrix();
  exit(0);
}

*/

void TransferNB::build_KLm() {
  update_top_terms();

  std::map<std::string, long double> min;
  std::map<std::string, long long unsigned int>::const_iterator it_i = sum_DF_.begin();
  while (it_i != sum_DF_.end()) {
    std::string tp_i = it_i->first;

    // vocabulary for tp_i:
    unsigned long long int sum_tf_i = 0;
    std::set<std::string> voc_i;
    std::set<Term>::const_iterator vi = top_terms_[tp_i].begin();
    while (voc_i.size() < k_ && vi != top_terms_[tp_i].end()) {
      voc_i.insert(vi->term);
      ++vi;
    }

    KLm_[tp_i][tp_i] = 1.0;
    std::map<std::string, long long unsigned int>::const_iterator it_j = it_i; ++it_j;
    while (it_j != sum_DF_.end()) {
      std::string tp_j = it_j->first;
/*
      // vocabulary for tp_i:
      std::set<std::string> voc_j;
      std::set<Term>::const_iterator vj = top_terms_[tp_j].begin();
      while (vj != top_terms_[tp_j].end()) {
        voc_j.insert(vj->term);
        ++vj;
      }
*/

      // vocabulary for tp_j:
      unsigned long long int sum_tf_j = 0;
      std::set<std::string> voc_j;
      std::set<Term>::const_iterator vj = top_terms_[tp_j].begin();
      while (voc_j.size() < k_ && vj != top_terms_[tp_j].end()) {
        voc_j.insert(vj->term);
        ++vj;
      }

      std::set<std::string> vocabulary;
      std::set_union(voc_i.begin(), voc_i.end(), voc_j.begin(), voc_j.end(),
                     std::insert_iterator< std::set<std::string> >(vocabulary, vocabulary.begin()));

      long double KLij = 0.0, KL_prior_ij = 0.0;
      long double KLji = 0.0, KL_prior_ji = 0.0;

      std::set<std::string>::const_iterator c_it = classes_.begin();
      while (c_it != classes_.end()) {
        long double pc_i = apriori(*c_it, tp_i, classes_.size());
        long double pc_j = apriori(*c_it, tp_j, classes_.size());

        KL_prior_ij += /*( pc_i == 0.0 || pc_j == 0.0) ? 0.0 : */pc_i * log(pc_i / pc_j);
        KL_prior_ji += /*( pc_i == 0.0 || pc_j == 0.0) ? 0.0 : */pc_j * log(pc_j / pc_i);

        long double parcel_i = 0.0, parcel_j = 0.0; 
        std::set<std::string>::const_iterator v_it = vocabulary.begin();
        while (v_it != vocabulary.end()) {
          // P(w|p) = \sum_{c} P(w|c,p) * P(c|p)
          long double cond_i = term_conditional(*v_it, *c_it, tp_i, vocabulary, vocabulary.size());
          long double cond_j = term_conditional(*v_it, *c_it, tp_j, vocabulary, vocabulary.size());

          parcel_i += /*( cond_i == 0.0 || cond_j == 0.0) ? 0.0 : */cond_i * log(cond_i / cond_j);
          parcel_j += /*( cond_i == 0.0 || cond_j == 0.0) ? 0.0 : */cond_j * log(cond_j / cond_i);

          ++v_it;
        }

        KLij += pc_i * parcel_i;
        KLji += pc_j * parcel_j;
        ++c_it;
      } 

//      std::cerr << "Sum[" << tp_i << "," << tp_j << "]=" << (KL_prior_ij + KLij) << "     Sum[" << tp_j << "," << tp_i << "]=" << (KL_prior_ji + KLji) << std::endl;

      long double val_ij = 1.0 - (KL_prior_ij + KLij);
      long double val_ji = 1.0 - (KL_prior_ji + KLji);
      KLm_[tp_i][tp_j] = val_ij;
      KLm_[tp_j][tp_i] = val_ji;

      if (min.find(tp_i) == min.end() || min[tp_i] > val_ij) min[tp_i] = val_ij;
      if (min.find(tp_j) == min.end() || min[tp_j] > val_ji) min[tp_j] = val_ji;

      ++it_j;
    }

    ++it_i;
  }

/*
  it_i = sum_DF_.begin();
  while (it_i != sum_DF_.end()) {
    std::string tp_i = it_i->first;
    // range normalization -> [0,1]
    std::map<std::string, long long unsigned int>::const_iterator it_j = sum_DF_.begin();
    while (it_j != sum_DF_.end()) {
      std::string tp_j = it_j->first;
      long double val_ij = KLm_[tp_i][tp_j];
      KLm_[tp_i][tp_j] = (val_ij - min[tp_i]) / (1.0 - min[tp_i]);
      ++it_j;
    }
    ++it_i;
  }
*/
  print_matrix();
}

void TransferNB::push_top_k_term(const Term &term, const std::string &tp) {
//  if ( false && top_terms_[tp].size() == k_ ) {
//    Term cand = *(top_terms_[tp].begin()); // select smallest Term
//    if (cand.weight < term.weight) {
//      top_terms_[tp].erase(cand);
//      top_terms_[tp].insert(term);
//    }
//  }
//  else {
    top_terms_[tp].insert(term);
//  }
}

void TransferNB::update_top_terms_mi(const std::string &term, const std::string &tp) {
  long double mi = 0.0;
  std::set<std::string>::iterator c_it = classes_.begin();
  while (c_it != classes_.end()) {
    std::string cl = *c_it;
    std::string idx_tp  = get_idx(term, tp);
    std::string idx_tcp = get_idx(get_idx(term, cl), tp);
    long double n = static_cast<long double>(get_value(sum_DF_, tp)) + 4.0;
    long double n_c = static_cast<long double>(get_value(DF_, get_idx(cl, tp))) + 2.0;
    long double n_k = static_cast<long double>(get_value(DF_t_, idx_tp) + 2.0);
    long double n_kc = static_cast<long double>(get_value(DF_tc_, idx_tcp) + 1.0);

    long double p_c  = n_c / n;
    long double p_nc = (n - n_c) / n;

    long double p_k  = n_k / n;
    long double p_nk = (n - n_k) / n;

    long double p_kc   = n_kc / n;
    long double p_nkc  = (n_c - n_kc) / n;
    long double p_knc  = (n_k - n_kc) / n;
    long double p_nknc = (n - n_k - n_c + n_kc) / n;

    if (p_k == 0.0 || p_c == 0.0 || p_kc == 0.0 || p_nkc == 0.0 || p_knc == 0.0 || p_nknc == 0.0) {
      std::cerr << "Term  = " << term << std::endl;
      std::cerr << "Class = " << term << std::endl << std::endl;

      std::cerr << "P(k)     = " << p_k    << std::endl;
      std::cerr << "P(k')    = " << p_nk   << std::endl;
      std::cerr << "P(c)     = " << p_c    << std::endl;
      std::cerr << "P(c')    = " << p_nc   << std::endl;
      std::cerr << "P(k,c)   = " << p_kc   << std::endl;
      std::cerr << "P(k',c)  = " << p_nkc  << std::endl;
      std::cerr << "P(k,c')  = " << p_knc  << std::endl;
      std::cerr << "P(k',c') = " << "(" << n << " - " << n_k << " - " << n_c << " + " << n_kc << ") / " << n << " = " << p_nknc << std::endl << std::endl;

      exit(1);
    }

    mi += p_kc   * log((p_kc)   / (p_k * p_c))   +
          p_nkc  * log((p_nkc)  / (p_nk * p_c))  +
          p_knc  * log((p_knc)  / (p_k * p_nc))  +
          p_nknc * log((p_nknc) / (p_nk * p_nc));
    ++c_it;
  }
  push_top_k_term(Term(term, mi), tp);
}

void TransferNB::update_top_terms_ig(const std::string &term, const std::string &tp) {
  double parc1 = 0.0, parc2 = 0.0, parc3 = 0.0;
  std::set<std::string>::iterator c_it = classes_.begin();
  while (c_it != classes_.end()) {
    std::string cl = *c_it;
    std::string idx_tp  = get_idx(term, tp);
    std::string idx_tcp = get_idx(get_idx(term, cl), tp);
    double n = static_cast<double>(get_value(sum_DF_, tp)) + 4.0;
    double n_c = static_cast<double>(get_value(DF_, get_idx(cl, tp))) + 2.0;
    double n_k = static_cast<double>(get_value(DF_t_, idx_tp) + 2.0);
    double n_kc = static_cast<double>(get_value(DF_tc_, idx_tcp) + 1.0);

    double p_c  = n_c / n;
    double p_nc = (n - n_c) / n;

    double p_k  = n_k / n;
    double p_nk = (n - n_k) / n;

    double p_kc   = n_kc / n;
    double p_nkc  = (n_c - n_kc) / n;

    if (p_k == 0.0 || p_c == 0.0 || p_kc == 0.0 || p_nkc == 0.0) {
      std::cerr << "Term  = " << term << std::endl;
      std::cerr << "Class = " << term << std::endl << std::endl;

      std::cerr << "P(k)     = " << p_k    << std::endl;
      std::cerr << "P(c)     = " << p_c    << std::endl;
      std::cerr << "P(k')    = " << p_nk   << std::endl;
      std::cerr << "P(c')    = " << p_nc   << std::endl;
      std::cerr << "P(k,c)   = " << p_kc   << std::endl;
      std::cerr << "P(k',c)  = " << "(" << n_c << " - " << n_kc << ") / " << n << "=" << p_nkc  << std::endl;

      exit(1);
    }

    parc1 += p_c  * log(p_c);
    parc2 += p_kc * log(p_kc / p_k);
    parc3 += p_nkc * log(p_nkc / p_nk);

    ++c_it;
  }

  double ig = - parc1 + parc2 + parc3;
  push_top_k_term(Term(term, ig), tp);
}

void TransferNB::update_top_terms_bns(const std::string &term, const std::string &tp) {
  double max_bns = std::numeric_limits<double>::min();
  std::set<std::string>::iterator c_it = classes_.begin();
  while (c_it != classes_.end()) {
    std::string cl = *c_it;
    std::string idx_tp  = get_idx(term, tp);
    std::string idx_tcp = get_idx(get_idx(term, cl), tp);
    double n = static_cast<double>(get_value(sum_DF_, tp)) + 4.0;
    double n_k = static_cast<double>(get_value(DF_t_, idx_tp) + 2.0);
    double n_kc = static_cast<double>(get_value(DF_tc_, idx_tcp) + 1.0);

    double p_kc   = n_kc / n;
    double p_knc  = (n_k - n_kc) / n;

    if (p_kc == 0.0 || p_knc == 0.0) {
      std::cerr << "Term  = " << term << std::endl;
      std::cerr << "Class = " <<  cl  << std::endl << std::endl;
      std::cerr << "Time  = " <<  tp  << std::endl << std::endl;
      std::cerr << "P(k,c)   = "      << p_kc   << std::endl;
      std::cerr << "P(k,c')  = "      << p_knc  << std::endl;
      exit(1);
    }

    double b = fabs(NormalCDFInverse(p_kc) - NormalCDFInverse(p_knc));
    if (max_bns < b) max_bns = b;
    ++c_it;
  }

  push_top_k_term(Term(term, max_bns), tp);
}

void TransferNB::update_top_terms() { 
  std::map<std::string, std::set<std::string> >::const_iterator y_it = vocab_years_.begin();
  while (y_it != vocab_years_.end()) {
    std::set<std::string>::const_iterator v_it = y_it->second.begin();
    while (v_it != y_it->second.end()) {
//      update_top_terms_bns(*v_it, y_it->first);
//      update_top_terms_ig(*v_it, y_it->first);
      update_top_terms_mi(*v_it, y_it->first);
      ++v_it;
    }
    ++y_it;
  }
}

void TransferNB::update_model(const std::vector<std::string> &tokens, bool upd_cl) {
  std::string tp = tokens[1];
  std::map<std::string, IncNB>::const_iterator it = classifiers_.find(tp);
  if (it == classifiers_.end()) classifiers_[tp] = IncNB(alpha_);
  if (upd_cl) classifiers_[tp].update_model(tokens);

  std::string id = tokens[0];
  std::string cl = tokens[2];
  cl.replace(0, 6, "");
  classes_.insert(cl);
  class_years_[tp].insert(cl);
  years_.insert(tp);
  sum_DF_[tp]++;
  DF_[get_idx(cl, tp)]++;

  for (unsigned int i = 3; i < tokens.size()-1; i+=2) {
    std::string term = tokens[i];
    long long unsigned int tf = atoi(tokens[i+1].c_str());
    vocabulary_.insert(term);
    std::string idx_tc  = get_idx(term, cl);
    std::string idx_tcp = get_idx(idx_tc, tp);
    std::string idx_tp  = get_idx(term, tp);
    TF_[idx_tcp] += tf;
    DF_tc_[idx_tcp]++;
    DF_t_[idx_tp]++;
    sum_TF_[get_idx(cl, tp)] += tf;
    sum_TF_p_[tp] += tf;
    vocab_years_[tp].insert(term);
  }
  total_docs_++;
}

void TransferNB::update_model(const std::string &line) {
  std::vector<std::string> tokens;
  stringTokenize(line, tokens, ";");
  if (tokens.size() > 3) {
    update_model(tokens);
  }
}

void TransferNB::do_transfer(const std::string &tp_src, const std::vector<std::string> &tokens) {
  std::map<std::string, IncNB>::iterator it = classifiers_.begin();
  while (it != classifiers_.end()) {
    std::string tp_tgt = it->first;
    if (tp_src != tp_tgt) {
      long double kl= KLm_[tp_src][tp_tgt];
      long double p = 0.8; // static_cast<long double>(rand())/static_cast<long double>(RAND_MAX);
      if (kl > p) {
        std::vector<std::string> tks(tokens);
        std::string cl = tks[2]; cl.replace(0, 6, "");
        long double pc = it->second.get_prior(cl);
        p = static_cast<long double>(rand())/static_cast<long double>(RAND_MAX);
        if (p > pc) {
          tks[1] = tp_tgt;
          it->second.update_model(tks);
        }
      }
    }
    ++it;
  }
}

void TransferNB::transfer_instances(const char *fn) {
    std::map<std::string, IncNB>::iterator it = classifiers_.begin();
    while (it != classifiers_.end()) {
      std::cerr << "=========== MODEL FOR " << it->first << " ===========" << std::endl;
      it->second.print_model_summary();
      ++it;
    }


  std::cerr << "[TRANSFERRING INSTANCES] total of " << classifiers_.size() << " classifiers." << std::endl;
  std::ifstream file(fn);
  if (file) {
    std::string line;
    while (std::getline(file, line)) {
      std::vector<std::string> tokens;
      stringTokenize(line, tokens, ";");
      if (tokens.size() > 3) {
        std::string tp_src = tokens[1];
        std::string real_cl = tokens[2]; real_cl.replace(0, 6, "");
        // query KLm and see which time points are similar to tp
        // scan line [tp] of KLm e look after the ones >= \delta
        do_transfer(tp_src, tokens);
      }
    }
    file.close();
    std::map<std::string, IncNB>::iterator it = classifiers_.begin();
    while (it != classifiers_.end()) {
      std::cerr << "=========== MODEL FOR " << it->first << " ===========" << std::endl;
      it->second.print_model_summary();
      ++it;
    }
  }
  else {
    std::cout << "Error while opening training file." << std::endl;
    exit(1);
  }
}

void TransferNB::train(const char *fn) {
  if (init_) return;

  std::ifstream file(fn);
  if (file) {
    std::string line;
    while (std::getline(file, line)) {
      update_model(line);
    }
    file.close();
    build_KLm();
    transfer_instances(fn);
    init_ = true;
  }
  else {
    std::cout << "Error while opening training file." << std::endl;
    exit(1);
  }
}

void TransferNB::test(const char *fn) {
  if (!init_) panic("Where is the training file ?");
  std::ifstream file(fn);
  if (file) {
    unsigned int n = 0;
    std::map< std::string, std::vector< std::vector<std::string> > > buffer;
    std::string line;
    while (std::getline(file, line)) {
      std::vector<std::string> tokens;
      stringTokenize(line, tokens, ";");
      if (tokens.size() > 3) {
        std::map<std::string, Output> outs;
        Output res = classify(tokens, outs);
        std::string pred_cl = res.scores.top().class_name;
        double max_prob = res.scores.top().score / res.norm;

        // TODO INCLUIR LOGICA DE ABSTENCAO !!!

       // let us evaluate if we should update classifier for tokens[1]:
        if (res.uncertainty > 0.9 && max_prob > 0.9) {
          std::vector<std::string> tks(tokens);
          std::string tp = tks[1];
          tks[2] = std::string("CLASS=") + pred_cl;
          std::map<std::string, IncNB>::iterator it = classifiers_.find(tp);
          if (it == classifiers_.end()) {
            std::map<std::string, std::vector< std::vector<std::string> > >::iterator b_it = buffer.find(tp);
            if (b_it == buffer.end()) {
              std::cerr << "Creating buffer for " << tp << std::endl;
              buffer.insert(std::make_pair(tp, std::vector< std::vector<std::string> >()));
              std::cerr << " OK. Including vector." << std::endl;
              buffer[tp].push_back(tokens);
              std::cerr << "  DONE." << std::endl;
            }
            else {
              if (b_it->second.size() < BUFFER_SZ) {
                std::cerr << "Inserting instance in an existing buffer: " << tp << " (current bf_sz=" << buffer[tp].size() << std::endl;
                buffer[tp].push_back(tokens);
                std::cerr << " DONE." << std::endl;    
              }
              else {
                std::cerr << "> Spawning a new classifier for " << tp << std::endl;
                classifiers_[tp] = IncNB(alpha_);
                for (unsigned int i = 0; i < buffer[tp].size(); i++) {
                  classifiers_[tp].update_model(buffer[tp][i]);
                  update_model(buffer[tp][i], false);
                }
                classifiers_[tp].update_model(tks);
                update_model(tks, false);
                classifiers_[tp].print_model_summary();
                build_KLm();
                for (unsigned int i = 0; i < buffer[tp].size(); i++) {
                  // scan all target points in time and consider the KLM_[tp][target]
                  do_transfer(tp, buffer[tp][i]);
                }
                buffer.erase(tp);
                print_matrix();
              }
            }
          }
          else {
            std::cerr << " > Including doc from " << tks[1] << " agreement=" << res.uncertainty << " predicted=" << tks[2] << " correct=" << tokens[2] << " sco=" << max_prob << std::endl;
            long double pc = classifiers_[tp].get_prior(pred_cl);
            long double p = static_cast<long double>(rand())/static_cast<long double>(RAND_MAX);
            if (p > pc) classifiers_[tp].update_model(tks);
            update_model(tks, false);
          }
        }

        std::cout << res.id << ":" << res.uncertainty << ":" << tokens[1] << " CLASS=" << res.real_class << " CLASS=" << pred_cl << ":" << max_prob << std::endl;
        n++;
      }
    }
    file.close();
    print_matrix();
    std::map<std::string, IncNB>::iterator it = classifiers_.begin();
    while (it != classifiers_.end()) {
      std::cerr << "=========== MODEL FOR " << it->first << " ===========" << std::endl;
      it->second.print_model_summary();
      ++it;
    }
  }
  else {
    std::cout << "Error while opening training file." << std::endl;
    exit(1);
  }
}

Output TransferNB::classify(const std::vector<std::string> &tokens, std::map<std::string, Output> &outs) {
  std::string id = tokens[0];
  std::string tp = tokens[1];

  std::map<std::string, long double> w_votes;
  std::map<std::string, long double> votes;
  unsigned int total_votes = 0;

  long double p = 0.6;//static_cast<long double>(rand())/static_cast<long double>(RAND_MAX);
//  std::cerr << "[classify] id=" << id << std::endl;
  std::map<std::string, IncNB>::const_iterator it = classifiers_.begin();
  while (it != classifiers_.end()) { 
    long double kl = (KLm_.find(tp) == KLm_.end()) ? 1.0 : (KLm_[tp][it->first]);
    if (kl > p) {
      std::cerr << "Classifying doc from " << tp << " " << ((KLm_.find(tp) == KLm_.end()) ? "<ALL>": "<KLM>")<< std::endl;
      Output res = it->second.classify(tokens);
      std::cerr << "  DONE." << std::endl;
      outs[it->first] = res;
      Score sco = res.scores.top();
//      std::cerr << " tp=" << it->first << " norm=" << res.norm << ": " << std::endl;
      while (!res.scores.empty()) {
        Score sco = res.scores.top();
        long double w_sco = (sco.score/res.norm) * ((KLm_[it->first].find(tp) == KLm_[it->first].end()) ? 1.0 : (KLm_[it->first][tp]));
//        std::cerr << "  ++ sco["<< sco.class_name << "]=" << (sco.score/res.norm) << std::endl;
        w_votes[sco.class_name] += w_sco;
        res.scores.pop();
      }
      votes[sco.class_name]++;
      total_votes++;
    }
    ++it;
  }

  std::string pred_cl;
  long double max_w_vote = 0.0;
  std::map<std::string, long double>::const_iterator v_it = w_votes.begin();
  while (v_it != w_votes.end()) {
    if (max_w_vote < v_it->second) {
      max_w_vote = v_it->second;
      pred_cl = v_it->first;
    }
    ++v_it;
  }
  long double agreement_rate = static_cast<long double>(votes[pred_cl]) / static_cast<long double>(total_votes);

  std::string real_cl = tokens[2]; real_cl.replace(0, 6, "");

  Output out(id, tp, real_cl, agreement_rate);

  std::map<std::string, long double>::const_iterator s_it = w_votes.begin();
  while (s_it != w_votes.end()) {
    double sc = s_it->second; // * (static_cast<long double>(votes[s_it->first]) / static_cast<long double>(classifiers_.size()));
    out.push_score(Score(s_it->first, sc));
    out.norm += sc;
    ++s_it;
  }

  return out;
}

Output TransferNB::classify(const std::string &line, std::map<std::string, Output> &outs) {
  if (!init_) panic("Where is the training file ?");

  std::vector<std::string> tokens;
  stringTokenize(line, tokens, ";");
  if (tokens.size() > 3) {
    return classify(tokens, outs);
  }
  const std::string error("ERROR");
  Output err(error, error, error, 1.0);
  err.push_score(Score(error, 0.0));
  return err;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "USAGE: " << argv[0] << " [TRAIN] [TEST]" << std::endl;
    return 1;
  }

  TransferNB nb(1.0);
  nb.train(argv[1]);
  nb.test(argv[2]);
}
