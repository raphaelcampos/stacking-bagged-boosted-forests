#ifndef TCPP_H__
#define TCPP_H__

#include "supervised_classifier.hpp"

#define PNAIVEBAYES_STR "pnaivebayes"
#define LNAIVEBAYES_STR "lnaivebayes"
#define CNAIVEBAYES_STR "cnaivebayes"
#define GNAIVEBAYES_STR "gnaivebayes"
#define IITI_STR "iiti"
#define BITI_STR "biti"
#define RF_STR "randomforest"
#define RF2_STR "rf"
#define RFKNN_STR "rf-knn"
#define RFKRAND_STR "rf-krand"
#define BRF_STR "rf-boosted"
#define DT_STR "decisiontree"
#define KNN_STR "knn"
#define ROCCHIO_STR "rocchio"
#define ONSCO_STR "onsco"

#define D_COS_STR "cosine"
#define D_L2_STR "l2"
#define D_L1_STR "l1"

enum method_t {
  ROCCHIO=0,
  KNN,
  NB_PROB,
  NB_LOG,
  NB_COMPL,
  NB_GAUSS,
  ONSCO,
  IITI,
  BITI,
  RF1,
  DTree,
  RF2,
  RFKNN,
  RFKRAND,
  BRF,
  UNKNOWN
};

enum fs_t {
  BNS=0,
  ODDS,
  IG,
  CC,
  NONE
};

struct params_t {
  unsigned int round;
  method_t method;
  bool raw;

  DistType dt;

  fs_t fs;     // feature selector
  double fs_p; // percentage of vocabulary

  char *train_file;
  char *test_file;
  char *out_file;

  bool validation; // cross_fold validation
  unsigned int replications;  // number of replications

  unsigned int k;  // knn

  bool temporal;
  char *twf_file;
  double beta;
  double alpha;
  double lambda;
  bool unif_prior;

  double rf_m;
  int rf_trees;
  double rf_docs;
  int rf_height;
  bool rf_trn;

  bool two_pass;
  method_t fst_pass_cl;
  method_t snd_pass_cl;

  bool onsco;
  bool ext_onsco;

  bool window;
  bool grad_window;
  unsigned int w_size;
};

void init_params(params_t &p) {
  p.method = ROCCHIO;
  p.raw = false;
  p.dt = COSINE;
  p.fs = NONE;
  p.fs_p = 1.0;
  p.k = 30;
  p.train_file = NULL;
  p.test_file = NULL;
  p.out_file = NULL;
  p.twf_file = NULL;
  p.temporal = false;
  p.beta = 1.0;
  p.alpha = 1.0;
  p.lambda = 0.0;
  p.rf_m = 1.0;
  p.rf_trees = 10;
  p.rf_docs = 1.0;
  p.rf_height = 0;
  p.rf_trn=false;
  p.unif_prior = false;
  p.round = 0;
  p.validation = false;
  p.replications = 0;
  p.two_pass = false;
  p.fst_pass_cl = ROCCHIO;
  p.snd_pass_cl = ROCCHIO;
  p.onsco = false;
  p.ext_onsco=false;
  p.window = false;
  p.grad_window = false;
  p.w_size = 0;
}

void print_usage(const char *msg = NULL);

method_t parse_method(const char* m);
fs_t parse_fs (const unsigned int t);

TemporalClassifier *get_temporal_classifier(const method_t &m, params_t &p);
SupervisedClassifier *get_classifier(params_t &p);


bool run_classifier(params_t &p);
bool run_validation(params_t &p);


#endif
