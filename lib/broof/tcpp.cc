#include <string>
#include <cstdio>
#include <iostream>

#include "nb.hpp"
#include "nb_log.hpp"
#include "nb_compl.hpp"
#include "nb_gaussian.hpp"

#include "knn.hpp"

#include "rocchio.hpp"

#include "dt.hpp"

#include "rf.hpp"
#include "rf_knn.hpp"
#include "rf_kr.hpp"
#include "rf_bst.hpp"

#include "temporal_rocchio.hpp"
#include "temporal_nb.hpp"
#include "temporal_nb_log.hpp"
#include "temporal_knn.hpp"

#include "iti_incremental.hpp"
#include "iti_batch.hpp"
#include "random_forest.hpp"

#include "two_pass_classifier.hpp"
#include "onscores_classifier.hpp"

#include "ext_onscores_classifier.hpp"

#include "fs_bns.hpp"
#include "fs_or.hpp"

#include "kfold.hpp"

#include "tcpp.h"

method_t parse_method(const char* m) {
  if (strcmp(m, PNAIVEBAYES_STR) == 0) return NB_PROB;
  else if (strcmp(m, LNAIVEBAYES_STR) == 0) return NB_LOG;
  else if (strcmp(m, CNAIVEBAYES_STR) == 0) return NB_COMPL;
  else if (strcmp(m, GNAIVEBAYES_STR) == 0) return NB_GAUSS;
  else if (strcmp(m, KNN_STR) == 0) return KNN;
  else if (strcmp(m, ROCCHIO_STR) == 0) return ROCCHIO;
  else if (strcmp(m, ONSCO_STR) == 0) return ONSCO;
  //else if (strcmp(m, IITI_STR) == 0) return IITI;
  else if (strcmp(m, BITI_STR) == 0) return BITI;
  else if (strcmp(m, RF_STR) == 0) return RF1;
  else if (strcmp(m, DT_STR) == 0) return DTree;
  else if (strcmp(m, RF2_STR) == 0) return RF2;
  else if (strcmp(m, RFKNN_STR) == 0) return RFKNN;
  else if (strcmp(m, RFKRAND_STR) == 0) return RFKRAND;
  else if (strcmp(m, BRF_STR) == 0) return BRF;
  else {
    std::stringstream msg; msg << "Invalid method name (" << m << ").";
    print_usage(msg.str().data());
    return UNKNOWN;
  }
}

fs_t parse_fs (const unsigned int t) {
  switch(t) {
    case 0: return BNS;
    case 1: return ODDS;
    case 2: return IG;
    case 3: return CC;
    default: return NONE;
  }
}

DistType parse_dist (const char *d) {
  if (strcmp(d, D_COS_STR) == 0) return COSINE;
  else if (strcmp(d, D_L2_STR) == 0) return L2;
  else if (strcmp(d, D_L1_STR) == 0) return L1;
  else return COSINE;
}

void print_usage(const char *err_msg) {
  if (err_msg) std::cerr << "ERROR: " << err_msg << std::endl << std::endl;

  std::cerr << "USAGE: tcpp -d <TRAIN> -t <TEST> "
            << "[-r <ROUND>] [-M <METHOD>] [-m <VARIANT>] [-T <TWF>] [-k <K>] "
            << "[-F <FS_TYPE> -f <voc %>] [-v <V>]"
            << std::endl << std::endl;
  std::cerr << "ARGUMENT        DESCRIPTION" << std::endl;
  std::cerr << "<TRAIN>         training file" << std::endl;
  std::cerr << "<TEST>          test file" << std::endl;
  std::cerr << "<ROUND>         informs the current trial number" << std::endl;
  std::cerr << "<METHOD>        specify the classification algorithm. Possible values are:" << std::endl;
  std::cerr << "  naivebayes    Naive Bayes probabilistic classifier" << std::endl;
  std::cerr << "  knn           K Nearest Neighbors classifier (KNN)" << std::endl;
  std::cerr << "  rocchio       Rocchio (centroid-based) classifier" << std::endl;
  std::cerr << "<FS_TYPE>       Feature selection method" << std::endl;
  std::cerr << "  0: BNS"        << std::endl;
  std::cerr << "  1: ODDS RATIO" << std::endl;
  std::cerr << "<TWF>           Temporally-aware algorithms. TWF in <TWF>." << std::endl;
  std::cerr << "<V>             Number of replications (currently, K-Fold Cross Validation)" << std::endl << std::endl;
  std::cerr << "KNN specific arguments:" << std::endl;
  std::cerr << "  -k <K>        number of nearest neighbors" << std::endl << std::endl;
  std::cerr << "Naive Bayes specific arguments:" << std::endl;
  std::cerr << "<VARIANT>       Variants of the algorithms." << std::endl
            << "                Currently, they are specific to Naive Bayes" << std::endl
            << "                and are specified by the following integers:" << std::endl;
  std::cerr << "  0: Log-based NaiveBayes (default)" << std::endl;
  std::cerr << "  1: Complement NaiveBayes" << std::endl;
  std::cerr << "  2: True probability NaiveBayes" << std::endl;
}


void errorMessage(const std::string &msg) {
  std::cerr << std:: endl << "Fatal error: " << msg << std::endl << std::endl;
  print_usage();
  exit(EXIT_FAILURE);
}


SupervisedClassifier * get_traditional_classifier(method_t &m, params_t &p){
 SupervisedClassifier * cc = NULL;
 
 switch(m) {
  case NB_LOG:
     cc = new nb_log(p.round, p.alpha, p.lambda, p.unif_prior);
     break;
  case NB_COMPL:
     cc = new nb_compl(p.round, p.alpha, p.lambda);
     break;
  case NB_PROB:
     cc = new nb(p.round, p.alpha, p.lambda, p.unif_prior);
     break;
  case NB_GAUSS:
     cc = new nb_gaussian(p.round, p.alpha);
     break;
  case KNN:
     cc = new knn(p.k, p.round);
     break;
  case ROCCHIO:
     cc = new Rocchio(p.round);
     break;
  case IITI:
     cc = new iti_incremental(p.round);
     break;
  case BITI:
     cc = new iti_batch(p.round);
     break;
  case RF1:
     cc = new RandomForest(p.round, p.rf_m, p.rf_trees, p.rf_docs, p.rf_height);
     break;
  case DTree:
     cc = new DT(p.round);
     break;
  case RF2:
     cc = new RF(p.round, p.rf_m, p.rf_trees, p.rf_height);
     break;
  case RFKNN:
     cc = new RF_KNN(p.round, p.rf_m, p.k, p.rf_trees);
     break;
  case RFKRAND:
     cc = new RF_KR(p.round, p.rf_m, p.k, p.rf_trees);
     break;
  case BRF:
     cc = new RF_BOOST(p.round, p.rf_m, p.rf_trees, p.rf_height, p.rf_trn);
     break;
  default:
     std::stringstream ss;
     ss << "Invalid method id (id=" << p.method <<")";
     errorMessage(ss.str());
 }
 return cc;
}

TemporalClassifier * get_temporal_classifier(method_t &m, params_t &p){
  TemporalClassifier * cc = NULL;
  switch(m) {
    case ROCCHIO:
    cc = new TemporalRocchio(std::string(p.twf_file), p.round, p.beta, p.window, p.grad_window, p.w_size);
    break;
  case NB_PROB:
    cc = new temporal_nb(std::string(p.twf_file), p.round, p.alpha, p.beta, p.lambda, p.unif_prior, p.window, p.grad_window, p.w_size);
    break;
  case NB_LOG:
    cc = new temporal_nb_log(std::string(p.twf_file), p.round, p.alpha, p.beta, p.lambda, p.unif_prior, p.window, p.grad_window, p.w_size);
    break;
  case KNN:
    cc = new temporal_knn(std::string(p.twf_file), p.k, p.round, p.beta, p.window, p.grad_window, p.w_size);
    break;
  default:
    std::stringstream ss;
    ss << "Invalid method id (id=" << p.method <<")";
    errorMessage(ss.str());
  }
  return cc;
}

FeatureSelector * get_fs(params_t &p){
  FeatureSelector * fs = NULL;
  switch(p.fs) {
    case BNS: {
      std::cerr << "[TCPP] Including feature selector BNS..." << std::endl;
      fs = new fs_bns(p.train_file, p.fs_p, true);
      break;
    }
    case ODDS: {
      std::cerr << "[TCPP] Including feature selector ODDS RATIO..." << std::endl;
      fs = new fs_odds_ratio(p.train_file, p.fs_p, true);
      break;
    }
    case IG:
    case CC:
    default: break;
  }
  return fs;
}

SupervisedClassifier *get_classifier(params_t &p) { 
  SupervisedClassifier * cc = NULL;
  if(!p.temporal && !p.two_pass && !p.window && !p.grad_window && !p.onsco && !p.ext_onsco){
    cc = get_traditional_classifier(p.method, p);
  }
  else if(!p.two_pass && !p.window && !p.grad_window && !p.onsco && !p.ext_onsco){
    cc = get_temporal_classifier(p.method, p);
  }
  else if(p.window && p.grad_window && !p.two_pass && !p.onsco && !p.ext_onsco){
    if(!p.temporal)
      errorMessage("You must specify a TWF using -T <twf>");

    cc = get_temporal_classifier(p.method, p);
  }
  else if(!p.grad_window && !p.two_pass && !p.onsco && !p.ext_onsco){
    if(p.temporal)
      errorMessage("You must not use -T <twf> for binary twf(window) classifier.");

    p.twf_file = (char*)""; 

    cc = get_temporal_classifier(p.method, p);
  }
  else if((p.onsco || p.ext_onsco) && !p.two_pass){
    if(!p.temporal && (!p.window || p.grad_window)) 
      errorMessage("You must specify a TWF using -T <twf>");
    if(p.window && !p.grad_window){
      if(p.temporal)
        errorMessage("You must not use -T <twf> for binary twf(window) classifier.");
      else
        p.twf_file = (char*)"";
    }

    SupervisedClassifier * ic = NULL;
    if (p.method == NB_LOG || p.method == NB_COMPL || p.method == NB_PROB) {
      std::cerr << "[TCPP] Swapping to NB_PROB" << std::endl;
      p.method = NB_PROB;
      ic = get_traditional_classifier(p.method, p);
      nb *c = dynamic_cast<nb *>(ic);
//      MyBig alpha("0.0e+800");
//      alpha += static_cast<MyBig>(p.alpha);
//      c->set_alpha(alpha);
      std::cerr << "  > alpha=" << c->get_alpha() << std::endl; 
    }
    else ic = get_traditional_classifier(p.method, p);
    
    if(p.fs != NONE)
      ic->set_feature_selector(get_fs(p));

    if (p.ext_onsco)
      cc = new ExtOnScoresClassifier(p.twf_file, ic, p.round, p.beta);
    else
      cc = new OnScoresClassifier(p.twf_file, ic, p.round, p.beta);
  }
  else if(p.two_pass && !p.onsco && !p.ext_onsco){
    if(p.temporal)
      errorMessage("You must not use -T <twf> for a two-pass classifier."); 

    SupervisedClassifier * fst = get_traditional_classifier(p.fst_pass_cl, p);

    p.twf_file = (char*)""; 

    TemporalClassifier * snd = get_temporal_classifier(p.snd_pass_cl, p);
    if(p.fs != NONE)
      snd->set_feature_selector(get_fs(p));

    cc = new TwoPassClassifier(fst, snd, p.round, p.beta);
  }
  else if(p.two_pass && (p.onsco || p.ext_onsco)){
    if(p.temporal)
      errorMessage("You must not use -T <twf> for a two-pass classifier.");

    p.twf_file = (char*)"";

    TemporalClassifier * os = NULL;
    SupervisedClassifier * fst = get_traditional_classifier(p.fst_pass_cl, p);
    SupervisedClassifier * snd = get_traditional_classifier(p.snd_pass_cl, p);

    if(p.fs != NONE)
      snd->set_feature_selector(get_fs(p));

    if (p.ext_onsco) {
      os = new ExtOnScoresClassifier(p.twf_file, snd, p.round, p.beta);
    }
    else {
      os = new OnScoresClassifier(p.twf_file, snd, p.round, p.beta);
    }
    
    if(p.fs != NONE)
      os->set_feature_selector(get_fs(p));

    cc = new TwoPassClassifier(fst, os, p.round, p.beta);
  }
  else{
    std::cerr << "Invalid Input" << std::endl;
    exit(1);
  }
  if(!p.two_pass){
    if(p.fs != NONE)
      cc->set_feature_selector(get_fs(p));
  }
  if (p.raw) cc->use_raw_weights();
  cc->set_distance(p.dt);
  return cc;
}

bool run_classifier(params_t &p) {
  std::string train = std::string(p.train_file);
  std::string test = std::string("");
  if (!p.validation) test = std::string(p.test_file);

  SupervisedClassifier *cc = NULL;
  cc = get_classifier(p);
  if (cc) {
    std::cerr << "[TCPP] Selected " << typeid(*cc).name() << std::endl;
    struct timeval init_time;
    struct timeval spent_time;
    gettimeofday(&init_time, NULL);
    cc->train(train);
    cc->test(test);
    spent_time = Utils::get_time_spent(&init_time);
    std::cerr << "[TIME]" << spent_time.tv_sec << " " << static_cast<double>(spent_time.tv_usec/1000000) << std::endl;
    delete cc; cc = NULL;
  }
  else return false;
  return true;
}

bool run_validation(params_t &p) {
  std::string train = std::string(p.train_file);
  std::string test = std::string("");
  if (!p.validation) test = std::string(p.test_file);

  Validator *v = NULL;
  SupervisedClassifier *cc = NULL;
  
  cc = get_classifier(p);
  if (cc == NULL) {
    std::cerr << "[TCPP] NULL CLASSIFIER !!!" << std::endl;
    exit(1);
  }

  // FIXME When random sup-sampling implemented,
  //       modify me to properly select the validation
  //       algorithm
  if (cc) {
    v = new KFold(p.train_file, p.replications, cc);
    if (v) {
      v->do_validation();
      delete v; v = NULL;
    }
    else {
      delete cc; cc = NULL;
      return false;
    }
    delete cc; cc = NULL;
  }
  else return false;
  return true;
}

int main(int argc, char** argv) {
  int opt = -1;
  params_t params;
  init_params(params);
  while ((opt = getopt (argc, argv, "r:d:t:o:m:n:N:H:x:k:v:s:F:f:a:b:l:T:1:2:D:PESwguRZ")) != -1) {
    switch (opt) {
      case 't': params.test_file = optarg; break;
      case 'd': params.train_file = optarg; break;
      case 'o': params.out_file = optarg; break;
      case 'r': params.round = atoi(optarg); break;
      case 'm': params.method = parse_method(optarg); break;
      case 'k': params.k = atoi(optarg); break;
      case 'v': params.validation = true;
                params.replications = atoi(optarg); break;
      case 'F': params.fs = parse_fs(atoi(optarg)); break;
      case 'f': params.fs_p = atof(optarg); break;
      case 'T': params.temporal = true;
                params.twf_file = optarg; break;
      case 'a': params.alpha = atof(optarg); break;
      case 'b': params.beta = atof(optarg); break;
      case 'l': params.lambda = atof(optarg); break;
      case 'u': params.unif_prior = true; break;
      case 'P': params.two_pass = true; break;
      case '1': params.fst_pass_cl = parse_method(optarg); break;
      case '2': params.snd_pass_cl = parse_method(optarg); break;
      case 'S': params.onsco = true; break;
      case 'E': params.ext_onsco=true; break;
      case 'w': params.window = true; break;
      case 'g': params.grad_window = true; break;
      case 's': params.w_size = atoi(optarg); break;
      case 'n': params.rf_trees = atoi(optarg); break;
      case 'x': params.rf_m = atof(optarg); break;
      case 'N': params.rf_docs = atof(optarg); break;
      case 'H': params.rf_height = atoi(optarg); break;
      case 'R': params.raw = true; break;
      case 'D': params.dt = parse_dist(optarg); break;
      case 'Z': params.rf_trn = true; break; 
      case '?':
        if ((optopt == 'd') || (optopt == 't') || (optopt == 'r') ||
            (optopt == 'm') || (optopt == 'k') || (optopt == 'T'))
          std::cerr << "Option -" << (char)optopt
                    << " requires an argument." << std::endl;
        else if (isprint (optopt))
          std::cerr << "Unknown option `-" << (char)optopt
                    << "'." << std::endl;
        else
          std::cerr << "Unknown option character `-" << (char)optopt
                    << "'." << std::endl;
        print_usage();
        return 1;
      default:
        print_usage();
        return 1;
    }
  }

  if (params.validation) {
    if (!params.train_file) {
      print_usage("You must specify the input data with [-d <INPUT>].");
      return 1;
    }
    else run_validation(params);
  }
  else {
    if (!params.test_file || !params.train_file) {
      print_usage("You must specify the training and testing files with [-d <TRAIN_FILE> -t <TEST_FILE>].");
      return 1;
    }
    else if(!run_classifier(params)) return 1;
  }

  return 0;
}
