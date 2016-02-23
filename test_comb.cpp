#include <iostream>

#include "lib/broof/rf_bst.hpp"
#include "lib/broof/rf_knn.hpp"


class RF_KNN_BOOST : public SupervisedClassifier{
  public:
    RF_KNN_BOOST(unsigned int r, double m=1.0, unsigned int num_trees=10, unsigned int k = 30, unsigned int n_boost_it = 10,unsigned int maxh=0, bool trn_err=false) : SupervisedClassifier(r) { lazy = new RF_KNN(r, 0.03, k, num_trees); broof = new RF_BOOST(r, m, n_boost_it); }
    ~RF_KNN_BOOST();
    void reset_model();
    bool parse_train_line(const std::string&);
    void train(const std::string&);
    void parse_test_line(const std::string&);
    void build();
    void set_doc_delete(const bool&);
    Scores<double> classify(const DTDocument*);
   
   private:
    double norm(Scores<double> s){
       double norm = 0;
       while (!s.empty()) {
          Similarity<double> sim = s.top();
          norm += sim.similarity;
          s.pop();
        }
        return norm;
    }
    std::vector<const DTDocument*> docs_;
    RF_BOOST * broof;
    RF_KNN * lazy;
};


RF_KNN_BOOST::~RF_KNN_BOOST(){
	delete [] lazy;
	delete [] broof;
}

void RF_KNN_BOOST::reset_model(){
    lazy->reset_model();
    broof->reset_model();
}

bool RF_KNN_BOOST::parse_train_line(const std::string& line){
  return lazy->parse_train_line(line) && broof->parse_train_line(line);
}

void RF_KNN_BOOST::train(const std::string& train_fn){
  //SupervisedClassifier::train(train_fn);
  lazy->train(train_fn);
  broof->train(train_fn);
}

Scores<double> RF_KNN_BOOST::classify(const DTDocument* doc){
  Scores<double> similarities(doc->get_id(), doc->get_class());
  
  return similarities;
}

void RF_KNN_BOOST::parse_test_line(const std::string& line){

  Scores<double> s = lazy->classify(line);
  Scores<double> similarities(s.get_id(), s.get_class());
  std::map<std::string, double> sco;
  double n = norm(s);
  while (!s.empty()) {
      Similarity<double> sim = s.top();
      sco[sim.class_name] += sim.similarity/n;// * (oob_err == 0.0 ? 1.0 : oob_err == 1.0 ? 0.0 : log((1.0-oob_err)/oob_err));
      s.pop();
    }
  s = broof->classify(line);
  n = norm(s);
  while (!s.empty()) {
      Similarity<double> sim = s.top();
      sco[sim.class_name] += sim.similarity/n;// * (oob_err == 0.0 ? 1.0 : oob_err == 1.0 ? 0.0 : log((1.0-oob_err)/oob_err));
      s.pop();
    }

    std::map<std::string, double>::const_iterator s_it = sco.begin();
  while (s_it != sco.end()) {
    similarities.add(s_it->first, s_it->second);
    ++s_it;
  }

  get_outputer()->output(similarities);
}

using namespace std;


int main(int argc, char const *argv[])
{
	RF_KNN_BOOST * broof = new RF_KNN_BOOST(0, 0.0005, 200, 30, 200);
	  
	broof->train(argv[1]);

	broof->set_output_file(argv[3]);
	broof->test(argv[2]);
    
	return EXIT_SUCCESS	;
}
