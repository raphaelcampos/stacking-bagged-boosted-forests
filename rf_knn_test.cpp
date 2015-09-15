// OpenCV
#include <iostream>
#include "Dataset.h"

#include "tcpp/tree.hpp"

#include "tcpp/rf_knn.hpp"
#include "tcpp/rf_bst.hpp"
#include "tcpp/rf.hpp"


using namespace std;


void tccpPrepareTrainSamples(RF_KNN* rf, Dataset &training, unsigned int K)
{
	printf("Allocation Matrix %dx%d...\n", K , training.dimension());
/*
	vector<DTDocument> samples;

	for(int i = 0; i < K; i++) {
        DTDocument *doc = new DTDocument(); //samples.push_back(doc);

        unsigned int idx = i;//sim.doc_id;
		
		doc->set_id(Utils::toString(idx));
		doc->set_class(Utils::toString(training.getSamples()[idx].y));

        std::map<unsigned int, double>::iterator it;
		for(it = training.getSamples()[idx].features.begin(); it != training.getSamples()[idx].features.end(); ++it){
			unsigned int term_id = it->first;
			double term_count = it->second;
			doc->insert_term(term_id, term_count);
		
			rf->insert_knn_term(term_id, doc, term_cout);
		}

		rf->add_document(doc);
    }

    //rf->updateIDF();
  	//rf->updateTFIDF();*/

    printf("finish...\n");

}

DTDocument * tcppPrepareSample(std::map<unsigned int, double> test_features, Dataset &training){
	DTDocument * doc = new DTDocument();

	std::map<unsigned int, double>::iterator it;
	for(it = test_features.begin(); it != test_features.end(); ++it){
		unsigned int term_id = it->first;
		double term_count = it->second;

		doc->insert_term(term_id, term_count);
	}
	return doc;
}

int main(int argc, char const *argv[])
{
	Dataset training, test_set;
	
	RF_KNN * knn_rf = new RF_KNN(0, 0.03, 30, 200);
	  
	knn_rf->train(argv[1]);

	knn_rf->set_output_file(argv[3]);
	knn_rf->test(argv[2]);

	/*RF * rf = new RF(0, 1.0, 200);
	  
	rf->train("lib/gtknn/data/treino.dat");

	rf->set_output_file("saida.out");
	rf->test("lib/gtknn/data/teste.dat");
    */
	return EXIT_SUCCESS	;
}
