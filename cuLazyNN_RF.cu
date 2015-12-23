#include "cuLazyNN_RF.cuh"

#include <iostream>

// tcpp
#include "tcpp/tree.hpp"
#include "tcpp/rf.hpp"

inline void prepareTrainSamples(RF *rf, Dataset &training, cuSimilarity *k_nearest, unsigned int K)
{
	for(int i = 0; i < K; i++) {
		DTDocument *doc = new DTDocument();
        cuSimilarity &sim = k_nearest[i];

		unsigned int idx = sim.doc_id;
	    doc->set_id(Utils::toString(idx));
	    doc->set_class(Utils::toString(training.getSamples()[idx].y));

        std::map<unsigned int, float>::iterator it;
		for(it = training.getSamples()[idx].features.begin(); it != training.getSamples()[idx].features.end(); ++it){
			unsigned int term_id = it->first;
			float term_count = it->second;
			
			doc->insert_term(term_id, term_count); //* log((double)training.size() / float(max(1, training.getIdf(term_id))));
		}
    	
    	rf->add_document(doc);
    }
}

cuLazyNN_RF::cuLazyNN_RF(){
}

cuLazyNN_RF::~cuLazyNN_RF(){
}

cuLazyNN_RF::cuLazyNN_RF(Dataset &data) : cuKNN(data){
	training = data;
}

void cuLazyNN_RF::train(Dataset &data){
	training = data;
	cuKNN.train(data);
}

int cuLazyNN_RF::classify(const std::map<unsigned int, float> &test_features, int K){

	DTDocument * doc = new DTDocument();
	doc->set_id("0");doc->set_class("1");
	Scores<double> similarities(doc->get_id(), doc->get_class());

	std::map<unsigned int, float>::const_iterator it;
	for(it = test_features.begin(); it != test_features.end(); ++it){
		unsigned int term_id = it->first;
		float term_count = it->second;

		doc->insert_term(term_id, 1 + log(term_count));
	}

	cuSimilarity *k_nearest = cuKNN.getKNearestNeighbors(test_features, K);
	
	RF * rf = new RF(0, 0.03, 200);
	
	prepareTrainSamples(rf, training, k_nearest, K);
	
	rf->build();

	similarities = rf->classify(doc);
	
	delete rf;
	delete doc;
	delete[] k_nearest;

	return atoi(similarities.top().class_name.c_str());

	//return cuKNN.getMajorityVote(k_nearest, K);
}
