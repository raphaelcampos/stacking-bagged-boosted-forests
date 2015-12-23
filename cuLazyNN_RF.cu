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

cuLazyNN_RF::cuLazyNN_RF(Dataset &data, int n_gpus) : cuKNN(data, n_gpus){
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

		doc->insert_term(term_id, term_count);
	}

	cuSimilarity *k_nearest = cuKNN.getKNearestNeighbors(test_features, K);
	//printf("cuLazyNN_RF\n");
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

std::vector<int> cuLazyNN_RF::classify(Dataset &test, int K){
	std::vector<cuSimilarity*> idxs = cuKNN.getKNearestNeighbors(test, K);
	std::vector<int> pred;
	
	int correct_cosine = 0, wrong_cosine = 0;

	for (int i = 0; i < idxs.size(); ++i)
	{
		DTDocument * doc = new DTDocument();
		doc->set_id("0");doc->set_class("1");
		Scores<double> similarities(doc->get_id(), doc->get_class());

		std::map<unsigned int, float> &test_features = test.getSamples()[i].features;

		std::map<unsigned int, float>::const_iterator it;
		for(it = test_features.begin(); it != test_features.end(); ++it){
			unsigned int term_id = it->first;
			float term_count = it->second;

			doc->insert_term(term_id, term_count);
		}

		RF * rf = new RF(0, 0.03, 100);
	
		prepareTrainSamples(rf, training, idxs[i], K);
		
		rf->build();

		similarities = rf->classify(doc);
		
		delete rf;
		delete doc;
		
		free(idxs[i]);

		pred.push_back(atoi(similarities.top().class_name.c_str()));

		if(pred.back() == test.getSamples()[i].y) {
            correct_cosine++;   
        } else {
            wrong_cosine++;
        }

        std::cerr.precision(4);
        std::cerr.setf(std::ios::fixed);
        std::cerr << "\r" << double(i+1)/test.getSamples().size() * 100 << "%" << " - " << double(correct_cosine) / (i+1);

	}

	return pred;
}
