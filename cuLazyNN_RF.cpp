#include "cuLazyNN_RF.h"

#include <iostream>

// tcpp
#include "tcpp/tree.hpp"
#include "tcpp/rf.hpp"

void prepareTrainSamples(RF *rf, Dataset &training, cuSimilarity *k_nearest, unsigned int K)
{
	for(int i = 0; i < K; i++) {
		DTDocument *doc = new DTDocument();
        cuSimilarity &sim = k_nearest[i];

		unsigned int idx = sim.doc_id;
	    doc->set_id(Utils::toString(idx));
	    doc->set_class(Utils::toString(training.getSamples()[idx].y));

        std::map<unsigned int, double>::iterator it;
		for(it = training.getSamples()[idx].features.begin(); it != training.getSamples()[idx].features.end(); ++it){
			unsigned int term_id = it->first;
			double term_count = it->second;
			
			doc->insert_term(term_id, term_count * log((double)training.size() / float(max(1, training.getIdf(term_id))))); //* log((double)training.size() / float(max(1, training.getIdf(term_id))));
		}
    	
    	rf->add_document(doc);
    }
}

cuLazyNN_RF::cuLazyNN_RF(){
}

cuLazyNN_RF::~cuLazyNN_RF(){
	doc_to_class.clear();
	entries.clear();
}

cuLazyNN_RF::cuLazyNN_RF(Dataset &data){
	training = data;

	convertDataset(data);

	buildInvertedIndex();
}

void cuLazyNN_RF::train(Dataset &data){
	training = data;

	convertDataset(data);

	buildInvertedIndex();
}

int cuLazyNN_RF::classify(std::map<unsigned int, double> test_features, int K){

	DTDocument * doc = new DTDocument();

	doc->set_id("0");
	doc->set_class("1");
	std::vector<Entry> query;
	std::map<unsigned int, double>::iterator it;
	for(it = test_features.begin(); it != test_features.end(); ++it){
		unsigned int term_id = it->first;
		double term_count = it->second;

		query.push_back(Entry(0, term_id, term_count)); // doc_id, term_id, term_count
		doc->insert_term(term_id, term_count * log((double)training.size() / float(max(1, training.getIdf(term_id)))));
	}

    //Creates an empty document if there are no terms
    if(query.empty()) {
        query.push_back(Entry(0, 0, 0));
    }

	cuSimilarity *k_nearest = KNN(inverted_index, query, K, CosineDistance);
	Scores<double> similarities(doc->get_id(), doc->get_class());

	RF * rf = new RF(0, 1.0, 100);
	  
	rf->set_doc_delete(false);
	
	double end, start = gettime();
	prepareTrainSamples(rf, training, k_nearest, K);
	end = gettime();
    printf("Total time taken preparing samples: %lf seconds\n", end - start);

	rf->build();

	similarities = rf->classify(doc);
	
	delete rf;
	delete doc;

	return atoi(similarities.top().class_name.c_str());
}

void cuLazyNN_RF::convertDataset(Dataset &data){
	
	num_terms = 0;
	// delete all old entries
	entries.clear();

	num_docs = data.getSamples().size();
	for (unsigned int i = 0; i < num_docs; ++i)
	{
		unsigned int doc_id = i;

		std::map<unsigned int, double>::iterator it;
		for(it = data.getSamples()[i].features.begin(); it != data.getSamples()[i].features.end(); ++it){
			unsigned int term_id = it->first;
			double term_cout = it->second;

			num_terms = std::max(num_terms, term_id + 1);
	
			entries.push_back(Entry(doc_id, term_id, term_cout)); // doc_id, term_id, term_count
		}

		doc_to_class[doc_id] = data.getSamples()[i].y;
	}
}

void cuLazyNN_RF::buildInvertedIndex(){
	inverted_index = make_inverted_index(num_docs, num_terms, entries);
}
