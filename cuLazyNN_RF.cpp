#include "cuLazyNN_RF.h"

#include <iostream>

cuLazyNN_RF::cuLazyNN_RF(){
}

cuLazyNN_RF::cuLazyNN_RF(Dataset &data){
	convertDataset(data);

	buildInvertedIndex();

	
}

void cuLazyNN_RF::train(Dataset &data){
	convertDataset(data);
}

double cuLazyNN_RF::classify(std::map<unsigned int, double> test_sample, int K){
	return 10;
}

void cuLazyNN_RF::convertDataset(Dataset &data){
	// delete all old entries
	entries.clear();

	num_docs = data.getSamples().size();
	for (int i = 0; i < num_docs; ++i)
	{
		// doc_id start at 1 that's why (i + 1)
		// it is a gtknn conversion
		unsigned int doc_id = i + 1;

		std::map<unsigned int, double>::iterator it;
		for(it = data.getSamples()[i].features.begin(); it != data.getSamples()[i].features.end(); ++it){
			unsigned int term_id = it->first;
			double term_cout = it->second;

			num_terms = std::max(num_terms, term_id + 1);
	
			entries.push_back(Entry(doc_id, term_id, term_cout)); // doc_id, term_id, term_count
		}
	}
}

void cuLazyNN_RF::buildInvertedIndex(){
	inverted_index = make_inverted_index(num_docs, num_terms, entries);
}
