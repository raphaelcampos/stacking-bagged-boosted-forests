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

	buildInvertedIndex();
}

int cuLazyNN_RF::classify(std::map<unsigned int, double> test_features, int K){
	
	std::vector<Entry> query;
	std::map<unsigned int, double>::iterator it;
	for(it = test_features.begin(); it != test_features.end(); ++it){
		unsigned int term_id = it->first;
		double term_cout = it->second;

		query.push_back(Entry(0, term_id, term_cout)); // doc_id, term_id, term_count
	}

    //Creates an empty document if there are no terms
    if(query.empty()) {
        query.push_back(Entry(0, 0, 0));
    }

	Similarity *k_nearest = KNN(inverted_index, query, K, CosineDistance);


	std::map<int, int> vote_count;
    std::map<int, int>::iterator v_it;
    for(int i = 0; i < K; i++) {
        Similarity &sim = k_nearest[i];
        vote_count[doc_to_class[sim.doc_id]]++;
    }

    int guessed_class = -1;
    int max_votes = 0;

    for(v_it = vote_count.begin(); v_it != vote_count.end(); v_it++) {
        if(v_it->second > max_votes) {
            max_votes = v_it->second;
            guessed_class = v_it->first;
        }
    }

	return guessed_class;
}

void cuLazyNN_RF::convertDataset(Dataset &data){
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
