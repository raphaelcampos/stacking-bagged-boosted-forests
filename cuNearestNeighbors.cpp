#include "cuNearestNeighbors.h"

#include <iostream>

cuNearestNeighbors::cuNearestNeighbors(){
}

cuNearestNeighbors::~cuNearestNeighbors(){
	doc_to_class.clear();
	entries.clear();
}

cuNearestNeighbors::cuNearestNeighbors(Dataset &data){
	training = data;

	convertDataset(data);

	buildInvertedIndex();
}

void cuNearestNeighbors::train(Dataset &data){
	training = data;

	convertDataset(data);

	buildInvertedIndex();
}

int cuNearestNeighbors::classify(std::map<unsigned int, double> &test_features, int K){
	return getMajorityVote(getKNearestNeighbors(test_features, K), K);
}

cuSimilarity * cuNearestNeighbors::getKNearestNeighbors(std::map<unsigned int, double> &test_features, int K){

	std::vector<Entry> query;
	std::map<unsigned int, double>::iterator it;
	for(it = test_features.begin(); it != test_features.end(); ++it){
		unsigned int term_id = it->first;
		double term_count = it->second;

		query.push_back(Entry(0, term_id, term_count)); // doc_id, term_id, term_count
	}

    //Creates an empty document if there are no terms
    if(query.empty()) {
        query.push_back(Entry(0, 0, 0));
    }

	return KNN(inverted_index, query, K, CosineDistance);
}

void cuNearestNeighbors::convertDataset(Dataset &data){
	
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

int cuNearestNeighbors::getMajorityVote(cuSimilarity *k_nearest, int K){
	std::map<int, int> vote_count;
    std::map<int, int>::iterator it;

    for(int i = 0; i < K; i++) {
        cuSimilarity &sim = k_nearest[i];
        vote_count[doc_to_class[sim.doc_id]]++;
    }

    int max_votes = 0;
    int guessed_class = -1;
    for(it = vote_count.begin(); it != vote_count.end(); it++) {
        if(it->second > max_votes) {
            max_votes = it->second;
            guessed_class = it->first;
        }
    }

    return guessed_class;
}

void cuNearestNeighbors::buildInvertedIndex(){
	inverted_index = make_inverted_index(num_docs, num_terms, entries);
}
