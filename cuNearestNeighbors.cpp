#include "cuNearestNeighbors.h"

#include <iostream>


cuNearestNeighbors::~cuNearestNeighbors(){
	doc_to_class.clear();
	entries.clear();

	// ver como liberar memoria da placa
	freeInvertedIndex(inverted_index);
}

cuNearestNeighbors::cuNearestNeighbors(Dataset &data){
	convertDataset(data);

	buildInvertedIndex();
}

void cuNearestNeighbors::train(Dataset &data){
	convertDataset(data);

	buildInvertedIndex();
}

int cuNearestNeighbors::classify(std::map<unsigned int, float> &test_features, int K){
	
	cuSimilarity *k_nearest = getKNearestNeighbors(test_features, K);
	int vote = getMajorityVote(k_nearest, K);

	delete[] k_nearest;
	return vote;
}

cuSimilarity * cuNearestNeighbors::getKNearestNeighbors(const std::map<unsigned int, float> &test_features, int K){

	std::vector<Entry> query;
	std::map<unsigned int, float>::const_iterator end = test_features.end();
	for(std::map<unsigned int, float>::const_iterator it = test_features.begin(); it != end; ++it){
		unsigned int term_id = it->first;
		double term_count = it->second;

		// it means that query has higher dimensonality
		// than traning set. Thus, we remove that term
		if(term_id < num_terms)		
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

		std::map<unsigned int, float>::iterator it;
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
	std::map<int, double> vote_count;

	cuSimilarity &closest = k_nearest[0];
	cuSimilarity &further = k_nearest[K-1];

    for(int i = 0; i < K; ++i) {
        cuSimilarity &sim = k_nearest[i];
        vote_count[doc_to_class[sim.doc_id]] += sim.distance;
        //vote_count[doc_to_class[sim.doc_id]]+=((further.distance-sim.distance)/(further.distance-closest.distance))*((sim.distance+further.distance)/(closest.distance+further.distance))*(i);
        
        //vote_count[doc_to_class[sim.doc_id]]+=((further.distance-sim.distance)/(further.distance-closest.distance))*((double)i);
    }

    int max_votes = 0;
    int guessed_class = -1;
    std::map<int, double>::iterator end = vote_count.end();
    for(std::map<int, double>::iterator it = vote_count.begin(); it != end; it++) {
        if(it->second > max_votes) {
            max_votes = it->second;
            guessed_class = it->first;
        }
    }

    return guessed_class;
}

void cuNearestNeighbors::buildInvertedIndex(){
	inverted_index = make_inverted_index(num_docs, num_terms, entries);

	entries.clear();
}
