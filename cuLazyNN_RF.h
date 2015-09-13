
#ifndef _cuLazyNN_RF__
#define _cuLazyNN_RF__

// Gt-kNN
#include "structs.cuh"
#include "utils.cuh"
#include "inverted_index.cuh"
#include "knn.cuh"
#include "cuda_distances.cuh"	

#include "LazyNN_RF.h"
#include "cuNearestNeighbors.h"

#include <map>

class cuLazyNN_RF : LazyNN_RF{
	
	public:
		cuLazyNN_RF();
		cuLazyNN_RF(Dataset &data);

		~cuLazyNN_RF();

		void train(Dataset &data);
		int classify(const std::map<unsigned int, float> &test_sample, int K);

	private:
		Dataset training;

		cuNearestNeighbors cuKNN;	

		// Dataset statistics
		unsigned int num_docs;
		unsigned int num_terms;
};

#endif