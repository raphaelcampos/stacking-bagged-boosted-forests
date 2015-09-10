
#ifndef _cuNearestNeighbors__
#define _cuNearestNeighbors__

// Gt-kNN
#include "structs.cuh"
#include "utils.cuh"
#include "inverted_index.cuh"
#include "knn.cuh"
#include "cuda_distances.cuh"	

#include "LazyNN_RF.h"

#include <map>

class cuNearestNeighbors {
	
	public:
		cuNearestNeighbors();
		cuNearestNeighbors(Dataset &data);

		~cuNearestNeighbors();

		void train(Dataset &data);
		int classify(std::map<unsigned int, double> &test_sample, int K);
		cuSimilarity * getKNearestNeighbors(std::map<unsigned int, double> &test_features, int K);
		int getMajorityVote(cuSimilarity *k_nearest, int K);

	private:
		Dataset training;

		// gtknn dataset formart
		std::vector<Entry> entries;

		// Dataset statistics
		unsigned int num_docs;
		unsigned int num_terms;

		InvertedIndex inverted_index;

		std::map<unsigned int, int> doc_to_class;

		/**
		 * Converts Dataset obj to gtknn format
		 * and OpenCV format as well.
		 */
		void convertDataset(Dataset &data);

		/**
		 * Builds the inverted index on GPU
		 * based on the given dataset.
		 */
		void buildInvertedIndex();

		void createRF();

		//void prepareTrainSamples(RF * rf, cuSimilarity *k_nearest, unsigned int K);
};

#endif