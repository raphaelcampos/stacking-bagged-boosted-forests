
#ifndef _cuLazyNN_RF__
#define _cuLazyNN_RF__

// gtknn
#include "structs.cuh"
#include "utils.cuh"
#include "inverted_index.cuh"
#include "knn.cuh"
#include "cuda_distances.cuh"

#include "LazyNN_RF.h"

#include <map>

class cuLazyNN_RF : LazyNN_RF{
	
	public:
		cuLazyNN_RF();
		cuLazyNN_RF(Dataset &data);

		void train(Dataset &data);
		double classify(std::map<unsigned int, double> test_sample, int K);

	private:
		// gtknn dataset formart
		std::vector<Entry> entries;

		// OpenCV dataset format

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
};

#endif