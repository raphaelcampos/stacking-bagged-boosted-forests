
#ifndef _cuLazyNN_RF__
#define _cuLazyNN_RF__

// Gt-kNN
#include "structs.cuh"
#include "utils.cuh"
#include "inverted_index.cuh"
#include "knn.cuh"
#include "cuda_distances.cuh"	

#include "LazyNN_RF.h"
#include "cuNearestNeighbors.cuh"

#include <map>

class cuLazyNN_RF : LazyNN_RF{
	
	public:
		/**
		 * Default constructor.
		 */
		cuLazyNN_RF();

		/**
		 * Constructor. Trains the model based on the given training set.
		 * \param data - Training set
		 */
		cuLazyNN_RF(Dataset &data);

		/**
		 * Destructor.
		 */
		~cuLazyNN_RF();

		/**
		 * Trains the model based on the given training set.
		 * \param data - Training set
		 */
		void train(Dataset &data);
		
		/**
		 * Classify a given feature vector.
		 * \param  test_sample - Feature vector
		 * \param  K           - K nearest neighbors to training the random forest
		 */
		int classify(const std::map<unsigned int, float> &test_sample, int K);

	private:
		Dataset training;

		cuNearestNeighbors cuKNN;	

		// Dataset statistics
		unsigned int num_docs;
		unsigned int num_terms;
};

#endif