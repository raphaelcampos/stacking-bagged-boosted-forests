
#ifndef _cuLazyNN_BROOF__
#define _cuLazyNN_BROOF__

// Gt-kNN
#include "structs.cuh"
#include "utils.cuh"
#include "inverted_index.cuh"
#include "knn.cuh"
#include "cuda_distances.cuh"	

#include "LazyNN_RF.h"
#include "cuNearestNeighbors.cuh"

#include <map>

class cuLazyNN_Boost : LazyNN_RF{
	
	public:
		/**
		 * Default constructor.
		 */
		cuLazyNN_Boost();

		/**
		 * Constructor. Trains the model based on the given training set.
		 * \param data - Training set
		 */
		cuLazyNN_Boost(Dataset &data, float max_features = 0.15, int n_boost_iter = 10, int n_gpus = 1);

		/**
		 * Destructor.
		 */
		~cuLazyNN_Boost();

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

		/**
		 * Classify a given feature vector.
		 * \param  test_sample - Feature vector
		 * \param  K           - K nearest neighbors to training the random forest
		 */
		std::vector<int> classify(Dataset &test, int K);

	private:
		Dataset training;

		cuNearestNeighbors cuKNN;	

		float max_features;
		int n_boost_iter;

		// Dataset statistics
		unsigned int num_docs;
		unsigned int num_terms;
};

#endif