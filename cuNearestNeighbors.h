
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
		/**
		 * Default constructor.
		 */
		cuNearestNeighbors(){};
		
		/**
		 * Constructor.
		 * \param data - Training set
		 */
		cuNearestNeighbors(Dataset &data);

		/**
		 * Destructor.
		 */
		~cuNearestNeighbors();

		/**
		 * Trains the model based on the given training set.
		 * \param data - Training set
		 */
		void train(Dataset &data);

		/**
		 * Classify a given feature vector.
		 * \param  test_sample - Feature vector
		 * \param  K           - Hiperparameter K, Number of nearest neighbor
		 * \return             Feature vector predicted class
		 */
		int classify(std::map<unsigned int, float> &test_sample, int K);
		
		/**
		 * Returns the K nearrest neighbors to a given feature vector
		 * \param  test_features -  Feature vector
		 * \param  K             -	Number of nearest neighbors              
		 */
		cuSimilarity * getKNearestNeighbors(const std::map<unsigned int, float> &test_features, int K);
		

		/**
		 * Return the winner class in the "election"
		 * @param  k_nearest - K nearest neighbors
		 * @param  K         - Number of nearest neighbors
		 */
		int getMajorityVote(cuSimilarity *k_nearest, int K);

	private:
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
		 * \param - Training set
		 */
		void convertDataset(Dataset &data);

		/**
		 * Builds the inverted index on GPU
		 * based on the given dataset.
		 */
		void buildInvertedIndex();
};

#endif