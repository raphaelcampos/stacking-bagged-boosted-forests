
#ifndef _cuNearestNeighbors__
#define _cuNearestNeighbors__

// Gt-kNN
#include "structs.cuh"
#include "utils.cuh"
#include "inverted_index.cuh"
#include "knn.cuh"
#include "cuda_distances.cuh"
#include "Dataset.h"

#include <map>

class cuNearestNeighbors {
	
	public:
		/**
		 * Default constructor.
		 */
		cuNearestNeighbors(int n_gpus = 1){this->n_gpus = n_gpus;};
		
		/**
		 * Constructor.
		 * \param data - Training set
		 */
		cuNearestNeighbors(Dataset &data, int n_gpus = 1);

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
		 * \param  K           - Hiperparameter K, Number of nearest neighbors
		 * \return             Feature vector predicted class
		 */
		int classify(std::map<unsigned int, float> &test_sample, int K);

		/**
		 * Classify a given test set.
		 * \param  test 	   - Test set
		 * \param  K           - Hiperparameter K, Number of nearest neighbors
		 * \return             integer vector containing the predicted class for each test sample in test set.
		 */
		std::vector<int> classify(Dataset &test, int K);
		
		/**
		 * Returns the K nearrest neighbors to a given feature vector
		 * \param  test_features -  Feature vector
		 * \param  K             -	Number of nearest neighbors              
		 */
		cuSimilarity * getKNearestNeighbors(const std::map<unsigned int, float> &test_features, int K);

		std::vector<cuSimilarity*> getKNearestNeighbors(Dataset &test, int K);

		/**
		 * Return the winner class in the "election"
		 * \param  k_nearest - K nearest neighbors
		 * \param  K         - Number of nearest neighbors
		 */
		int getMajorityVote(cuSimilarity *k_nearest, int K);

		/**
		 * Returns the number of gpus used for kNN
		 * \return  number of gpus used
		 */
		int getNGpus(){return this->n_gpus;};

	private:
		// gtknn dataset formart
		std::vector<Entry> entries;

		// Dataset statistics
		unsigned int num_docs;
		unsigned int num_terms;

		InvertedIndex* inverted_indices;

		std::map<unsigned int, int> doc_to_class;

		int n_gpus;

		/**
		 * Converts Dataset obj to gtknn format.
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