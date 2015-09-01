
#ifndef _cuLazyNN_RF__
#define _cuLazyNN_RF__

// Gt-kNN
#include "structs.cuh"
#include "utils.cuh"
#include "inverted_index.cuh"
#include "knn.cuh"
#include "cuda_distances.cuh"

// OpenCV
#include <opencv2/opencv.hpp>
	
#include "LazyNN_RF.h"

#include <map>

using namespace cv;
using namespace cv::ml;

class cuLazyNN_RF : LazyNN_RF{
	
	public:
		cuLazyNN_RF();
		cuLazyNN_RF(Dataset &data);

		void train(Dataset &data);
		int classify(std::map<unsigned int, double> test_sample, int K);

	private:
		Dataset training;

		// gtknn dataset formart
		std::vector<Entry> entries;

		// Random Forest object
		Ptr<RTrees> randomForest;

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

		Ptr<TrainData> prepareTrainSamples(Similarity *k_nearest, unsigned int K);
};

#endif