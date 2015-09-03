#include "cuLazyNN_RF.h"

#include <iostream>

cuLazyNN_RF::cuLazyNN_RF(){
}

cuLazyNN_RF::~cuLazyNN_RF(){
	if(randomForest != NULL){
		delete randomForest;
	}
}

cuLazyNN_RF::cuLazyNN_RF(Dataset &data){
	training = data;

	convertDataset(data);

	buildInvertedIndex();
}

void cuLazyNN_RF::train(Dataset &data){
	training = data;

	convertDataset(data);

	buildInvertedIndex();
}

int cuLazyNN_RF::classify(std::map<unsigned int, double> test_features, int K){

	Mat testSample( 1, training.dimension(), CV_32F);
	std::vector<Entry> query;
	std::map<unsigned int, double>::iterator it;
	for(it = test_features.begin(); it != test_features.end(); ++it){
		unsigned int term_id = it->first;
		double term_count = it->second;

		query.push_back(Entry(0, term_id, term_count)); // doc_id, term_id, term_count

		float *ptr = testSample.ptr<float>(0);
		ptr[term_id] = term_count;
	}

    //Creates an empty document if there are no terms
    if(query.empty()) {
        query.push_back(Entry(0, 0, 0));
    }

	Similarity *k_nearest = KNN(inverted_index, query, K, CosineDistance);

	Ptr<RTrees> randomForest = RTrees::create();
	
	Ptr<TrainData> dt = prepareTrainSamples(k_nearest, K);
	printf("Training...\n");
	randomForest->train(dt);
	printf("Predicting...\n");
	return (int)randomForest->predict(testSample);
}

void cuLazyNN_RF::convertDataset(Dataset &data){
	
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

void cuLazyNN_RF::buildInvertedIndex(){
	inverted_index = make_inverted_index(num_docs, num_terms, entries);
}

void cuLazyNN_RF::createRF(){
	randomForest = RTrees::create();
    // Commented in order to allow the trees
    // be grown to the their maximal depth
    //randomForest->setMaxDepth(4);
    /*randomForest->setMinSampleCount(2);
    randomForest->setRegressionAccuracy(0.f);
    randomForest->setUseSurrogates(false);
    randomForest->setMaxCategories(16);
    randomForest->setPriors(Mat());
    randomForest->setCalculateVarImportance(false);
    randomForest->setActiveVarCount(1);
    randomForest->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 5, 0));*/
}


Ptr<TrainData> cuLazyNN_RF::prepareTrainSamples(Similarity *k_nearest, unsigned int K)
{
	printf("Allocation Matrix %dx%d...\n", K , training.dimension());

	Mat samples(K, training.dimension(), CV_32F);
	Mat responses(K, 1, CV_32F);

	for(int i = 0; i < K; i++) {
        Similarity &sim = k_nearest[i];

		unsigned int idx = sim.doc_id;
	    responses.at<double>(i, 0) = training.getSamples()[idx].y;
		float *ptr = responses.ptr<float>(i);
		ptr[0] = training.getSamples()[idx].y;

        std::map<unsigned int, double>::iterator it;
		for(it = training.getSamples()[idx].features.begin(); it != training.getSamples()[idx].features.end(); ++it){
			unsigned int term_id = it->first;
			double term_cout = it->second;
			//cout << term_cout << endl;
			float *ptr =  samples.ptr<float>(i);
			ptr[term_id] = term_cout;
		}
    }

    printf("Creating TrainData...\n");
	return TrainData::create(samples, ROW_SAMPLE, responses);
}
