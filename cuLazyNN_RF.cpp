#include "cuLazyNN_RF.h"

#include <iostream>

cuLazyNN_RF::cuLazyNN_RF(){
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

	/*
	std::map<int, int> vote_count;
    std::map<int, int>::iterator v_it;
    for(int i = 0; i < K; i++) {
        Similarity &sim = k_nearest[i];
        vote_count[doc_to_class[sim.doc_id]]++;
    }

    int guessed_class = -1;
    int max_votes = 0;

    for(v_it = vote_count.begin(); v_it != vote_count.end(); v_it++) {
        if(v_it->second > max_votes) {
            max_votes = v_it->second;
            guessed_class = v_it->first;
        }
    }
	*/

   	printf("Training set - Num samples : %d, Dimension : %d\n", (int)training.size(), training.dimension());

   	Mat samples(K, training.dimension(), CV_32FC1);
	Mat responses(K, 1, CV_32FC1);

	std::cout << "Before loop " << K << std::endl << responses << std::endl;

	for(int i = 0; i < K; i++) {
        printf("%d\n", i);
	Similarity &sim = k_nearest[i];
	printf("before response - doc_id : %d\n", training.getSamples()[sim.doc_id].y);
        responses.at<double>(i, 0) = training.getSamples()[sim.doc_id].y;
	float *ptr = responses.ptr<float>(i);
	ptr[0] = training.getSamples()[sim.doc_id].y;
	printf("after response - doc_id : %d\n", sim.doc_id);
        std::map<unsigned int, double>::iterator it;
		for(it = training.getSamples()[sim.doc_id].features.begin(); it != training.getSamples()[sim.doc_id].features.end(); ++it){
			unsigned int term_id = it->first;
			double term_cout = it->second;
	
			float *ptr = samples.ptr<float>(i);
			ptr[term_id] = term_cout;
		}
    }

	std::cout << " Responses : " <<  responses << std::endl;


	printf("Creating DataTrain...\n");
    	//Ptr<TrainData> dt = TrainData::create(samples, ROW_SAMPLE, responses);

	printf("Training random forest...\n");

	printf("samples dim : %dx%d -  resposnses dim : %dx%d\n", samples.rows, samples.cols, responses.rows, responses.cols);
	randomForest->train(TrainData::create(samples, ROW_SAMPLE, responses));

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
    randomForest->setMaxDepth(4);
    randomForest->setMinSampleCount(2);
    randomForest->setRegressionAccuracy(0.f);
    randomForest->setUseSurrogates(false);
    randomForest->setMaxCategories(16);
    randomForest->setPriors(Mat());
    randomForest->setCalculateVarImportance(false);
    randomForest->setActiveVarCount(1);
    randomForest->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 5, 0));
}


Ptr<TrainData> cuLazyNN_RF::prepareTrainSamples(Similarity *k_nearest, unsigned int K)
{

	printf("Training set - Num samples : %d, Dimension : %d, Num classes : \n", (int)training.size(), training.dimension());

	Mat samples(K, training.dimension(), CV_32F);
	Mat responses(K, 1, CV_32F);

	std::cout << "aqui" << std::endl;

	for(int i = 0; i < K; i++) {
        Similarity &sim = k_nearest[i];
        samples.at<int>(i, 0) = training.getSamples()[sim.doc_id].y;

        std::map<unsigned int, double>::iterator it;
		for(it = training.getSamples()[sim.doc_id].features.begin(); it != training.getSamples()[sim.doc_id].features.end(); ++it){
			unsigned int term_id = it->first;
			double term_cout = it->second;
	
			responses.at<double>(i, 0) = term_cout;
		}
    }

    
    return TrainData::create(samples, ROW_SAMPLE, responses);
}
