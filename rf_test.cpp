// OpenCV
#include <opencv2/opencv.hpp>
#include <iostream>
#include "Dataset.h"

#include "tcpp/tree.hpp"
#include "tcpp/rf.hpp"

using namespace cv;
using namespace cv::ml;

using namespace std;

Ptr<TrainData> prepareTrainSamples(Dataset &training, unsigned int K)
{
	printf("Allocation Matrix %dx%d...\n", K , training.dimension());

	Mat samples(K, training.dimension(), CV_32F);
	Mat responses(K, 1, CV_32SC1);

	for(int i = 0; i < K; i++) {
        //responses.at<int>(i, 0) = training.getSamples()[i].y;
		unsigned int idx = i;//sim.doc_id;
	    responses.at<double>(i, 0) = training.getSamples()[idx].y;
		int *ptr = responses.ptr<int>(i);
		ptr[0] = training.getSamples()[idx].y;

        std::map<unsigned int, double>::iterator it;
		for(it = training.getSamples()[idx].features.begin(); it != training.getSamples()[idx].features.end(); ++it){
			unsigned int term_id = it->first;
			double term_cout = it->second;
			//cout << term_cout << endl;
			float *ptr =  samples.ptr<float>(i);
			ptr[term_id] = (1 + log(term_cout)) * log((double)training.size() / float(max(1, training.getIdf(term_id))));
		}
    }


	return TrainData::create(samples, ROW_SAMPLE, responses);
}

void tccpPrepareTrainSamples(RF* rf, Dataset &training, unsigned int K)
{
	printf("Allocation Matrix %dx%d...\n", K , training.dimension());

	vector<DTDocument> samples;

	for(int i = 0; i < K; i++) {
        DTDocument *doc = new DTDocument(); //samples.push_back(doc);

        unsigned int idx = i;//sim.doc_id;
		
		doc->set_id(Utils::toString(idx));
		doc->set_class(Utils::toString(training.getSamples()[idx].y));

        std::map<unsigned int, double>::iterator it;
		for(it = training.getSamples()[idx].features.begin(); it != training.getSamples()[idx].features.end(); ++it){
			unsigned int term_id = it->first;
			double term_count = it->second;
			doc->insert_term(term_id, (1 + log(term_count)) * log((double)training.size() / float(max(1, training.getIdf(term_id)))));
		
		}

		rf->add_document(doc);
    }

    printf("finish...\n");

}

Mat prepareSample(std::map<unsigned int, double> test_features, Dataset &training){
	Mat testSample( 1, training.dimension(), CV_32F);

	std::map<unsigned int, double>::iterator it;
	for(it = test_features.begin(); it != test_features.end(); ++it){
		unsigned int term_id = it->first;
		double term_count = it->second;

		float *ptr = testSample.ptr<float>(0);
		ptr[term_id] = (1 + log(term_count)) * log((double)training.size() / float(max(1, training.getIdf(term_id))));
	}

	return testSample;
}

DTDocument * tcppPrepareSample(std::map<unsigned int, double> test_features, Dataset &training){
	DTDocument * doc = new DTDocument();

	std::map<unsigned int, double>::iterator it;
	for(it = test_features.begin(); it != test_features.end(); ++it){
		unsigned int term_id = it->first;
		double term_count = it->second;

		doc->insert_term(term_id, term_count * log((double)training.size() / float(max(1, training.getIdf(term_id)))));
	}
	return doc;
}

int main(int argc, char const *argv[])
{
	// Random Forest object
	Ptr<RTrees> randomForest = RTrees::create();
	
	randomForest = RTrees::create();
    randomForest->setMaxDepth(INT_MAX);
    /*randomForest->setMinSampleCount(2);
    randomForest->setRegressionAccuracy(0);
    randomForest->setUseSurrogates(false);
    randomForest->setMaxCategories(15);
    randomForest->setPriors(Mat());*/
    //randomForest->setCalculateVarImportance(true);
    //randomForest->setActiveVarCount(6);
    randomForest->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 0.01));

	Dataset training, test_set;
	training.loadGtKnnFormat("lib/gtknn/data/treino.dat");
	printf("Training set - Num samples : %d, Dimension : %d, Num classes : \n", (int)training.size(), training.dimension());

	
	RF * rf = new RF(0, 1.0, 200);
	  
	rf->set_doc_delete(false);
	
	tccpPrepareTrainSamples(rf, training, training.size());

	printf("Training...\n");
	rf->build();	
	
	test_set.loadGtKnnFormat("lib/gtknn/data/teste.dat");
	printf("Test set - Num samples : %d, Dimension : %d, Num classes : \n", (int)test_set.size(), test_set.dimension());
	
	int correct_cosine = 0, wrong_cosine = 0;
	for (int i = 0; i < test_set.getSamples().size(); ++i)
    {
        Scores<double> similarities = rf->classify(tcppPrepareSample(test_set.getSamples()[i].features, training));
    	float guessed_class = atoi(similarities.top().class_name.c_str());

		printf("Guessed class : %f - Real class : %d\n", guessed_class, test_set.getSamples()[i].y);
        if(std::abs(guessed_class - test_set.getSamples()[i].y) <= FLT_EPSILON) {
            correct_cosine++;   
        } else {
            wrong_cosine++;
        }
    }


    delete rf;

    printf("Cosine similarity\n");
    printf("Correct: %d Wrong: %d\n", correct_cosine, wrong_cosine);
    printf("Accuracy: %lf%%\n\n", double(correct_cosine) / double(test_set.size()));

	return EXIT_SUCCESS	;
}
