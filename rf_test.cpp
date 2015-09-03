// OpenCV
#include <opencv2/opencv.hpp>
#include <iostream>
#include "Dataset.h"

using namespace cv;
using namespace cv::ml;

using namespace std;

Ptr<TrainData> prepareTrainSamples(Dataset &training, unsigned int K)
{
	printf("Allocation Matrix %dx%d...\n", K , training.dimension());

	Mat samples(K, training.dimension(), CV_32F);
	Mat responses(K, 1, CV_32F);

	for(int i = 0; i < K; i++) {
        //responses.at<int>(i, 0) = training.getSamples()[i].y;
		unsigned int idx = i;//sim.doc_id;
	    responses.at<double>(i, 0) = training.getSamples()[idx].y;
		float *ptr = responses.ptr<float>(i);
		ptr[0] = training.getSamples()[idx].y;

        std::map<unsigned int, double>::iterator it;
		for(it = training.getSamples()[idx].features.begin(); it != training.getSamples()[idx].features.end(); ++it){
			unsigned int term_id = it->first;
			double term_cout = it->second;
			//cout << term_cout << endl;
			float *ptr =  samples.ptr<float>(idx);
			ptr[term_id] = term_cout;
		}
    }

	return TrainData::create(samples, ROW_SAMPLE, responses);
}

int main(int argc, char const *argv[])
{
	
	// Random Forest object
	Ptr<RTrees> randomForest = RTrees::create();
	/*randomForest->setMinSampleCount(2);
    randomForest->setRegressionAccuracy(0.f);
    randomForest->setUseSurrogates(false);
    randomForest->setMaxCategories(16);
    randomForest->setPriors(Mat());
    randomForest->setCalculateVarImportance(false);
    randomForest->setActiveVarCount(1);
    randomForest->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 5, 0));*/


	Dataset training, test;
	training.loadGtKnnFormat("lib/gtknn/data/treino.dat");
	
	printf("Training set - Num samples : %d, Dimension : %d, Num classes : \n", (int)training.size(), training.dimension());

	printf("Training...\n");
	randomForest->train(prepareTrainSamples(training, training.size()));

	test.loadGtKnnFormat("lib/gtknn/data/teste.dat");
	printf("Test set - Num samples : %d, Dimension : %d, Num classes : \n", (int)test.size(), test.dimension());
	


	return EXIT_SUCCESS	;
}
