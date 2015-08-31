#ifndef _LazyNN_RF__
#define _LazyNN_RF__

#include "Dataset.h"

#include <map>

class LazyNN_RF{
	
	public:
		virtual void train(Dataset &data) = 0;
		virtual int classify(std::map<unsigned int, double> test_sample, int K) = 0;
};

#endif