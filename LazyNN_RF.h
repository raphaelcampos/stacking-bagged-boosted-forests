#ifndef _LazyNN_RF__
#define _LazyNN_RF__

#include "Dataset.h"

#include <map>

class LazyNN_RF{
	
	public:
		virtual void train(Dataset &data) = 0;
		virtual int classify(const std::map<unsigned int, float> &test_sample, int K) = 0;
};

#endif