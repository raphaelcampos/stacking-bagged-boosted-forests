#ifndef _DATA_SET__
#define _DATA_SET__

#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <map>

struct sample
{
	std::map<unsigned int, double> features;
	double y;
	std::string labels;
};

class Dataset{
	
	public:
		void loadSVMlightFormat(const char* input);
		std::vector<sample>& getSamples();
		size_t size();

	private:
		std::vector<sample> samples;

		void string_tokenize(const std::string &str,
                       std::vector<std::string> &tokens,
                       const std::string &delimiters = " ");		

};

#endif