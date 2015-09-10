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
	int y;
	std::string labels;

};

class Dataset{
	
	public:

		Dataset() : dim(0)
		{}
		~Dataset();

		void loadSVMlightFormat(const char* input);
		void loadGtKnnFormat(const char* input);
		std::vector<sample>& getSamples();
		size_t size();
		int dimension();
		int getIdf(int term_id);

	private:
		std::vector<sample> samples;
		std::map<int, int> idf;
		int dim;

		void string_tokenize(const std::string &str,
                       std::vector<std::string> &tokens,
                       const std::string &delimiters = " ");		

		int get_class(std::string token);

};

#endif
