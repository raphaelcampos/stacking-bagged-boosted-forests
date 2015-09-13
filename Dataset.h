#ifndef _DATA_SET__
#define _DATA_SET__

#include <utility> 
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <map>


struct sample
{
	std::map<unsigned int, float> features;
	int y;
};

class Dataset{
	
	public:

		Dataset() : dim(0) {}
		~Dataset();

		void loadSVMlightFormat(const char* input);
		void loadGtKnnFormat(const char* input);
		std::vector<sample>& getSamples();
		size_t size();
		int dimension();
		int getIdf(int term_id);

    	std::vector<sample>::const_iterator sample_begin() const {return samples.begin();}
    	std::vector<sample>::const_iterator sample_end() const{return samples.end();}

    	std::vector<sample>::iterator sample_begin() {return samples.begin();}
    	std::vector<sample>::iterator sample_end() {return samples.end();}

		void randomize_samples();  	

		std::pair<Dataset, Dataset> split(float portion);

		int num_class(){
			return doc_per_class.size();
		}

		std::map<int, int> doc_per_class;

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
