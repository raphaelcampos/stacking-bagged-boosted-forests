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
		/**
		 * Default constructor.
		 */
		Dataset() : dim(0) {}
		
		/**
		 * Destructor.
		 */
		~Dataset();

		/**
		 * Loads dataset from a svmlight format.
		 * \param input - File location.
		 */
		void loadSVMlightFormat(const char* input);

		/**
		 * Loads dataset from a GTkNN format.
		 * \param input - File location.
		 */
		void loadGtKnnFormat(const char* input);

		/**
		 * Returns the vector containg the dataset's samples.
		 */
		std::vector<sample>& getSamples();

		/**
		 * Dataset size.
		 */
		size_t size();

		/**
		 * Dataset features dimensionality.
		 */
		int dimension();

		/**
		 * Returns idf given a term_id
		 * \param  term_id - Vocabulary term identification
		 */
		int getIdf(int term_id);

		/**
		 * 
		 */
    	std::vector<sample>::const_iterator sample_begin() const {return samples.begin();}
    	
    	/**
		 * 
		 */
    	std::vector<sample>::const_iterator sample_end() const{return samples.end();}

    	/**
		 * 
		 */
    	std::vector<sample>::iterator sample_begin() {return samples.begin();}
    	
    	/**
		 * 
		 */
    	std::vector<sample>::iterator sample_end() {return samples.end();}

    	/**
		 * Randomize dataset samples.
		 */
		void randomize_samples();  	

		/**
		 * Splits the dataset in two disjoint dataset given a portion.
		 */
		std::pair<Dataset, Dataset> split(float portion);

		/**
		 * Total number of classes in the collection.
		 */
		int num_class(){
			return doc_per_class.size();
		}

		std::map<int, int> doc_per_class;

	private:
		std::vector<sample> samples;
		
		std::map<int, int> idf;
		int dim;

		/**
		 * Tokenize a given string
		 * \param str - string to be tokenized
		 * \param tokens - Mutable list of string. It will contain str tokens
		 * \param delimiters - Delimiter to separe tokens
		 */
		void string_tokenize(const std::string &str,
                       std::vector<std::string> &tokens,
                       const std::string &delimiters = " ");		

		/**
		 * Recognize class given a class token from GTkNN format
		 * \param  token - Class token
		 */
		int get_class(std::string token);

};

#endif
