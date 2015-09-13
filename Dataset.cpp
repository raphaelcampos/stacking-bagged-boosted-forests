#include "Dataset.h"

Dataset::~Dataset(){
	samples.clear();
	idf.clear();
	doc_per_class.clear();
}

void Dataset::loadSVMlightFormat(const char* input){
	std::ifstream file(input);
	if (file) {
		samples.clear();

		std::string ln;
		while (std::getline(file, ln)) {
		  sample smp;
		  std::vector<std::string> tokens;
		  string_tokenize(ln, tokens, " ");
		  
		  // input_format: class t1:f1 t2:f2 ...
		  for (unsigned int i = 1; i < tokens.size(); i++) {
		    std::vector<std::string> pair;
		    string_tokenize(tokens[i], pair, ":");
		    int term_id = atoi(pair[0].data());
		    float term_count = atof(pair[1].data());
			smp.features[term_id] = term_count;

		    dim = std::max(dim, term_id + 1);
			if(term_count > 0){
                                idf[term_id] += 1;
                        }

		  }
		  
		  smp.y = atof(tokens[0].data());
		  samples.push_back(smp);
			
			doc_per_class[smp.y]++;	
		}
	}
	else {
		std::cerr << "Failed to open input file." << std::endl;
		exit(1);
	}
}

void Dataset::loadGtKnnFormat(const char* input){
	std::ifstream file(input);
	if (file) {
		int num_docs = 1;
		samples.clear();
		std::string line;
		while(!file.eof()) {
	        std::getline(file, line);
	        if(line == "") continue;

	        int doc_id = num_docs++;
	        std::vector<std::string> tokens;
	        string_tokenize(line, tokens, " ");

	        // input_format: docId;class=C;t1;f1;t2;f2...
	        sample smp;
	        for(int i = 2, size = tokens.size(); i + 1 < size; i+=2) {
	            int term_id = atoi(tokens[i].c_str());
	            int term_count = atoi(tokens[i+1].c_str());

	            dim = std::max(dim, term_id + 1);
	            
	            smp.features[term_id] = term_count;
	        
			if(term_count > 0){
				idf[term_id] += 1;
			}
		}
	        
	        smp.y = get_class(tokens[1]);
	        samples.push_back(smp);
	        doc_per_class[smp.y]++;
	    }

		file.close();
	}
	else {
		std::cerr << "Failed to open input file." << std::endl;
		exit(1);
	}
}

std::vector<sample>& Dataset::getSamples(){
	return samples;
}

size_t Dataset::size(){
	return samples.size();
}

int Dataset::dimension(){
	return dim;
}

int Dataset::getIdf(int term_id){
	std::map<int, int>::iterator it = idf.find(term_id);
	if(it != idf.end())
		return it->second;
	else
		return 0;
}

void Dataset::randomize_samples(){
	long n = samples.size()-1;
    while (n > 0)
    {
        // pick a random index to swap into t[n]
        const unsigned long idx = std::rand()%(n+1);

        // swap our randomly selected index into the n position
        std::swap(samples[idx], samples[n]);

        --n;
    }
}

std::pair<Dataset, Dataset> Dataset::split(float portion){
	Dataset data1,data2;
	data1.samples.insert(data1.samples.begin(), samples.begin() + int(samples.size() * portion), samples.end());
	data2.samples.insert(data2.samples.begin(), samples.begin(), samples.begin() + int(samples.size() * portion));

	return std::make_pair(data2, data1);
}

void Dataset::string_tokenize(const std::string &str,
                       std::vector<std::string> &tokens,
                       const std::string &delimiters) {
    std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    std::string::size_type pos = str.find_first_of(delimiters, lastPos);
    while (std::string::npos != pos || std::string::npos != lastPos) {
      tokens.push_back(str.substr(lastPos, pos - lastPos));
      lastPos = str.find_first_not_of(delimiters, pos);
      pos = str.find_first_of(delimiters, lastPos);
    }
}

int Dataset::get_class(std::string token) {
    std::vector<std::string> class_tokens;
    string_tokenize(token, class_tokens, "=");

    if(class_tokens.size() == 1) {
        return atoi(class_tokens[0].c_str());
    } else {
        return atoi(class_tokens[1].c_str());
    }
}
