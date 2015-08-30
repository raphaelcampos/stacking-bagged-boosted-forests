#include "Dataset.h"

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
		  // output_format: docId;class=C;t1;f1;t2;f2...
		  
		  for (unsigned int i = 1; i < tokens.size(); i++) {
		    std::vector<std::string> pair;
		    string_tokenize(tokens[i], pair, ":");
		    smp.features[atoi(pair[0].data())] = atof(pair[1].data());
		  }
		  
		  smp.y = atof(tokens[0].data());
		  samples.push_back(smp);
		}
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