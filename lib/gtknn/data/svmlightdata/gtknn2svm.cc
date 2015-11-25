#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <map>

void string_tokenize(const std::string &str,
                   std::vector<std::string> &tokens,
                   const std::string &delimiters = " ") {
	std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
	std::string::size_type pos = str.find_first_of(delimiters, lastPos);
	while (std::string::npos != pos || std::string::npos != lastPos) {
		tokens.push_back(str.substr(lastPos, pos - lastPos));
		lastPos = str.find_first_not_of(delimiters, pos);
		pos = str.find_first_of(delimiters, lastPos);
	}
}

int get_class(std::string token) {
    std::vector<std::string> class_tokens;
    string_tokenize(token, class_tokens, "=");

    if(class_tokens.size() == 1) {
        return atoi(class_tokens[0].c_str());
    } else {
        return atoi(class_tokens[1].c_str());
    }
}

void svm2gtknn(const char *input) {
  std::ifstream file(input);
  if (file) {
    std::string ln;
    unsigned int id = 1;
    while (std::getline(file, ln)) {
      std::vector<std::string> tokens;
      string_tokenize(ln, tokens, " ");
      // input_format: class t1:f1 t2:f2 ...
      // output_format: docId;class=C;t1;f1;t2;f2...
      std::map<int, double> tfs;
      for (unsigned int i = 1; i < tokens.size(); i++) {
        std::vector<std::string> pair;
        string_tokenize(tokens[i], pair, ":");
        tfs[atoi(pair[0].data())] = atof(pair[1].data());
      }
      
      std::stringstream data;
      data << id << " " << tokens[0];
      std::map<int, double>::const_iterator it = tfs.begin();
      while (it != tfs.end()) {
        data << " " << it->first << " " << it->second;
        ++it;
      }
      std::cout << data.str() << std::endl; id++;
    }
  }
  else {
    std::cerr << "Failed to open input file." << std::endl;
    exit(1);
  }

}

void gtknn2svm(const char* input){
	std::ifstream file(input);
	if (file) {
		int num_docs = 1;

		std::string line;
		while(!file.eof()) {
	        std::getline(file, line);
	        if(line == "") continue;

	        int doc_id = num_docs++;
	        std::vector<std::string> tokens;
	        string_tokenize(line, tokens, " ");

	        // input_format: docId;class=C;t1;f1;t2;f2...
	        // output_format: class t1:f1 t2:f2 ...
	        std::cout <<  get_class(tokens[1]) << " ";
	        for(int i = 2, size = tokens.size(); i + 1 < size; i+=2) {
	            int term_id = atoi(tokens[i].c_str());
	            int term_count = atoi(tokens[i+1].c_str());

	            std::cout << term_id << ":" << term_count << " ";
			}
	        
	        std::cout << std::endl;

	    }

		file.close();
	}
	else {
		std::cerr << "Failed to open input file." << std::endl;
		exit(1);
	}
}

int main(int argc, char **argv) {
  char *input;
  if (!(input = argv[1])) {
    std::cerr << "Usage: " << argv[0] << " [input]" << std::endl;
    exit(1);
  }
  gtknn2svm(input);
  return 0;
}
