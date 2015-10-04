#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

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

int main(int argc, char * argv[]){
  std::map<std::string, bool> Docs;
  std::map<std::string, unsigned int> offsets;
  
  std::ifstream dataset_a(argv[1]);

  if(dataset_a){
      std::string line;
      unsigned int of = dataset_a.tellg();
      while (dataset_a >> line) {
        std::vector<std::string> tokens;
        string_tokenize(line, tokens, ";");

        if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return 0;

        std::string doc_id = tokens[0];
        Docs.insert(std::pair<std::string, bool>(doc_id, false));
        offsets.insert(std::pair<std::string, unsigned int>(doc_id, of));
        of = dataset_a.tellg();
      }
 
      std::ifstream dataset_b(argv[2]);
      if(dataset_b){
        while (dataset_b >> line) {
	  std::vector<std::string> tokens;
          string_tokenize(line, tokens, ";");

          if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return 0;

          std::string doc_id = tokens[0];
          
          std::map<std::string, bool>::iterator it = Docs.find(doc_id);
          if(it != Docs.end())
            it->second = true;
          
        }
        dataset_b.close();

        dataset_a.clear();
        dataset_a.seekg(0, std::ios::beg);

        std::map<std::string, bool>::iterator it = Docs.begin();

        while(it != Docs.end()){
          if(it->second){
            dataset_a.seekg(offsets[it->first]);
            dataset_a >> line;
            std::cout << line << std::endl;
          }
          ++it;
        }
        dataset_a.close();
      }
  }

}
