#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstdlib>
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
  std::map<std::string, std::set<std::string> > TermSet;
  std::map<std::string, std::set<std::string> > DocSet;
  std::map<std::string, unsigned int> offsets;
  
  std::ifstream dataset(argv[1]);
  std::string dir_name = argv[2];
  mkdir(dir_name.data(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

  if (dataset) {
      std::cerr << "Building terms ..." << std::endl;
      std::string line;
      unsigned int of = dataset.tellg();
      while (dataset >> line) {
        std::vector<std::string> tokens;
        string_tokenize(line, tokens, ";");

        if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return 0;

        std::string doc_id = tokens[0];
        std::string doc_class = tokens[1];
        for (size_t i = 2; i < tokens.size()-1; i+=2) {
          unsigned int tf = atoi(tokens[i+1].data());
          std::string term_id = tokens[i];
          std::map<std::string, std::set<std::string> >::iterator it = TermSet.find(term_id);
          if(it == TermSet.end()){
            std::set<std::string> docs;
            it = TermSet.insert(std::pair<std::string, std::set<std::string> >(term_id, docs)).first;
          }
          it->second.insert(doc_id);
        }
        offsets.insert(std::pair<std::string, unsigned int>(doc_id, of));
        of = dataset.tellg();
      }
      dataset.clear();
      dataset.seekg(0, std::ios::beg);
      
      std::cerr << "Building docs ..." << std::endl;
      while (dataset >> line) {
        static int num_doc = 1;
        std::cerr << num_doc << "\r";
        num_doc++;
        std::vector<std::string> tokens;
        string_tokenize(line, tokens, ";");

        if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return 0;

        std::string doc_id = tokens[0];
        std::string doc_class = tokens[1];

        std::map<std::string, std::set<std::string> >::iterator it_doc = DocSet.find(doc_id);          
        if(it_doc == DocSet.end()) it_doc = DocSet.insert(std::pair<std::string, std::set<std::string> >(doc_id, std::set<std::string>())).first;

        for (size_t i = 2; i < tokens.size()-1; i+=2) {
          unsigned int tf = atoi(tokens[i+1].data());
          std::string term_id = tokens[i];
          std::map<std::string, std::set<std::string> >::iterator it = TermSet.find(term_id);
          
          std::set<std::string>::iterator it_td = it->second.begin();
          while(it_td != it->second.end()){
            it_doc->second.insert(*it_td);
            ++it_td;
          }
        }
      }
      dataset.clear();

      std::cerr << "Generating datasets ..." << std::endl;
      
      std::map<std::string, std::set<std::string> >::iterator it_docs = DocSet.begin();
      while(it_docs != DocSet.end()){
        std::string out_name = dir_name + "/doc" + it_docs->first;
        std::ofstream out_sets(out_name.data());
        std::set<std::string>::iterator it_i = it_docs->second.begin();
        while(it_i != it_docs->second.end()){
          if(it_docs->first.compare(*it_i) == 0) {
            ++it_i;
            continue;
          }

          dataset.seekg(offsets[*it_i]);
          dataset >> line;
          out_sets << line << std::endl;
          ++it_i;
        }
        out_sets.close();
        ++it_docs;
      }
      dataset.close();
  }
  
}

  
