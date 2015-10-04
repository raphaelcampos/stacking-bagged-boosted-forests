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
  
  std::string dir_name = argv[2];
  mkdir(dir_name.data(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

  std::ifstream teste(argv[1]);

  if(teste){
    std::string line;
    while(teste >> line){
      std::vector<std::string> tokens;
      string_tokenize(line, tokens, ";");

      if ((tokens.size() < 4) || (tokens.size() % 2 != 0)) return 0;

      std::string doc_id = tokens[0];
      std::string out_name = dir_name + "/doc" + doc_id;
      std::ofstream out_doc(out_name.data());

      out_doc << line << std::endl;

    }
    teste.close();
  }
}
