#include <iostream>

#include "lib/broof/rf.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
	RF * broof = new RF(0, atof(argv[5]), atoi(argv[4]));
	  
	broof->train(argv[1]);

	broof->set_output_file(argv[3]);
	broof->test(argv[2]);
    
	return EXIT_SUCCESS	;
}
