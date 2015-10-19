#include <iostream>

#include "lib/broof/rf_bst.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
	RF_BOOST * broof = new RF_BOOST(0, 0.004047256, atoi(argv[4]));
	  
	broof->train(argv[1]);

	broof->set_output_file(argv[3]);
	broof->test(argv[2]);
    
	return EXIT_SUCCESS	;
}
