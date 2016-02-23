// OpenCV
#include <iostream>

#include "tcpp/tree.hpp"

#include "tcpp/rf_knn.hpp"
#include "tcpp/rf.hpp"
#include "tcpp/knn.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
	RF_KNN * knn_rf = new RF_KNN(0, 0.03, atof(argv[5]), atoi(argv[4]));
	  
	knn_rf->train(argv[1]);

	knn_rf->set_output_file(argv[3]);
	knn_rf->test(argv[2]);
	

	/*RF * rf = new RF(0, 0.03, 200);
	  
	rf->train(argv[1]);

	rf->set_output_file(argv[3]);
	rf->test(argv[2]);
	*/

	/*knn * kNN = new knn();
	  
	kNN->train(argv[1]);

	kNN->set_output_file(argv[3]);
	kNN->test(argv[2]);*/
	
    
	return EXIT_SUCCESS	;
}
