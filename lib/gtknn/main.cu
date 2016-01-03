/*********************************************************************
11	
12	 Copyright (C) 2015 by Wisllay Vitrio
13	
14	 This program is free software; you can redistribute it and/or modify
15	 it under the terms of the GNU General Public License as published by
16	 the Free Software Foundation; either version 2 of the License, or
17	 (at your option) any later version.
18	
19	 This program is distributed in the hope that it will be useful,
20	 but WITHOUT ANY WARRANTY; without even the implied warranty of
21	 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
22	 GNU General Public License for more details.
23	
24	 You should have received a copy of the GNU General Public License
25	 along with this program; if not, write to the Free Software
26	 Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
27	
28	 ********************************************************************/

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <vector>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <string>
#include <sstream>


#include "structs.cuh"
#include "utils.cuh"
#include "inverted_index.cuh"
#include "knn.cuh"
#include "cuda_distances.cuh"
#include <cuda.h>

#include <map>
using namespace std;

struct FileStats {
    int num_docs;
    int num_terms;

    std::map<int, int> doc_to_class;

    FileStats() : num_docs(0), num_terms(0) {}
};

FileStats readTrainingFile(std::string &file, std::vector<Entry> &entries);
void readTest(std::string &filename, vector<string>& inputs);
void updateStatsMaxFeatureTest(FileStats &stats, vector<string>& inputs);

void processTestFile(InvertedIndex &index, FileStats &stats, vector<string>& input, 
	string &file, int K, std::string distance, stringstream &fileout, stringstream &filedists);
bool makeQuery(InvertedIndex &inverted_index, FileStats &stats, std::string &line, int K,
	void(*distance)(InvertedIndex, Entry*, int*, cuSimilarity*, int D), stringstream &fileout,
	stringstream &filedists, DeviceVariables *dev_vars);

void makeQuery(InvertedIndex &inverted_index, float* data, int* indices, int begin, int end, int K,
	void(*distance)(InvertedIndex, Entry*, int*, cuSimilarity*, int D), int* idxs, DeviceVariables *dev_vars);

int get_class(std::string token);


/**
 * Receives as parameters the training file name and the test file name
 */

static int num_tests = 0;
static int correct_l2 = 0, correct_cosine = 0, correct_l1 = 0;
static int wrong_l2 = 0, wrong_cosine = 0, wrong_l1 = 0;
int biggestQuerySize = -1;


int main(int argc, char **argv) {
   
	if (argc != 8) {
		std::cerr << "Wrong parameters. Correct usage: <executable> <training_file> <test_file> <k> <cosine | l2 | l1> <output_classifications_file> <output_distances_file> <gpu_number>" << std::endl;
		exit(1);
	}
	
	int gpuNum;
	cudaGetDeviceCount(&gpuNum);

	if (gpuNum > atoi(argv[7])){
		gpuNum = atoi(argv[7]);
		if (gpuNum < 1)
			gpuNum = 1;
	}
	std::cerr << "Using " << gpuNum << "GPUs" << endl;

    omp_set_num_threads(gpuNum);

	//truncate output files
	ofstream ofsf1(argv[5], ofstream::trunc);
	ofstream ofsf2(argv[6], ofstream::trunc);
	ofsf1.close();
	ofsf2.close();

	ofstream ofsfileoutput(argv[5], ofstream::out | ofstream::app); 
	ofstream ofsfiledistances(argv[6], ofstream::out | ofstream::app);

	vector<string> inputs;// to read the whole test file in memory
	double starts, ends;

	std::string trainingFileName(argv[1]);
	std::string testFileName(argv[2]);

	printf("Reading files...\n");
	std::vector<Entry> entriess;
	starts = gettime();
	FileStats statss = readTrainingFile(trainingFileName, entriess);
	readTest(testFileName, inputs);
	updateStatsMaxFeatureTest(statss, inputs);
	ends = gettime();

	printf("time taken: %lf seconds\n", ends - starts);
	//fprintf(stderr,"sizeof Entry %u , sizeof cuSimilarity %u\n",sizeof(Entry), sizeof(cuSimilarity));
	
	vector<stringstream*> outputClassString, outputDistanceString;
	//Each thread builds an output string, so it can be flushed at once at the end of the program
	for (int i = 0; i < gpuNum; i++){
		outputClassString.push_back(new stringstream);
		outputDistanceString.push_back(new stringstream);
	}

#pragma omp parallel shared(inputs) shared(outputClassString) shared(outputDistanceString)
    {
    	int cpuid = omp_get_thread_num();
    	cudaSetDevice(cpuid);

		int k = atoi(argv[3]);
		std::string distanceFunction(argv[4]);

		std::vector<Entry>& entries = entriess;
		double start, end;

		FileStats stats = statss;

		start = gettime();
		InvertedIndex inverted_index = make_inverted_index(stats.num_docs, stats.num_terms, entries);
		end = gettime();

		#pragma omp single nowait
		printf("Total time taken for insertion: %lf seconds\n", end - start);	
				
		processTestFile(inverted_index, stats, inputs, testFileName, k,
			distanceFunction, *outputClassString[cpuid], *outputDistanceString[cpuid]);

		gpuAssert(cudaDeviceReset());
		
	}
	starts = gettime();
	for (int i = 0; i < gpuNum; i++){
		ofsfileoutput << outputClassString[i]->str();
		ofsfiledistances << outputDistanceString[i]->str();
	}
	ends = gettime();

	printf("time taken to write output: %lf seconds\n", ends - starts);

	ofsfileoutput.close();
	ofsfiledistances.close();
    return 0;
}

FileStats readTrainingFile(std::string &filename, std::vector<Entry> &entries) {
    std::ifstream input(filename.c_str());
    std::string line;

    FileStats stats;

    while(!input.eof()) {
        std::getline(input, line);
        if(line == "") continue;

        int doc_id = stats.num_docs++;
        std::vector<std::string> tokens = split(line, ' ');

        stats.doc_to_class[doc_id] = get_class(tokens[1]);

        for(int i = 2, size = tokens.size(); i + 1 < size; i+=2) {
            int term_id = atoi(tokens[i].c_str());
            int term_count = atoi(tokens[i+1].c_str());
            stats.num_terms = std::max(stats.num_terms, term_id + 1);
            entries.push_back(Entry(doc_id, term_id, term_count));
        }
    }

    input.close();

    return stats;
}

void initDeviceVariables(DeviceVariables *dev_vars, int K, int num_docs){
	
	int KK = 1;    //Obtain the smallest power of 2 greater than K (facilitates the sorting algorithm)
    while(KK < K) KK <<= 1;
    
    dim3 grid, threads;
    
    get_grid_config(grid, threads);
		
	gpuAssert(cudaMalloc(&dev_vars->d_dist, num_docs * sizeof(cuSimilarity)));
	gpuAssert(cudaMalloc(&dev_vars->d_nearestK, KK * grid.x * sizeof(cuSimilarity)));
	gpuAssert(cudaMalloc(&dev_vars->d_query, biggestQuerySize * sizeof(Entry)));
	gpuAssert(cudaMalloc(&dev_vars->d_index, biggestQuerySize * sizeof(int)));
	gpuAssert(cudaMalloc(&dev_vars->d_count, biggestQuerySize * sizeof(int)));
	gpuAssert(cudaMalloc(&dev_vars->d_qnorms, 2 * sizeof(float)));
	
}

void freeDeviceVariables(DeviceVariables *dev_vars){
	gpuAssert(cudaFree(dev_vars->d_dist));
	gpuAssert(cudaFree(dev_vars->d_nearestK));
	gpuAssert(cudaFree(dev_vars->d_query));
	gpuAssert(cudaFree(dev_vars->d_index));
	gpuAssert(cudaFree(dev_vars->d_count));
	gpuAssert(cudaFree(dev_vars->d_qnorms));	
}

void freeDeviceVariables(DeviceVariables *dev_vars, InvertedIndex &index){
	gpuAssert(cudaFree(dev_vars->d_dist));
	gpuAssert(cudaFree(dev_vars->d_nearestK));
	gpuAssert(cudaFree(dev_vars->d_query));
	gpuAssert(cudaFree(dev_vars->d_index));
	gpuAssert(cudaFree(dev_vars->d_count));
	gpuAssert(cudaFree(dev_vars->d_qnorms));

	gpuAssert(cudaFree(index.d_count));
	gpuAssert(cudaFree(index.d_index));
	gpuAssert(cudaFree(index.d_inverted_index));
	gpuAssert(cudaFree(index.d_norms));
	gpuAssert(cudaFree(index.d_normsl1));
	
}

extern "C"
int* kneighbors(InvertedIndex* index, int K, float* data, int* indices, int* indptr, int nnz, int n, int n_gpu){
	
	std::string distance = "cosine"; 

	int gpuNum;
	cudaGetDeviceCount(&gpuNum);

	if (gpuNum > n_gpu){
		gpuNum = n_gpu;
		if (gpuNum < 1)
			gpuNum = 1;
	}

	std::vector<std::pair<int, int> > intervals;
	
	biggestQuerySize = -1;
	for (int i = 0; i < n - 1; ++i)
	{	
		intervals.push_back(std::make_pair(indptr[i], indptr[i + 1]));
		biggestQuerySize = std::max(biggestQuerySize, indptr[i + 1] - indptr[i]);
	}

	printf("biggestQuerySize : %d\n", biggestQuerySize);

	int* idxs = new int[(n-1)*K];

	std::cerr << "Using " << gpuNum << "GPUs" << endl;

	omp_set_num_threads(gpuNum);
	
	#pragma omp parallel shared(intervals) shared(data) shared(indices) shared(index) shared(idxs)
	{
		int cpuid = omp_get_thread_num();
    		cudaSetDevice(cpuid);
		int num_test_local = 0, i;

		printf("thread : %d\n", cpuid);

    	DeviceVariables dev_vars;
	
		initDeviceVariables(&dev_vars, K, index[cpuid].num_docs);

		double start = gettime();

    		#pragma omp for
		for (i = 0; i < intervals.size(); ++i)
		{
			num_test_local++;
			
			if(distance == "cosine" || distance == "both") {
				makeQuery(index[cpuid], data, indices, intervals[i].first, intervals[i].second, K, CosineDistance, &idxs[i*K], &dev_vars);
	        	}

	        	if(distance == "l2" || distance == "both") {
	        		makeQuery(index[cpuid], data, indices, intervals[i].first, intervals[i].second, K, EuclideanDistance, &idxs[i*K], &dev_vars);
			}

			if(distance == "l1" || distance == "both") {
				makeQuery(index[cpuid], data, indices, intervals[i].first, intervals[i].second, K, ManhattanDistance, &idxs[i*K], &dev_vars);
			}
		}
		
		printf("num tests in thread %d: %d\n", omp_get_thread_num(), num_test_local);

		#pragma omp barrier
		double end = gettime();

		#pragma omp master
		printf("Time to process: %lf seconds\n", end - start);
	
		freeDeviceVariables(&dev_vars);
	}

	return idxs;
}

std::vector<Entry> csr2entries(float* data, int* indices, int* indptr, int nnz, int n){
	std::vector<Entry> entries;
	for (int i = 0; i < n-1; ++i)
	{
		int begin = indptr[i], end = indptr[i+1];
		for(int j = begin; j < end; ++j) {
	        int term_id = indices[j];
	        int term_count = data[j];

	        entries.push_back(Entry(i, term_id, term_count));
	    }
	}
	return entries;
}

InvertedIndex* make_inverted_indices(int num_docs, int num_terms, std::vector<Entry> entries, int n_gpu){
	
	int gpuNum;
	cudaGetDeviceCount(&gpuNum);

	if (gpuNum > n_gpu){
		gpuNum = n_gpu;
		if (gpuNum < 1)
			gpuNum = 1;
	}
	
	std::cerr << "Using " << gpuNum << "GPUs" << endl;

    	omp_set_num_threads(gpuNum);

	InvertedIndex* indices = new InvertedIndex[gpuNum];

	#pragma omp parallel shared(entries) shared(indices)
    	{
    		int cpuid = omp_get_thread_num();
    		cudaSetDevice(cpuid);
		printf("thread_idx : %d\n", cpuid);
		double start, end;

		start = gettime();
		indices[cpuid] = make_inverted_index(num_docs, num_terms, entries);
		end = gettime();

		//#pragma omp single nowait
		printf("Total time taken for insertion: %lf seconds\n", end - start);	
	}

	return indices;
}

extern "C"
InvertedIndex* csr_make_inverted_indices(int num_docs, int num_terms, float* data, int* indices, int* indptr, int nnz, int n, int n_gpu){
	std::vector<Entry> entries = csr2entries(data, indices, indptr, nnz, n);

	return make_inverted_indices(num_docs, num_terms, entries, n_gpu);
}

extern "C"
InvertedIndex* make_inverted_indices(int num_docs, int num_terms, Entry * entries, int n_entries, int n_gpu){
	
	int gpuNum;
	cudaGetDeviceCount(&gpuNum);

	if (gpuNum > n_gpu){
		gpuNum = n_gpu;
		if (gpuNum < 1)
			gpuNum = 1;
	}
	
	std::cerr << "Using " << gpuNum << "GPUs" << endl;

    omp_set_num_threads(gpuNum);

	InvertedIndex* indices = new InvertedIndex[n_gpu];

	#pragma omp parallel shared(entries) shared(indices)
    {
    	int cpuid = omp_get_thread_num();
    	cudaSetDevice(cpuid);

		double start, end;

		start = gettime();
		indices[cpuid] = make_inverted_index(num_docs, num_terms, entries, n_entries);
		end = gettime();

		#pragma omp single nowait
		printf("Total time taken for insertion: %lf seconds\n", end - start);	
	}

	return indices;
}

void readTest(std::string &filename, vector<string>& inputs) {
    std::ifstream input(filename.c_str());
    std::string line;    

    while(!input.eof()) {		
        std::getline(input, line);
        if(line == "") continue;
		num_tests++;
		inputs.push_back(line);	
    }
}

void updateStatsMaxFeatureTest(FileStats &stats, vector<string>& inputs) {
  
#pragma omp parallel num_threads(4)
   {
	   int localBiggest = -1;
	   int localNumterms = stats.num_terms;
		#pragma omp for
		for(int i = 0; i < inputs.size(); i++) {     		
			
			std::vector<std::string> tokens = split(inputs[i], ' ');
			
			localBiggest= std::max((int)tokens.size() / 2,  localBiggest);
			for(int i = 2, size = tokens.size(); i + 1 < size; i+=2) {
				int term_id = atoi(tokens[i].c_str());
				localNumterms = std::max(localNumterms, term_id + 1);
			}
		}
		
		#pragma omp critical
		{
			stats.num_terms = std::max(localNumterms, stats.num_terms);
			biggestQuerySize = std::max( biggestQuerySize, localBiggest);
		}
		
	}
}

void processTestFile(InvertedIndex &index, FileStats &stats, vector<string>& input_t, 
	std::string &filename, int K, std::string distance, stringstream &outputfile, stringstream &outputdists) {
     
    int num_test_local = 0, i;

	printf("Processing test file %s...\n", filename.c_str());  
	
	DeviceVariables dev_vars;
	
	initDeviceVariables(&dev_vars, K, index.num_docs);	
	
	double start = gettime();

	#pragma omp for
	for (i = 0; i < input_t.size();i++){
		
		num_test_local++;

		if(distance == "cosine" || distance == "both") {
			if (makeQuery(index, stats, input_t[i], K, CosineDistance, outputfile, outputdists,&dev_vars)) {
				#pragma omp atomic
            			correct_cosine++;
            		} else {
				#pragma omp atomic
                		wrong_cosine++;
            		}
        	}

        	if(distance == "l2" || distance == "both") {

			if (makeQuery(index, stats, input_t[i], K, EuclideanDistance, outputfile, outputdists,&dev_vars)) {
				#pragma omp atomic
            			correct_l2++;
            		} else {
				#pragma omp atomic
            			wrong_l2++;
            		}
        	}

		if(distance == "l1" || distance == "both") {
			if (makeQuery(index, stats, input_t[i], K, ManhattanDistance, outputfile, outputdists,&dev_vars)) {
				#pragma omp atomic
            			correct_l1++;
            		} else {
				#pragma omp atomic
            			wrong_l1++;
            		}
		}
		
		input_t[i].clear();
	}
	
	freeDeviceVariables(&dev_vars, index);

	printf("num tests in thread %d: %d\n", omp_get_thread_num(), num_test_local);

	#pragma omp barrier

	double end = gettime();

	#pragma omp master
	printf("Total num tests %d\n", num_tests);   	

	#pragma omp master
    	{
		printf("Time taken for %d queries: %lf seconds\n\n", num_tests, end - start);

		if(distance == "cosine" || distance == "both") {
			printf("Cosine cuSimilarity\n");
			printf("Correct: %d Wrong: %d\n", correct_cosine, wrong_cosine);
			printf("Accuracy: %lf%%\n\n", double(correct_cosine) / double(num_tests));
		}

		if(distance == "l2" || distance == "both") {
			printf("L2 distance\n");
			printf("Correct: %d Wrong: %d\n", correct_l2, wrong_l2);
			printf("Accuracy: %lf%%\n\n", double(correct_l2) / double(num_tests));
		}

		if(distance == "l1" || distance == "both") {
			printf("L1 distance\n");
			printf("Correct: %d Wrong: %d\n", correct_l1, wrong_l1);
			printf("Accuracy: %lf%%\n\n", double(correct_l1) / double(num_tests));
		}
    }
}


void makeQuery(InvertedIndex &inverted_index, float* data, int* indices, int begin, int end, int K,
	void(*distance)(InvertedIndex, Entry*, int*, cuSimilarity*, int D), int* idxs, DeviceVariables *dev_vars) {

    std::vector<Entry> query;
    for(int i = begin; i < end; ++i) {
        int term_id = indices[i];
        int term_count = data[i];

        query.push_back(Entry(0, term_id, term_count));
    }

    //Creates an empty document if there are no terms
    if(query.empty()) {
        query.push_back(Entry(0, 0, 0));
    }

    cuSimilarity *k_nearest = KNN(inverted_index, query, K, distance, dev_vars);
    std::map<int, int> vote_count;
	std::map<int, int>::iterator it;


	for (int i = 0; i < K; i++) {
		cuSimilarity &sim = k_nearest[i];
		idxs[i] = sim.doc_id;
	}
	
	free(k_nearest);
}

bool makeQuery(InvertedIndex &inverted_index, FileStats &stats, std::string &line, int K,
	void(*distance)(InvertedIndex, Entry*, int*, cuSimilarity*, int D), 
	stringstream &outputfile, stringstream &outputdists, DeviceVariables *dev_vars) {

    std::vector<Entry> query;
    std::vector<std::string> tokens = split(line, ' ');

    int trueclass = get_class(tokens[1]);
    int docid = atoi(tokens[0].c_str());

    for(int i = 2, size = tokens.size(); i + 1 < size; i+=2) {
        int term_id = atoi(tokens[i].c_str());
        int term_count = atoi(tokens[i+1].c_str());

        query.push_back(Entry(0, term_id, term_count));
    }

    //Creates an empty document if there are no terms
    if(query.empty()) {
        query.push_back(Entry(0, 0, 0));
    }

    cuSimilarity *k_nearest = KNN(inverted_index, query, K, distance, dev_vars);
    std::map<int, int> vote_count;
	std::map<int, int>::iterator it;

	for (int i = 0; i < K; i++) {
		cuSimilarity &sim = k_nearest[i];
		vote_count[stats.doc_to_class[sim.doc_id]]++;
		outputdists << sim.distance << " ";
	}

	outputdists << std::endl;
	free(k_nearest);
	
    int guessed_class = -1;
    int max_votes = 0;

    for(it = vote_count.begin(); it != vote_count.end(); it++) {
        if(it->second > max_votes) {
            max_votes = it->second;
            guessed_class = it->first;
        }
    }

	outputfile << docid << " CLASS=" << trueclass << " CLASS=" << guessed_class << ":1" << std::endl;

    return trueclass == guessed_class;
}

int get_class(std::string token) {
    std::vector<std::string> class_tokens = split(token, '=');

    if(class_tokens.size() == 1) {
        return atoi(class_tokens[0].c_str());
    } else {
        return atoi(class_tokens[1].c_str());
    }
}
