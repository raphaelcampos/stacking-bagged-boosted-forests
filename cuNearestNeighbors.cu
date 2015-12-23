#include "cuNearestNeighbors.cuh"

#include <iostream>

void initDeviceVariables(DeviceVariables *dev_vars, int K, int num_docs, int biggestQuerySize = 10000){
	
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

cuSimilarity* makeQuery(InvertedIndex &inverted_index, std::map<unsigned int, float> &test_features, int K,
	void(*distance)(InvertedIndex, Entry*, int*, cuSimilarity*, int D), DeviceVariables *dev_vars) {

    std::vector<Entry> query;
    std::map<unsigned int, float>::const_iterator end = test_features.end();
	for(std::map<unsigned int, float>::const_iterator it = test_features.begin(); it != end; ++it){
		unsigned int term_id = it->first;
		double term_count = it->second;

		// it means that query has higher dimensonality
		// than traning set. Thus, we remove that term
		if(term_id < inverted_index.num_terms)		
			query.push_back(Entry(0, term_id, term_count)); // doc_id, term_id, term_count
	}

    //Creates an empty document if there are no terms
    if(query.empty()) {
        query.push_back(Entry(0, 0, 0));
    }

    return KNN(inverted_index, query, K, distance, dev_vars);
}

cuNearestNeighbors::~cuNearestNeighbors(){
	doc_to_class.clear();
	entries.clear();

	// ver como liberar memoria da placa
	for (int i = 0; i < n_gpus; ++i)
	{
		freeInvertedIndex(inverted_indices[i]);
	}
	delete [] inverted_indices;
}

cuNearestNeighbors::cuNearestNeighbors(Dataset &data, int n_gpus): n_gpus(n_gpus){
	convertDataset(data);

	buildInvertedIndex();
}

void cuNearestNeighbors::train(Dataset &data){
	convertDataset(data);

	buildInvertedIndex();
}

int cuNearestNeighbors::classify(std::map<unsigned int, float> &test_features, int K){
	
	cuSimilarity *k_nearest = getKNearestNeighbors(test_features, K);
	int vote = getMajorityVote(k_nearest, K);

	delete[] k_nearest;
	return vote;
}

cuSimilarity * cuNearestNeighbors::getKNearestNeighbors(const std::map<unsigned int, float> &test_features, int K){

	std::vector<Entry> query;
	std::map<unsigned int, float>::const_iterator end = test_features.end();
	for(std::map<unsigned int, float>::const_iterator it = test_features.begin(); it != end; ++it){
		unsigned int term_id = it->first;
		double term_count = it->second;

		// it means that query has higher dimensonality
		// than traning set. Thus, we remove that term
		if(term_id < num_terms)		
			query.push_back(Entry(0, term_id, term_count)); // doc_id, term_id, term_count
	}

    //Creates an empty document if there are no terms
    if(query.empty()) {
        query.push_back(Entry(0, 0, 0));
    }

    DeviceVariables dev_vars;
	
	initDeviceVariables(&dev_vars, K, inverted_indices[0].num_docs);

	cuSimilarity* k_nearest = KNN(inverted_indices[0], query, K, CosineDistance, &dev_vars);

	freeDeviceVariables(&dev_vars);

	return k_nearest;
}

std::vector<cuSimilarity*> cuNearestNeighbors::getKNearestNeighbors(Dataset &test, int K){

	std::string distance = "cosine"; 

	int gpuNum;
	cudaGetDeviceCount(&gpuNum);

	if (gpuNum > n_gpus){
		gpuNum = n_gpus;
		if (gpuNum < 1)
			gpuNum = 1;
	}
	n_gpus = gpuNum;

	std::vector<sample> &samples = test.getSamples();
	std::vector<std::pair<int, int> > intervals;
	std::vector<cuSimilarity*> idxs(samples.size());
	InvertedIndex* inverted_indices = this->inverted_indices;
	
	int biggestQuerySize = test.biggestQuerySize;

	omp_set_num_threads(gpuNum);
	
	#pragma omp parallel shared(samples) shared(inverted_indices) shared(idxs)
	{
		int num_test_local = 0, i;
		int cpuid = omp_get_thread_num();

    	cudaSetDevice(cpuid);

    	DeviceVariables dev_vars;
	
		initDeviceVariables(&dev_vars, K, inverted_indices[cpuid].num_docs, biggestQuerySize);


    	#pragma omp for
		for (i = 0; i < samples.size(); ++i)
		{
			num_test_local++;
			
			if(distance == "cosine" || distance == "both") {
				idxs[i] = makeQuery(inverted_indices[cpuid], samples[i].features, K, CosineDistance, &dev_vars);
        	}

	        if(distance == "l2" || distance == "both") {
        		idxs[i] = makeQuery(inverted_indices[cpuid], samples[i].features, K, EuclideanDistance, &dev_vars);
			}

			if(distance == "l1" || distance == "both") {
				idxs[i] = makeQuery(inverted_indices[cpuid], samples[i].features, K, ManhattanDistance, &dev_vars);
			}
		}
	
		freeDeviceVariables(&dev_vars);
	}

	return idxs;
}

void cuNearestNeighbors::convertDataset(Dataset &data){
	
	num_terms = 0;
	// delete all old entries
	entries.clear();

	num_docs = data.getSamples().size();
	for (unsigned int i = 0; i < num_docs; ++i)
	{
		unsigned int doc_id = i;

		std::map<unsigned int, float>::iterator it;
		for(it = data.getSamples()[i].features.begin(); it != data.getSamples()[i].features.end(); ++it){
			unsigned int term_id = it->first;
			double term_cout = it->second;

			num_terms = std::max(num_terms, term_id + 1);
	
			entries.push_back(Entry(doc_id, term_id, term_cout)); // doc_id, term_id, term_count
		}

		doc_to_class[doc_id] = data.getSamples()[i].y;
	}
}

int cuNearestNeighbors::getMajorityVote(cuSimilarity *k_nearest, int K){
	std::map<int, double> vote_count;

	//cuSimilarity &closest = k_nearest[0];
	//cuSimilarity &further = k_nearest[K-1];

    for(int i = 0; i < K; ++i) {
        cuSimilarity &sim = k_nearest[i];
        vote_count[doc_to_class[sim.doc_id]] += sim.distance;
        //vote_count[doc_to_class[sim.doc_id]]+=((further.distance-sim.distance)/(further.distance-closest.distance))*((sim.distance+further.distance)/(closest.distance+further.distance))*(i);
        
        //vote_count[doc_to_class[sim.doc_id]]+=((further.distance-sim.distance)/(further.distance-closest.distance))*((double)i);
    }

    int max_votes = 0;
    int guessed_class = -1;
    std::map<int, double>::iterator end = vote_count.end();
    for(std::map<int, double>::iterator it = vote_count.begin(); it != end; it++) {
        if(it->second > max_votes) {
            max_votes = it->second;
            guessed_class = it->first;
        }
    }

    return guessed_class;
}

void cuNearestNeighbors::buildInvertedIndex(){
	int gpuNum;
	cudaGetDeviceCount(&gpuNum);

	if (gpuNum > n_gpus){
		gpuNum = n_gpus;
		if (gpuNum < 1)
			gpuNum = 1;
	}

	n_gpus = gpuNum;

    omp_set_num_threads(gpuNum);

	this->inverted_indices = new InvertedIndex[gpuNum];

	std::vector<Entry> &entries = this->entries;
	InvertedIndex* inverted_indices = this->inverted_indices;
	#pragma omp parallel shared(entries) shared(inverted_indices)
	{
		int cpuid = omp_get_thread_num();
		cudaSetDevice(cpuid);
		
		inverted_indices[cpuid] = make_inverted_index(num_docs, num_terms, entries);	
	}

	entries.clear();
}
