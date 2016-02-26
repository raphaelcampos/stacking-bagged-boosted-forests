#include "extern_func.cuh"

int biggestQuerySize = -1;

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

	int* idxs = new int[(n-1)*K];

	omp_set_num_threads(gpuNum);
	
	#pragma omp parallel shared(intervals) shared(data) shared(indices) shared(index) shared(idxs)
	{
		int num_test_local = 0, i;
		int cpuid = omp_get_thread_num();
    	
    	cudaSetDevice(cpuid);
		
		//printf("thread : %d\n", cpuid);

    	DeviceVariables dev_vars;
	
		initDeviceVariables(&dev_vars, K, index[cpuid].num_docs);

		//double start = gettime();

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
		
		//printf("num tests in thread %d: %d\n", omp_get_thread_num(), num_test_local);

		//#pragma omp barrier
		//double end = gettime();

		//#pragma omp master
		//printf("Time to process: %lf seconds\n", end - start);
	
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
	
    omp_set_num_threads(gpuNum);

	InvertedIndex* indices = new InvertedIndex[gpuNum];

	#pragma omp parallel shared(entries) shared(indices)
    {
    	int cpuid = omp_get_thread_num();
    	cudaSetDevice(cpuid);
		
		indices[cpuid] = make_inverted_index(num_docs, num_terms, entries);	
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
	
	std::cerr << "Using " << gpuNum << "GPUs" << std::endl;

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

__host__ void freeInvertedIndexes(InvertedIndex* indexes, unsigned int n_indexes){
    for (int i = 0; i < n_indexes; ++i)
    {
        freeInvertedIndex(indexes[i]);
    }
    delete [] indexes;
}



#define CUDA_CALL(function, ...)  { \
    cudaError_t status = function(__VA_ARGS__); \
    anyCheck(status == cudaSuccess, cudaGetErrorString(status), #function, __FILE__, __LINE__); \
}

#define NVML_CALL(function, ...)  { \
    nvmlReturn_t status = function(__VA_ARGS__); \
    anyCheck(status == NVML_SUCCESS, nvmlErrorString(status), #function, __FILE__, __LINE__); \
}

void anyCheck(bool is_ok, const char *description, const char *function, const char *file, int line) {
    if (!is_ok) {
        fprintf(stderr,"Error: %s in %s at %s:%d\n", description, function, file, line);
        exit(EXIT_FAILURE);
    }
}

int device_infos() {
    int cudaDeviceCount;
    unsigned int nvmlDeviceCount = 0;
    struct cudaDeviceProp deviceProp;
    nvmlPciInfo_t nvmlPciInfo;
    nvmlMemory_t nvmlMemory;
    nvmlProcessInfo_t *infos;
    nvmlDevice_t nvmlDevice;
    size_t memUsed, memTotal;
    unsigned int nProcess;

    CUDA_CALL(cudaGetDeviceCount, &cudaDeviceCount);
    NVML_CALL(nvmlInit);
    NVML_CALL(nvmlDeviceGetCount, &nvmlDeviceCount);

    for (int deviceId = 0; deviceId < cudaDeviceCount; ++deviceId) {
        CUDA_CALL(cudaGetDeviceProperties, &deviceProp, deviceId);
        int nvmlDeviceId = -1;
        for (int nvmlId = 0; nvmlId < nvmlDeviceCount; ++nvmlId) {
            NVML_CALL(nvmlDeviceGetHandleByIndex, nvmlId, &nvmlDevice);
            NVML_CALL(nvmlDeviceGetPciInfo, nvmlDevice, &nvmlPciInfo);
            if (deviceProp.pciDomainID == nvmlPciInfo.domain &&
                deviceProp.pciBusID    == nvmlPciInfo.bus    &&
                deviceProp.pciDeviceID == nvmlPciInfo.device) {

                nvmlDeviceId = nvmlId;
                break;
            }
        }
        printf("Device %2d [nvidia-smi %2d]", deviceId, nvmlDeviceId);
        printf(" [PCIe %04x:%02x:%02x.0]", deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);
        printf(": %20s (CC %d.%d)", deviceProp.name, deviceProp.major, deviceProp.minor);
        if (nvmlDeviceId != -1) {
            NVML_CALL(nvmlDeviceGetMemoryInfo, nvmlDevice, &nvmlMemory);
            //NVML_CALL(nvmlDeviceGetComputeRunningProcesses, nvmlDevice, &nProcess, infos);
            memUsed = nvmlMemory.used / 1024 / 1024;
            memTotal = nvmlMemory.total / 1024 / 1024;
        } else {
            memUsed = memTotal = 0;
            nProcess = 0;
        }
        printf(": %5zu of %5zu MiB Used", memUsed, memTotal);
        printf(": Processes %d ", nProcess);
        printf("\n");
    }
    NVML_CALL(nvmlShutdown);
    return 0;
}
