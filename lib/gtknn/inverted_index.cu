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

#include <cstdio>
#include <omp.h>

#include "inverted_index.cuh"
#include "utils.cuh"

/**
* Make inverted index on GPU using an array of Entry.
* \param num_docs - Number of documents.
* \param num_terms - Number of terms.
* \param entries - Entries pointer.
* \param n_entries - Number of entries.
*/
__host__ InvertedIndex make_inverted_index(int num_docs, int num_terms, Entry * entries, int n_entries) {
    #pragma omp single nowait
    printf("Creating inverted index... \n");
    Entry *d_entries, *d_inverted_index;
    int *d_count, *d_index;
    float *d_norms, *d_normsl1;

    #pragma omp single nowait
    printf("Allocating memory...\n");


    gpuAssert(cudaMalloc(&d_inverted_index, n_entries * sizeof(Entry)));
    gpuAssert(cudaMalloc(&d_entries, n_entries * sizeof(Entry)));
    gpuAssert(cudaMalloc(&d_index, num_terms * sizeof(int)));
    gpuAssert(cudaMalloc(&d_count, num_terms * sizeof(int)));
    gpuAssert(cudaMalloc(&d_norms, num_docs * sizeof(float)));
    gpuAssert(cudaMalloc(&d_normsl1, num_docs * sizeof(float)));

    gpuAssert(cudaMemset(d_count, 0, num_terms * sizeof(int)));
    gpuAssert(cudaMemset(d_norms, 0, num_docs * sizeof(float)));
    gpuAssert(cudaMemset(d_normsl1, 0, num_docs * sizeof(float)));
    gpuAssert(cudaMemcpy(d_entries, &entries[0], n_entries * sizeof(Entry), cudaMemcpyHostToDevice));

    #pragma omp single nowait
    printf("Finished allocating\n");   

    dim3 grid, threads;
    get_grid_config(grid, threads);

    double start = gettime();
    
    count_occurrences<<<grid, threads>>>(d_entries, d_count, n_entries);   
    
    thrust::device_ptr<int> thrust_d_count(d_count);
    thrust::device_ptr<int> thrust_d_index(d_index);
    thrust::exclusive_scan(thrust_d_count, thrust_d_count + num_terms, thrust_d_index); 

    mount_inverted_index_and_compute_tf_idf<<<grid, threads>>>(d_entries, d_inverted_index, d_count, d_index, d_norms, d_normsl1, n_entries, num_docs);

    gpuAssert(cudaDeviceSynchronize());
    
    double end = gettime();

    #pragma omp single nowait
    printf("time for insertion: %lf\n", end - start);
    cudaFree(d_entries);
    return InvertedIndex(d_inverted_index, d_index, d_count, d_norms, d_normsl1, num_docs, n_entries, num_terms);
}

__host__ InvertedIndex make_inverted_index(int num_docs, int num_terms, std::vector<Entry> &entries) {
	#pragma omp single nowait
    printf("Creating inverted index... \n");
    Entry *d_entries, *d_inverted_index;
    int *d_count, *d_index;
    float *d_norms, *d_normsl1;

	#pragma omp single nowait
    printf("Allocating memory...\n");


    gpuAssert(cudaMalloc(&d_inverted_index, entries.size() * sizeof(Entry)));
    gpuAssert(cudaMalloc(&d_entries, entries.size() * sizeof(Entry)));
    gpuAssert(cudaMalloc(&d_index, num_terms * sizeof(int)));
    gpuAssert(cudaMalloc(&d_count, num_terms * sizeof(int)));
    gpuAssert(cudaMalloc(&d_norms, num_docs * sizeof(float)));
    gpuAssert(cudaMalloc(&d_normsl1, num_docs * sizeof(float)));

    gpuAssert(cudaMemset(d_count, 0, num_terms * sizeof(int)));
    gpuAssert(cudaMemset(d_norms, 0, num_docs * sizeof(float)));
    gpuAssert(cudaMemset(d_normsl1, 0, num_docs * sizeof(float)));
    gpuAssert(cudaMemcpy(d_entries, &entries[0], entries.size() * sizeof(Entry), cudaMemcpyHostToDevice));

	#pragma omp single nowait
	printf("Finished allocating\n");   

    dim3 grid, threads;
    get_grid_config(grid, threads);

    double start = gettime();
	
    count_occurrences<<<grid, threads>>>(d_entries, d_count, entries.size());	
	
	thrust::device_ptr<int> thrust_d_count(d_count);
	thrust::device_ptr<int> thrust_d_index(d_index);
	thrust::exclusive_scan(thrust_d_count, thrust_d_count + num_terms, thrust_d_index);	

    mount_inverted_index_and_compute_tf_idf<<<grid, threads>>>(d_entries, d_inverted_index, d_count, d_index, d_norms, d_normsl1, entries.size(), num_docs);

	gpuAssert(cudaDeviceSynchronize());
	
    double end = gettime();

	#pragma omp single nowait
    printf("time for insertion: %lf\n", end - start);
    cudaFree(d_entries);
    return InvertedIndex(d_inverted_index, d_index, d_count, d_norms, d_normsl1, num_docs, entries.size(), num_terms);
}

__global__ void count_occurrences(Entry *entries, int *count, int n) {
    int block_size = n / gridDim.x + (n % gridDim.x == 0 ? 0 : 1);		//Number of items for each block
    int offset = block_size * (blockIdx.x); 							//Beginning of the block
    int lim = offset + block_size; 										//End of block the
    if(lim >= n) lim = n;
    int size = lim - offset;											//Block size

    entries += offset;

    for(int i = threadIdx.x; i < size; i+= blockDim.x) {
        int term_id = entries[i].term_id;
        atomicAdd(count + term_id, 1);
    }
}

__global__ void mount_inverted_index_and_compute_tf_idf(Entry *entries, Entry *inverted_index, int *count, int *index, float *d_norms, float *d_normsl1, int n, int num_docs) {
    int block_size = n / gridDim.x + (n % gridDim.x == 0 ? 0 : 1);		//Number of items used by each block
    int offset = block_size * (blockIdx.x); 							//Beginning of the block
    int lim = offset + block_size; 										//End of the block
    if(lim >= n) lim = n;
    int size = lim - offset;											//Block size


    entries += offset;

    for(int i = threadIdx.x; i < size; i+= blockDim.x) {
        Entry entry = entries[i];
        int pos = atomicAdd(index + entry.term_id, 1);

        entry.tf_idf = (1 + log(float(entry.tf))) * log(float(num_docs) / float(count[entry.term_id]));
        //entry.tf * log(float(num_docs) / float(count[entry.term_id]));
        inverted_index[pos] = entry;

        atomicAdd(&d_norms[entry.doc_id], entry.tf_idf * entry.tf_idf);
        atomicAdd(&d_normsl1[entry.doc_id], entry.tf_idf);

    }
}

__host__ void freeInvertedIndex(InvertedIndex &index){
    gpuAssert(cudaFree(index.d_count));
    gpuAssert(cudaFree(index.d_index));
    gpuAssert(cudaFree(index.d_inverted_index));
    gpuAssert(cudaFree(index.d_norms));
    gpuAssert(cudaFree(index.d_normsl1));
}
