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
#include <cudpp.h>

#include "inverted_index.cuh"
#include "utils.cuh"

__host__ InvertedIndex make_inverted_index(int num_docs, int num_terms, std::vector<Entry> &entries) {
    printf("Creating inverted index... \n", num_terms);
    Entry *d_entries, *d_inverted_index;
    int *d_count, *d_index;
    float *d_norms, *d_normsl1;

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

    gpuAssert(cudaGetLastError());

    cudaDeviceSynchronize();


    dim3 grid, threads;
    get_grid_config(grid, threads);

    double start = gettime();
    count_occurrences<<<grid, threads>>>(d_entries, d_count, entries.size());

    gpuAssert(cudaGetLastError());

    prefix_scan(d_index, d_count, num_terms, CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE);

    mount_inverted_index_and_compute_tf_idf<<<grid, threads>>>(d_entries, d_inverted_index, d_count, d_index, d_norms, d_normsl1, entries.size(), num_docs);

    cudaDeviceSynchronize();

    gpuAssert(cudaGetLastError());

    double end = gettime();

    printf("time for insertion: %lf\n", end - start);
    cudaFree(d_entries);
    return InvertedIndex(d_inverted_index, d_index, d_count, d_norms, d_normsl1, num_docs, entries.size(), num_terms);
}

__host__ void prefix_scan(int *d_out, int *d_in, int num_terms, unsigned int options) {
    CUDPPHandle scan_plan = create_exclusive_scan_plan(theCudpp, num_terms, options);

    if(cudppScan(scan_plan, d_out, d_in, num_terms) != CUDPP_SUCCESS) {
        printf("prefix-sum error\n");
    }

    cudppDestroy(scan_plan);
}

__host__ CUDPPHandle create_exclusive_scan_plan(CUDPPHandle theCudpp, int num_elements, unsigned int options) {
    CUDPPConfiguration config;
    config.algorithm = CUDPP_SCAN;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_INT;
    config.options = options;

    CUDPPHandle scan_plan = 0;
    if(cudppPlan(theCudpp, &scan_plan, config, num_elements, 1, 0) != CUDPP_SUCCESS) {
        printf("Erro na criacao do plano\n");
        exit(1);
    }

    return scan_plan;
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
        inverted_index[pos] = entry;

        atomicAdd(&d_norms[entry.doc_id], pow(entry.tf_idf, 2.0f));
        atomicAdd(&d_normsl1[entry.doc_id], entry.tf_idf);

    }
}
