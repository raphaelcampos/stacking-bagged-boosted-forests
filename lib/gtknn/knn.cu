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

/* *
 * knn.cu
 */

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <queue>
#include <vector>
#include <set>
#include <functional>



#include "knn.cuh"
#include "structs.cuh"
#include "utils.cuh"
#include "inverted_index.cuh"
#include "cuda_distances.cuh"
#include "partial_bitonic_sort.cuh"

/*
* We pass the distance function  as a pointer  (*distance)
*/

__host__ cuSimilarity* KNN(InvertedIndex inverted_index, Entry *query, int K, int query_size, struct DeviceVariables *dev_vars) {
    //Obtain the smallest power of 2 greater than K (facilitates the sorting algorithm)
    int KK = 1;
    while(KK < K) KK <<= 1;

    dim3 grid, threads;
    get_grid_config(grid, threads);

    int *d_count = dev_vars->d_count, *d_index = dev_vars->d_index;
    Entry *d_query = dev_vars->d_query;
    cuSimilarity *d_dist = dev_vars->d_dist, *d_nearestK = dev_vars->d_nearestK, *h_nearestK;
    float *d_qnorm = &dev_vars->d_qnorms[0],*d_qnorml1 = &dev_vars->d_qnorms[1];
    
    gpuAssert(cudaMemcpy(d_query, &query[0], query_size * sizeof(Entry), cudaMemcpyHostToDevice));
    gpuAssert(cudaMemset(dev_vars->d_qnorms, 0,  2 * sizeof(float))); 
       
    get_term_count_and_tf_idf<<<grid, threads>>>(inverted_index, d_query, d_count, d_qnorm, d_qnorml1, query_size);

    thrust::device_ptr<int> thrust_d_count(d_count);
    thrust::device_ptr<int> thrust_d_index(d_index);
    thrust::inclusive_scan(thrust_d_count, thrust_d_count + query_size, thrust_d_index);
    
   /* int *index = (int*) malloc(query.size() * sizeof(int)); // Useless ?
    cudaMemcpyAsync(index, d_index, query.size() * sizeof(int), cudaMemcpyDeviceToHost);*/
    float qnorm, qnorml1, qnorms[2];

    gpuAssert(cudaMemcpy(qnorms, dev_vars->d_qnorms, 2* sizeof(float), cudaMemcpyDeviceToHost));
    
    qnorm = qnorms[0];
    qnorml1 = qnorms[1];
//  printf("%f\n", qnorm);
    float qnormeucl=qnorm;
    float qnormcos=sqrt(qnorm);
    if (qnormcos==0)qnormcos=1; //avoid NaN
    
    CosineDistance(inverted_index, d_query, d_index, d_dist, query_size);

    bitonicPartialSort<<<grid, threads>>>(d_dist, d_nearestK, inverted_index.num_docs, KK); 
 
    h_nearestK = (cuSimilarity*) malloc(KK * grid.x * sizeof(cuSimilarity));
    gpuAssert(cudaMemcpy(h_nearestK, d_nearestK, KK * grid.x * sizeof(cuSimilarity), cudaMemcpyDeviceToHost));
    
    gpuAssert(cudaDeviceSynchronize());
    //Priority queue to obtain the K nearest neighbors
    std::priority_queue<cuSimilarity> pq;
//    std::set<int> alreadyIn;

    for(int i = 0, lim = KK * grid.x; i < lim; i++) {
//      printf("sim: %f\n",sim.distance );

        //adjust the correct distances:
        //if(distance==CosineDistance) {
            h_nearestK[i].distance/=qnormcos;
            //printf("sim: %f, id: %d qnorm: %f\n",h_nearestK[i].distance, h_nearestK[i].doc_id, qnorm);
        /*}
        else if(distance==EuclideanDistance) {
            h_nearestK[i].distance-=qnormeucl;
            h_nearestK[i].distance=sqrt(h_nearestK[i].distance*-1.0)*-1.0;
            //printf("sim: %f, id: %d qnorm: %f\n",h_nearestK[i].distance, h_nearestK[i].doc_id, qnorm);
        }
        else if(distance==ManhattanDistance) {
            h_nearestK[i].distance-=qnorml1;
            //printf("sim: %f, id: %d qnorm: %f\n",h_nearestK[i].distance, h_nearestK[i].doc_id, qnorml1);
        }*/

//        if(alreadyIn.find(h_nearestK[i].doc_id)==alreadyIn.end()) continue;
//        alreadyIn.insert(h_nearestK[i].doc_id);
        if(pq.size() != K) {
            pq.push(h_nearestK[i]);
        } else {
            const cuSimilarity &sim = pq.top();
            if(sim > h_nearestK[i]) {
                pq.pop();
                pq.push(h_nearestK[i]);
            }
        }
    }

    int i = K - 1;
    while(!pq.empty()) {
        const cuSimilarity &sim = pq.top();
//      cuSimilarity sim = pq.top();
//      sim.distance=sim.distance/qnorm;
        h_nearestK[i--] = sim;
        pq.pop();
    } 

    return h_nearestK;
}

__host__ cuSimilarity* KNN(InvertedIndex inverted_index, std::vector<Entry> &query, int K,
        void (*distance)(InvertedIndex, Entry*, int*, cuSimilarity*, int D), struct DeviceVariables *dev_vars) {
    //Obtain the smallest power of 2 greater than K (facilitates the sorting algorithm)
    int KK = 1;
    while(KK < K) KK <<= 1;

    dim3 grid, threads;
    get_grid_config(grid, threads);

    int *d_count = dev_vars->d_count, *d_index = dev_vars->d_index;
    Entry *d_query = dev_vars->d_query;
	cuSimilarity *d_dist = dev_vars->d_dist, *d_nearestK = dev_vars->d_nearestK, *h_nearestK;
	float *d_qnorm = &dev_vars->d_qnorms[0],*d_qnorml1 = &dev_vars->d_qnorms[1];
    
    gpuAssert(cudaMemcpy(d_query, &query[0], query.size() * sizeof(Entry), cudaMemcpyHostToDevice));
    gpuAssert(cudaMemset(dev_vars->d_qnorms, 0,  2 * sizeof(float))); 
	   
    get_term_count_and_tf_idf<<<grid, threads>>>(inverted_index, d_query, d_count, d_qnorm, d_qnorml1, query.size());

	thrust::device_ptr<int> thrust_d_count(d_count);
	thrust::device_ptr<int> thrust_d_index(d_index);
	thrust::inclusive_scan(thrust_d_count, thrust_d_count + query.size(), thrust_d_index);
	
   /* int *index = (int*) malloc(query.size() * sizeof(int)); // Useless ?
    cudaMemcpyAsync(index, d_index, query.size() * sizeof(int), cudaMemcpyDeviceToHost);*/
    float qnorm, qnorml1, qnorms[2];

    gpuAssert(cudaMemcpy(qnorms, dev_vars->d_qnorms, 2* sizeof(float), cudaMemcpyDeviceToHost));
	
	qnorm = qnorms[0];
	qnorml1 = qnorms[1];
//	printf("%f\n", qnorm);
    float qnormeucl=qnorm;
    float qnormcos=sqrt(qnorm);
    if (qnormcos==0)qnormcos=1; //avoid NaN
	
	distance(inverted_index, d_query, d_index, d_dist, query.size());

	bitonicPartialSort<<<grid, threads>>>(d_dist, d_nearestK, inverted_index.num_docs, KK);	
 
    h_nearestK = (cuSimilarity*) malloc(KK * grid.x * sizeof(cuSimilarity));
    gpuAssert(cudaMemcpy(h_nearestK, d_nearestK, KK * grid.x * sizeof(cuSimilarity), cudaMemcpyDeviceToHost));
    
    gpuAssert(cudaDeviceSynchronize());
    //Priority queue to obtain the K nearest neighbors
    std::priority_queue<cuSimilarity> pq;
//    std::set<int> alreadyIn;

    for(int i = 0, lim = KK * grid.x; i < lim; i++) {
//		printf("sim: %f\n",sim.distance );

        //adjust the correct distances:
        if(distance==CosineDistance) {
            h_nearestK[i].distance/=qnormcos;
            //printf("sim: %f, id: %d qnorm: %f\n",h_nearestK[i].distance, h_nearestK[i].doc_id, qnorm);
        }
        else if(distance==EuclideanDistance) {
            h_nearestK[i].distance-=qnormeucl;
            h_nearestK[i].distance=sqrt(h_nearestK[i].distance*-1.0)*-1.0;
            //printf("sim: %f, id: %d qnorm: %f\n",h_nearestK[i].distance, h_nearestK[i].doc_id, qnorm);
        }
        else if(distance==ManhattanDistance) {
            h_nearestK[i].distance-=qnorml1;
            //printf("sim: %f, id: %d qnorm: %f\n",h_nearestK[i].distance, h_nearestK[i].doc_id, qnorml1);
        }

//        if(alreadyIn.find(h_nearestK[i].doc_id)==alreadyIn.end()) continue;
//        alreadyIn.insert(h_nearestK[i].doc_id);
        if(pq.size() != K) {
            pq.push(h_nearestK[i]);
        } else {
            const cuSimilarity &sim = pq.top();
            if(sim > h_nearestK[i]) {
                pq.pop();
                pq.push(h_nearestK[i]);
            }
        }
    }

    int i = K - 1;
    while(!pq.empty()) {
        const cuSimilarity &sim = pq.top();
//		cuSimilarity sim = pq.top();
//		sim.distance=sim.distance/qnorm;
        h_nearestK[i--] = sim;
        pq.pop();
    } 

    return h_nearestK;
}

__global__ void get_term_count_and_tf_idf(InvertedIndex inverted_index, Entry *query, int *count, float *d_qnorm, float *d_qnorml1, int N) {
    int block_size = N / gridDim.x + (N % gridDim.x == 0 ? 0 : 1);		//Partition size
    int offset = block_size * (blockIdx.x); 	//Beginning of the block
    int lim = min(offset + block_size, N); 		//End of the block
    int size = lim - offset; 					//Block size

    query += offset;
    count += offset;

    for(int i = threadIdx.x; i < size; i += blockDim.x) {
        Entry entry = query[i];

        int idf = inverted_index.d_count[entry.term_id];
        query[i].tf_idf = (1 + log(float(entry.tf))) * log(inverted_index.num_docs / float(max(1, idf)));//entry.tf * log(inverted_index.num_docs / float(max(1, idf)));
        count[i] = idf;
        atomicAdd(d_qnorm, query[i].tf_idf * query[i].tf_idf);
        atomicAdd(d_qnorml1, query[i].tf_idf );

    }
}
