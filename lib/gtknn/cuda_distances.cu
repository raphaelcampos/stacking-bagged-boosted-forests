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

#include "cuda_distances.cuh"

__host__ void CosineDistance(InvertedIndex inverted_index, Entry *d_query, int *index, cuSimilarity *dist, int D) {
    dim3 grid, threads;
    get_grid_config(grid, threads);

    initDistancesCosine<<<grid, threads>>>(inverted_index, dist);
    calculateDistancesCosine<<<grid, threads>>>(inverted_index, d_query, index, dist, D);
}

__global__ void initDistancesCosine(InvertedIndex inverted_index, cuSimilarity *dist) {
    int N = inverted_index.num_docs;
    int block_size = N / gridDim.x + (N % gridDim.x == 0 ? 0 : 1);		//Partition size
    int offset = block_size * (blockIdx.x); 	//Beginning of the block
    int lim = offset + block_size; 				//End of the block
    if(lim >= N) lim = N;
    int size = lim - offset;					//End of the block

    initDistancesCosineDevice(dist + offset, offset, size);
}

__device__ void initDistancesCosineDevice(cuSimilarity *dist, int offset, int N) {
    for(int i = threadIdx.x; i < N; i += blockDim.x) {
        dist[i] = cuSimilarity(i + offset, 0.0); //each position of dist has the memory position to a struct cuSimilarity, initialized by the cuSimilarity value (0.0) and the document id (i + offset)
    }
}

/*
 * Calculates distances from Entry x to all N Entry's in table
 * x has D elements
 * Computed distances are stored in dist
 */
__global__ void calculateDistancesCosine(InvertedIndex inverted_index, Entry *d_query, int *index, cuSimilarity *dist, int D) {
    __shared__ int N;

    if(threadIdx.x == 0) {
        N = index[D - 1];	//Total number of items to be queried
    }
    __syncthreads();

    int block_size = N / gridDim.x + (N % gridDim.x == 0 ? 0 : 1);		//Partition size
    int lo = block_size * (blockIdx.x); 								//Beginning of the block
    int hi = min(lo + block_size, N); 								//End of the block
    int size = hi - lo;

    int idx = 0;
    int end;

    for(int i = threadIdx.x; i < size; i += blockDim.x) {
        int pos = i + lo;

        while(true) {
            end = index[idx];

            if(end <= pos) {
                idx++;
            } else {
                break;
            }
        }

        Entry entry = d_query[idx]; 		//finds out the term
        int offset = end - pos;

        int idx2 = inverted_index.d_index[entry.term_id] - offset;
        Entry index_entry = inverted_index.d_inverted_index[idx2];

        float norm = sqrt(inverted_index.d_norms[index_entry.doc_id]); //it's not necessary if you only need the ranking order (and not the distances)!

        if(norm==0) {
            atomicAdd(&dist[index_entry.doc_id].distance, 0);
        } else {
            atomicAdd(&dist[index_entry.doc_id].distance, (entry.tf_idf * index_entry.tf_idf) / norm);
        }
    }
}


__global__ void initDistancesEuclidean(InvertedIndex inverted_index, cuSimilarity *dist) {
    int N = inverted_index.num_docs;
    int block_size = N / gridDim.x + (N % gridDim.x == 0 ? 0 : 1);		//Partition size
    int offset = block_size * (blockIdx.x); 	//Beginning of the block
    int lim = offset + block_size; 				//End of the block
    if(lim >= N) lim = N;
    int size = lim - offset;					//block size

    initDistancesEuclideanDevice(inverted_index.d_norms + offset, dist + offset, offset, size);
}

__host__ void EuclideanDistance(InvertedIndex inverted_index, Entry *d_query, int *index, cuSimilarity *dist, int D) {
    dim3 grid, threads;
    get_grid_config(grid, threads);

    initDistancesEuclidean<<<grid, threads>>>(inverted_index, dist);
    calculateDistancesEuclidean<<<grid, threads>>>(inverted_index, d_query, index, dist, D);
}

__device__ void initDistancesEuclideanDevice(float *d_norms, cuSimilarity *dist, int offset, int N) {
    for(int i = threadIdx.x; i < N; i += blockDim.x) {
        dist[i] = cuSimilarity(i + offset, -d_norms[i]); //adjust the norm to keep the decreasing ordering
    }
}

__global__ void initDistancesManhattan(InvertedIndex inverted_index, cuSimilarity *dist) {
    int N = inverted_index.num_docs;
    int block_size = N / gridDim.x + (N % gridDim.x == 0 ? 0 : 1);		//Partition size
    int offset = block_size * (blockIdx.x); 	//Beginning of the block
    int lim = offset + block_size; 				//End of the block
    if(lim >= N) lim = N;
    int size = lim - offset;					//block size

    initDistancesManhattanDevice(inverted_index.d_normsl1 + offset, dist + offset, offset, size);
}

__host__ void ManhattanDistance(InvertedIndex inverted_index, Entry *d_query, int *index, cuSimilarity *dist, int D) {
    dim3 grid, threads;
    get_grid_config(grid, threads);

    initDistancesManhattan<<<grid, threads>>>(inverted_index, dist);
    calculateDistancesManhattan<<<grid, threads>>>(inverted_index, d_query, index, dist, D);
}

__device__ void initDistancesManhattanDevice(float *d_normsl1, cuSimilarity *dist, int offset, int N) {
    for(int i = threadIdx.x; i < N; i += blockDim.x) {
        dist[i] = cuSimilarity(i + offset, d_normsl1[i]*-1.0 );
    }
}



/*
 * Calculates distances from Entry x to all N Entry's in table
 * x has D elements
 * Computed distances are stored in dist
 */
__global__ void calculateDistancesEuclidean(InvertedIndex inverted_index, Entry *d_query, int *index, cuSimilarity *dist, int D) {
    __shared__ int N;

    if(threadIdx.x == 0) {
        N = index[D - 1];	//Total number of items to be queried
    }
    __syncthreads();

    int block_size = N / gridDim.x + (N % gridDim.x == 0 ? 0 : 1);		//Partition size
    int lo = block_size * (blockIdx.x); 								//Beginning of the block
    int hi = min(lo + block_size, N); 								//End of the block
    int size = hi - lo;

    int idx = 0;
    int end;

    for(int i = threadIdx.x; i < size; i += blockDim.x) {
        int pos = i + lo;

        while(true) {
            end = index[idx];

            if(end <= pos) {
                idx++;
            } else {
                break;
            }
        }

        Entry entry = d_query[idx]; 		//finds out the term
        int offset = end - pos;

        int idx2 = inverted_index.d_index[entry.term_id] - offset;
        Entry index_entry = inverted_index.d_inverted_index[idx2];

        //(a-b)^2 = a^2-2ab+b^2.   Since we already have the norms, we just need to compute the 2ab
        atomicAdd(&dist[index_entry.doc_id].distance, 2.0 * (entry.tf_idf * index_entry.tf_idf));
    }
}

/*
 * Calculate distances from Entry x to all N Entry's in table
 * x has D elements
 * Computed distances are stored in dist
 */
__global__ void calculateDistancesManhattan(InvertedIndex inverted_index, Entry *d_query, int *index, cuSimilarity *dist, int D) {
    __shared__ int N;

    if(threadIdx.x == 0) {
        N = index[D - 1];	//Total number of items to be queried
    }
    __syncthreads();

    int block_size = N / gridDim.x + (N % gridDim.x == 0 ? 0 : 1);		//Partition size
    int lo = block_size * (blockIdx.x); 								//Beginning of the block
    int hi = min(lo + block_size, N); 								//End of the block
    int size = hi - lo;

    int idx = 0;
    int end;

    for(int i = threadIdx.x; i < size; i += blockDim.x) {
        int pos = i + lo;

        while(true) {
            end = index[idx];

            if(end <= pos) {
                idx++;
            } else {
                break;
            }
        }

        Entry entry = d_query[idx]; 		//finds out the term
        int offset = end - pos;

        int idx2 = inverted_index.d_index[entry.term_id] - offset;
        Entry index_entry = inverted_index.d_inverted_index[idx2];



        //we only need to compute the subtraction of the common dimensions. We can infer the remaining calculations from the norms.
        atomicAdd(&dist[index_entry.doc_id].distance, (entry.tf_idf+ index_entry.tf_idf)-abs(entry.tf_idf- index_entry.tf_idf));
    }
}
