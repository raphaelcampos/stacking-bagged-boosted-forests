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

/*
 * cuda_distances.cuh
 *
 *  Created on: Sep 20, 2014
 *      Author: silvereagle
 */

#ifndef CUDA_DISTANCES_CUH_
#define CUDA_DISTANCES_CUH_

#include "structs.cuh"
#include "inverted_index.cuh"
#include "utils.cuh"

/**
 * Cosine similarity distance
 */
__host__ void CosineDistance(InvertedIndex inverted_index, Entry *d_query, int *index, Similarity *dist, int D);

__global__ void initDistancesCosine(InvertedIndex inverted_index, Similarity *dist);

__device__ void initDistancesCosineDevice(Similarity *dist, int offset, int N);

__global__ void calculateDistancesCosine(InvertedIndex inverted_index, Entry *d_query, int *index, Similarity *dist, int D);

/**
 * Euclidean distance functions
 */
__host__ void EuclideanDistance(InvertedIndex inverted_index, Entry *d_query, int *index, Similarity *dist, int D);

__global__ void initDistancesEuclidean(InvertedIndex inverted_index, Similarity *dist);

__device__ void initDistancesEuclideanDevice(float *d_norms, Similarity *dist, int offset, int N);

__global__ void calculateDistancesEuclidean(InvertedIndex inverted_index, Entry *d_query, int *index, Similarity *dist, int D);

__global__ void calculateDistancesManhattan(InvertedIndex inverted_index, Entry *d_query, int *index, Similarity *dist, int D);

__host__ void ManhattanDistance(InvertedIndex inverted_index, Entry *d_query, int *index, Similarity *dist, int D);

__global__ void initDistancesManhattan(InvertedIndex inverted_index, Similarity *dist);

__device__ void initDistancesManhattanDevice(float *d_norms, Similarity *dist, int offset, int N);

#endif /* CUDA_DISTANCES_CUH_ */
