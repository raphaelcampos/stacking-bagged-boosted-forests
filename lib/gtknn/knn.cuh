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
 * knn.cuh
 *
 *  Created on: Dec 4, 2013
 *      Author: silvereagle
 */

#ifndef KNN_CUH_
#define KNN_CUH_

#include "inverted_index.cuh"

__global__ void calculateDistances(InvertedIndex inverted_index, Entry *d_query, int *index, cuSimilarity *dist, int D);

__global__ void bitonicPartialSort(cuSimilarity *dist, cuSimilarity *nearestK, int N, int K);

__global__ void get_term_count_and_tf_idf(InvertedIndex inverted_index, Entry *query, int *count, float *qnorm, float *qnorml1, int N);

__host__ cuSimilarity* KNN(InvertedIndex inverted_index, std::vector<Entry> &query, int K,
                         void (*distance)(InvertedIndex, Entry*, int*, cuSimilarity*, int D));

__device__ void bitonicPartialSort(cuSimilarity *dist, int N, int K);

__device__ void bitonicPartialMerge(cuSimilarity *dist, cuSimilarity *nearestK, int N, int K);

__device__ void initDistances(cuSimilarity *dist, int offset, int N);

__device__ void calculateDistancesDevice(InvertedIndex inverted_index, Entry *d_query, int *index, cuSimilarity *dist, int D);

#endif /* KNN_CUH_ */
