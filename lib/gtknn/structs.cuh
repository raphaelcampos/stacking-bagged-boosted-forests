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
 * structs.cuh
 *
 *  Created on: Dec 4, 2013
 *      Author: silvereagle
 */

#ifndef STRUCTS_CUH_
#define STRUCTS_CUH_

struct Entry {
    int doc_id;
    int term_id;
    int tf;
    float tf_idf;
    #ifdef __CUDACC__
    __host__ __device__ 
    #endif
    Entry(int doc_id, int term_id, int tf = 0, float tf_idf = 0.0) : doc_id(doc_id), term_id(term_id), tf(tf), tf_idf(tf_idf) {}

    bool operator < (const Entry& e) const {
        if(doc_id != e.doc_id) return doc_id < e.doc_id;
        return term_id < e.term_id;
    }
};

struct cuSimilarity {
    int doc_id;
    float distance;

    cuSimilarity() {}
    #ifdef __CUDACC__
    __host__ __device__ 
    #endif
    cuSimilarity(int doc_id, float distance) : doc_id(doc_id), distance(distance) {}

    #ifdef __CUDACC__
    __host__ __device__ 
    #endif
    bool operator < (const cuSimilarity &sim) const {
        return distance > sim.distance;
    }

    #ifdef __CUDACC__
    __host__ __device__ 
    #endif
    bool operator > (const cuSimilarity &sim) const {
        return distance < sim.distance;
    }
};

struct DeviceVariables{
	int *d_count, *d_index;
    Entry *d_query;
    cuSimilarity *d_dist, *d_nearestK, *h_nearestK;
    float *d_qnorms;// [2] =  *d_qnorm, *d_qnorml1;
};

#endif /* STRUCTS_CUH_ */
