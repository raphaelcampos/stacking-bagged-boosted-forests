/*********************************************************************
	
	Copyright (C) 2016 by Raphael Campos

	This program is free software; you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation; either version 2 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program; if not, write to the Free Software
	Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
	
********************************************************************/

#ifndef EXTERN_FUNC_CUH_
#define EXTERN_FUNC_CUH_

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
#include <map>

#include "structs.cuh"
#include "utils.cuh"
#include "inverted_index.cuh"
#include "knn.cuh"
#include "cuda_distances.cuh"
#include <cuda.h>

void initDeviceVariables(DeviceVariables *dev_vars, int K, int num_docs);

void freeDeviceVariables(DeviceVariables *dev_vars);

extern "C"
int* kneighbors(InvertedIndex* index, int K, float* data, int* indices, int* indptr, int nnz, int n, int n_gpu);

std::vector<Entry> csr2entries(float* data, int* indices, int* indptr, int nnz, int n);

InvertedIndex* make_inverted_indices(int num_docs, int num_terms, std::vector<Entry> entries, int n_gpu);

extern "C"
InvertedIndex* csr_make_inverted_indices(int num_docs, int num_terms, float* data, int* indices, int* indptr, int nnz, int n, int n_gpu);

extern "C"
InvertedIndex* make_inverted_indices(int num_docs, int num_terms, Entry * entries, int n_entries, int n_gpu);

#endif