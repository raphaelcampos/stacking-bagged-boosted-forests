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
 * utils.cuh
 *
 *  Created on: Dec 4, 2013
 *      Author: silvereagle
 */

#ifndef UTILS_CUH_
#define UTILS_CUH_

#include <string>
#include <cstdio>
#include <vector>
#include <omp.h>

extern int WARP_SIZE;

std::vector<std::string> split(const std::string &s, char delim);

double gettime();

void get_grid_config(dim3 &grid, dim3 &threads);

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
void __gpuAssert(cudaError_t stat, int line, std::string file);

//__device__ float atomicAdd(float* address, float val);

#define gpuAssert(value)  __gpuAssert((value),(__LINE__),(__FILE__))


#endif /* UTILS_CUH_ */
