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
 * partial_bitonic_sort.cuh
 *
 *  Created on: Sep 20, 2014
 *      Author: silvereagle
 */

#ifndef PARTIAL_BITONIC_SORT_CUH_
#define PARTIAL_BITONIC_SORT_CUH_

#include "structs.cuh"

__global__ void bitonicPartialSort(Similarity *dist, Similarity *nearestK, int N, int K);

__device__ void bitonicPartialMerge(Similarity *dist, Similarity *nearestK, int N, int K);

__device__ void bitonicPartialSort(Similarity *dist, int N, int K);

#endif /* PARTIAL_BITONIC_SORT_CUH_ */
