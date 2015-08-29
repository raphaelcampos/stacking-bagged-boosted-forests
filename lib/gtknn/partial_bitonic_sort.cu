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

#include "partial_bitonic_sort.cuh"

__global__ void bitonicPartialSort(Similarity *dist, Similarity *nearestK, int N, int K) {
    int block_size = N / gridDim.x + (N % gridDim.x == 0 ? 0 : 1);		//Tamanho da particao
    int offset = block_size * (blockIdx.x); 	//Inicio do bloco
    int lim = min(offset + block_size, N); 				//Fim do bloco
    int size = lim - offset;					//Tamanho do bloco

    bitonicPartialSort(dist + offset, size, K);
    __syncthreads();
    bitonicPartialMerge(dist + offset, nearestK + K * blockIdx.x, size, K);
}

/**
 * Performs the bitonic merge until all partitions get combined in a single one with the K smallest elements
 */
__device__ void bitonicPartialMerge(Similarity *dist, Similarity *nearestK, int N, int K) {
    int num_partitions = (N + K - 1) / K;			   //Numero de particoes a serem combinadas
    int partition_step = 2;							   //Step size (in partitions)
    while(num_partitions > 1) {
        for(int i = threadIdx.x; ; i += blockDim.x) {
            int pos1 = (i / K) * K * partition_step + i % K;	//Calculates the positions of the threads in relation to the active partitions
            if(pos1 >= N) break;
            int pos2 = pos1 + (partition_step >> 1) * K;

            //keep only the smallest value
            if(pos2 < N) {
                Similarity x = dist[pos2];
                if(x < dist[pos1]) {
                    dist[pos1] = x;
                }
            }
        }

        __syncthreads();

        //sort the partitions alternately increasing/decreasing
        for(int k = 2; k <= K; k <<= 1) {
            for(int step = k >> 1; step > 0; step >>= 1) {
                int kk = K >> 1;
                int sk = partition_step * K;

                for(int i = threadIdx.x; ; i += blockDim.x) {
                    int pos1 = ((i / kk) * sk) + ((i % kk) / step) * 2 * step + (i % kk) % step;

                    if(pos1 >= N) break;

                    int pos2 = pos1 ^ step;

                    if(pos1 < pos2 && pos2 < N) {
                        Similarity x = dist[pos1];
                        Similarity y = dist[pos2];

                        //0 is up
                        //1 is down
                        int dir = ((pos1 & k) == 0) ^ (((i / kk) & 1) == 0);

                        if((dir == 0 && x > y ) || (dir != 0 && x < y )) {
                            dist[pos1] = y;
                            dist[pos2] = x;
                        }
                    }
                }
                __syncthreads();
            }
        }
        num_partitions = (num_partitions + 1) >> 1;
        partition_step <<= 1;
    }

    for(int i = threadIdx.x; i < K; i += blockDim.x) {
        nearestK[i] = dist[i];
    }
}

/**
 * Performs the bitonic sorting until the partitions get to the size K
 */
__device__ void bitonicPartialSort(Similarity *dist, int N, int K) {
    for(int k = 2; k <= K; k <<= 1) {
        for(int step = k >> 1; step > 0; step >>= 1) {
            for(int i = threadIdx.x; ; i += blockDim.x) {
                int pos1 = (i / step) * 2 * step + i % step;
                if(pos1 >= N) break;
                int pos2 = pos1 ^ step;

                if(pos2 < N) {
                    Similarity x = dist[pos1];
                    Similarity y = dist[pos2];

                    if(( (pos1 & k) == 0 && x > y ) || ( (pos1 & k) != 0 && x < y )) {
                        dist[pos1] = y;
                        dist[pos2] = x;
                    }
                }
            }
        }
    }
}
