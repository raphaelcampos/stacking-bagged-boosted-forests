#include <cstdio>
#include <cstdlib>

#include <algorithm>

#include "nvml.h"

extern "C"
__host__ int device_infos();

/**
 * Returns an array containing the Device Id
 * ordered by available resource in descreasning order
 * @param int*	Device ids by order of resources usage
 * @param int 	Number of devices to be prioritized
 * @param float	GPU usage weight (default : 1.0)
 */
extern "C"
__host__ void device_priority_order(int *order, int numgpu, float beta = 1.0);