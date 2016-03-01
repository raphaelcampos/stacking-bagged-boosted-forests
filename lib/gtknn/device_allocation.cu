#include "device_allocation.cuh"

#define CUDA_CALL(function, ...)  { \
    cudaError_t status = function(__VA_ARGS__); \
    anyCheck(status == cudaSuccess, cudaGetErrorString(status), #function, __FILE__, __LINE__); \
}

#define NVML_CALL(function, ...)  { \
    nvmlReturn_t status = function(__VA_ARGS__); \
    anyCheck(status == NVML_SUCCESS, nvmlErrorString(status), #function, __FILE__, __LINE__); \
}

void anyCheck(bool is_ok, const char *description, const char *function, const char *file, int line) {
    if (!is_ok) {
        fprintf(stderr,"Error: %s in %s at %s:%d\n", description, function, file, line);
    }
}

int device_infos() {
    int cudaDeviceCount;
    unsigned int nvmlDeviceCount = 0;
    struct cudaDeviceProp deviceProp;
    nvmlPciInfo_t nvmlPciInfo;
    nvmlMemory_t nvmlMemory;
    nvmlDevice_t nvmlDevice;
    size_t memUsed, memTotal;
    unsigned int nProcess = 0;

    CUDA_CALL(cudaGetDeviceCount, &cudaDeviceCount);
    NVML_CALL(nvmlInit);
    NVML_CALL(nvmlDeviceGetCount, &nvmlDeviceCount);

    for (int deviceId = 0; deviceId < cudaDeviceCount; ++deviceId) {
        CUDA_CALL(cudaGetDeviceProperties, &deviceProp, deviceId);
        int nvmlDeviceId = -1;
        for (int nvmlId = 0; nvmlId < nvmlDeviceCount; ++nvmlId) {
            NVML_CALL(nvmlDeviceGetHandleByIndex, nvmlId, &nvmlDevice);
            NVML_CALL(nvmlDeviceGetPciInfo, nvmlDevice, &nvmlPciInfo);
            if (deviceProp.pciDomainID == nvmlPciInfo.domain &&
                deviceProp.pciBusID    == nvmlPciInfo.bus    &&
                deviceProp.pciDeviceID == nvmlPciInfo.device) {

                nvmlDeviceId = nvmlId;
                break;
            }
        }
        printf("Device %2d [nvidia-smi %2d]", deviceId, nvmlDeviceId);
        printf(" [PCIe %04x:%02x:%02x.0]", deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);
        printf(": %20s (CC %d.%d)", deviceProp.name, deviceProp.major, deviceProp.minor);
        if (nvmlDeviceId != -1) {
            NVML_CALL(nvmlDeviceGetMemoryInfo, nvmlDevice, &nvmlMemory);
            memUsed = nvmlMemory.used / 1024 / 1024;
            memTotal = nvmlMemory.total / 1024 / 1024;
        } else {
            memUsed = memTotal = 0;
            nProcess = 0;
        }
        printf(": %5zu of %5zu MiB Used", memUsed, memTotal);
        printf(": Processes %d ", nProcess);
        printf("\n");
    }
    NVML_CALL(nvmlShutdown);
    return 0;
}

void device_priority_order(int *order, int numgpu, float beta) {
    
    int cudaDeviceCount;
    unsigned int nvmlDeviceCount = 0;
    struct cudaDeviceProp deviceProp;
    nvmlPciInfo_t nvmlPciInfo;
    nvmlMemory_t nvmlMemory;
    nvmlProcessInfo_t *infos;
    nvmlDevice_t nvmlDevice;
    size_t memUsed, memTotal;
    unsigned int nProcess;

    // initialize order
    for (int i = 0; i < numgpu; order[i] = i, ++i)

    cudaGetDeviceCount(&cudaDeviceCount);
    
    if( NVML_SUCCESS != nvmlInit() || cudaDeviceCount <= 1) return;
    nvmlDeviceGetCount(&nvmlDeviceCount);

    std::pair<float, int>* resourceUsage = new std::pair<float, int>[cudaDeviceCount];
    float* memUsage = new float[cudaDeviceCount];
    float* gpuUsage = new float[cudaDeviceCount];

    for (int deviceId = 0; deviceId < cudaDeviceCount; ++deviceId) {
        cudaGetDeviceProperties(&deviceProp, deviceId);
        int nvmlDeviceId = -1;
        for (int nvmlId = 0; nvmlId < nvmlDeviceCount; ++nvmlId) {
            nvmlDeviceGetHandleByIndex(nvmlId, &nvmlDevice);
            nvmlDeviceGetPciInfo(nvmlDevice, &nvmlPciInfo);
            if (deviceProp.pciDomainID == nvmlPciInfo.domain &&
                deviceProp.pciBusID    == nvmlPciInfo.bus    &&
                deviceProp.pciDeviceID == nvmlPciInfo.device) {

                nvmlDeviceId = nvmlId;
                break;
            }
        }
        
        if (nvmlDeviceId != -1) {
            nvmlDeviceGetMemoryInfo(nvmlDevice, &nvmlMemory);
            nvmlDeviceGetComputeRunningProcesses(nvmlDevice, &nProcess, infos);
            memUsed = nvmlMemory.used / 1024 / 1024;
            memTotal = nvmlMemory.total / 1024 / 1024;
        } else {
            memUsed = memTotal = 0;
            nProcess = 0;
        }

        memUsage[deviceId] = memUsed/(float)memTotal;
        gpuUsage[deviceId] = nProcess;

    }

    float minMem = *std::min_element(memUsage, memUsage + cudaDeviceCount), maxMem = *std::max_element(memUsage, memUsage + cudaDeviceCount);
    float minGpu = *std::min_element(gpuUsage, gpuUsage + cudaDeviceCount), maxGpu = *std::max_element(gpuUsage, gpuUsage + cudaDeviceCount);  

    for (int deviceId = 0; deviceId < cudaDeviceCount; ++deviceId)
    {
        // Normalization
        memUsage[deviceId] = (maxMem - minMem) > 0 ? (memUsage[deviceId] - minMem)/(maxMem - minMem) : 1;
        gpuUsage[deviceId] = (maxGpu - minGpu) > 0 ? (gpuUsage[deviceId] - minGpu)/(maxGpu - minGpu) : 1;

        float key = (!gpuUsage[deviceId] && !memUsage[deviceId]) ? 0 : ((1 + beta*beta) * memUsage[deviceId] * gpuUsage[deviceId])/(memUsage[deviceId] + (beta*beta)*gpuUsage[deviceId]);

        resourceUsage[deviceId] = std::make_pair(key, deviceId);
    }

    std::sort(resourceUsage, resourceUsage + cudaDeviceCount);

    for (int i = 0; i < cudaDeviceCount && i < numgpu; ++i)
    {
        printf("Device #%d : Mem. usage: %f\n", resourceUsage[i].second, resourceUsage[i].first*100);
        order[i] =  resourceUsage[i].second;       
    }

    delete [] resourceUsage, memUsage, gpuUsage;
    nvmlShutdown();
}
