OMP= -Xcompiler -fopenmp

# Gencode arguments
ifeq ($(OS_ARCH),armv7l)
SMS ?= 20 30 32 35 37 50
else
SMS ?= 20 30 35 37 50
endif

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

ifeq ($(SHARED),T)
	SHARED=-Xcompiler -fPIC -shared
endif

Release: main.o knn.o partial_bitonic_sort.o inverted_index.o  cuda_distances.o utils.o
	nvcc $(GENCODE_FLAGS) -O3 -lgomp   main.o knn.o partial_bitonic_sort.o inverted_index.o  cuda_distances.o  utils.o -o gtknn

Shared: extern_func.o knn.o partial_bitonic_sort.o inverted_index.o  cuda_distances.o utils.o device_allocation.o
	nvcc -link -Xcompiler -fPIC -shared $(GENCODE_FLAGS) $(OMP) -O3 -lnvidia-ml extern_func.o knn.o partial_bitonic_sort.o inverted_index.o  cuda_distances.o  utils.o device_allocation.o -o "gtknn.so"

main.o: main.cu cuda_distances.cuh  knn.cuh  inverted_index.cuh utils.cuh structs.cuh
	nvcc $(GENCODE_FLAGS) $(SHARED) -O3 $(OMP)  -c main.cu

extern_func.o: extern_func.cu extern_func.cuh cuda_distances.cuh  knn.cuh inverted_index.cuh utils.cuh device_allocation.cuh structs.cuh
	nvcc $(GENCODE_FLAGS) $(SHARED) -O3 $(OMP) -c extern_func.cu

knn.o: knn.cu cuda_distances.cuh  knn.cuh  inverted_index.cuh utils.cuh structs.cuh partial_bitonic_sort.cuh 
	nvcc $(GENCODE_FLAGS) $(SHARED) -O3  $(OMP) -c knn.cu

partial_bitonic_sort.o: partial_bitonic_sort.cu partial_bitonic_sort.cuh 
	nvcc $(GENCODE_FLAGS) $(SHARED) -O3  $(OMP) -c partial_bitonic_sort.cu

inverted_index.o: inverted_index.cu inverted_index.cuh utils.cuh 
	nvcc $(GENCODE_FLAGS) $(SHARED) -O3  $(OMP) -c inverted_index.cu

cuda_distances.o: cuda_distances.cu cuda_distances.cuh 
	nvcc $(GENCODE_FLAGS) $(SHARED) -O3  $(OMP) -c cuda_distances.cu

device_allocation.o: device_allocation.cu device_allocation.cuh 
	nvcc $(GENCODE_FLAGS) $(SHARED) -lnvidia-ml -O3 $(OMP)  -c device_allocation.cu

utils.o: utils.cu utils.cuh 
	nvcc $(GENCODE_FLAGS) $(SHARED) -O3 $(OMP)  -c utils.cu

clean:
	rm *.o gtknn
