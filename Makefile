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
	SHARED := -fPIC -shared
endif

Release: main.o cuLazyNN_RF.o cuNearestNeighbors.o Dataset.o knn.o partial_bitonic_sort.o inverted_index.o  cuda_distances.o utils.o
	nvcc $(GENCODE_FLAGS) -O3 -lgomp -I"include/" main.o cuLazyNN_RF.o cuNearestNeighbors.o Dataset.o knn.o partial_bitonic_sort.o inverted_index.o cuda_distances.o  utils.o -o lazynn_rf

Shared: main.o knn.o partial_bitonic_sort.o inverted_index.o  cuda_distances.o utils.o
	nvcc -link -Xcompiler -fPIC -shared $(GENCODE_FLAGS) $(OMP) -O3 -I"include/" main.o knn.o partial_bitonic_sort.o inverted_index.o  cuda_distances.o  utils.o -o "gtknn.so"

main.o: main.cu cuLazyNN_RF.cuh cuNearestNeighbors.cuh Dataset.h lib/gtknn/cuda_distances.cuh  lib/gtknn/knn.cuh  lib/gtknn/inverted_index.cuh lib/gtknn/utils.cuh lib/gtknn/structs.cuh
	nvcc $(GENCODE_FLAGS) $(SHARED) -O3 $(OMP) -I"lib/gtknn/" -I"include/" -c main.cu

cuLazyNN_RF.o: cuLazyNN_RF.cu cuLazyNN_RF.cuh cuNearestNeighbors.cuh Dataset.h lib/gtknn/cuda_distances.cuh  lib/gtknn/knn.cuh  lib/gtknn/inverted_index.cuh lib/gtknn/utils.cuh lib/gtknn/structs.cuh
	nvcc $(GENCODE_FLAGS) $(SHARED) -O3 $(OMP) -I"lib/gtknn/" -c cuLazyNN_RF.cu

cuNearestNeighbors.o: cuNearestNeighbors.cu cuNearestNeighbors.cuh Dataset.h lib/gtknn/cuda_distances.cuh  lib/gtknn/knn.cuh  lib/gtknn/inverted_index.cuh lib/gtknn/utils.cuh lib/gtknn/structs.cuh
	nvcc $(GENCODE_FLAGS) $(SHARED) -O3 $(OMP) -I"lib/gtknn/" -c cuNearestNeighbors.cu

Dataset.o: Dataset.cpp Dataset.h
	nvcc $(GENCODE_FLAGS) $(SHARED) -O3 $(OMP) -c Dataset.cpp

knn.o: lib/gtknn/knn.cu lib/gtknn/cuda_distances.cuh  lib/gtknn/knn.cuh  lib/gtknn/inverted_index.cuh lib/gtknn/utils.cuh lib/gtknn/structs.cuh lib/gtknn/partial_bitonic_sort.cuh 
	nvcc $(GENCODE_FLAGS) $(SHARED) -O3  $(OMP) -c lib/gtknn/knn.cu

partial_bitonic_sort.o: lib/gtknn/partial_bitonic_sort.cu lib/gtknn/partial_bitonic_sort.cuh 
	nvcc $(GENCODE_FLAGS) $(SHARED) -O3  $(OMP) -c lib/gtknn/partial_bitonic_sort.cu

inverted_index.o: lib/gtknn/inverted_index.cu lib/gtknn/inverted_index.cuh lib/gtknn/utils.cuh 
	nvcc $(GENCODE_FLAGS) $(SHARED) -O3  $(OMP) -c lib/gtknn/inverted_index.cu

cuda_distances.o: lib/gtknn/cuda_distances.cu lib/gtknn/cuda_distances.cuh 
	nvcc $(GENCODE_FLAGS) $(SHARED) -O3  $(OMP) -c lib/gtknn/cuda_distances.cu

utils.o: lib/gtknn/utils.cu lib/gtknn/utils.cuh 
	nvcc $(GENCODE_FLAGS) $(SHARED) -O3 $(OMP)  -c lib/gtknn/utils.cu

clean:
	rm *.o lazynn_rf
