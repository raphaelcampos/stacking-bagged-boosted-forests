################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
#./cuda_distances.o \
./inverted_index.o \
./knn.o \
#./partial_bitonic_sort.o \
#./utils.o 

#CU_SRCS += \
#$(GTKNN_PATH)cuda_distances.cu \
$(GTKNN_PATH)inverted_index.cu \
$(GTKNN_PATH)knn.cu \
#$(GTKNN_PATH)partial_bitonic_sort.cu \
#$(GTKNN_PATH)utils.cu 

#CU_DEPS += \
#./cuda_distances.d \
./inverted_index.d \
./knn.d \
#./partial_bitonic_sort.d \
#./utils.d 

OBJS += \
./cuda_distances.o \
./inverted_index.o \
./knn.o \
./partial_bitonic_sort.o \
./utils.o 


# Each subdirectory must supply rules for building sources it contributes
%.o: $(GTKNN_PATH)%.cu
#	@echo 'Building file: $<'
#	@echo 'Invoking: NVCC Compiler'
#	nvcc -O3 -gencode arch=compute_20,code=sm_30 -odir "" -M -o "$(@:%.o=%.d)" "$<"  -L/usr/local/cudpp-2.1/lib/ -lcudpp  -I/usr/local/cudpp-2.1/include/
#	nvcc --device-c -O3 -gencode arch=compute_20,code=sm_30  -x cu -o  "$@" "$<"  -L/usr/local/cudpp-2.1/lib/ -lcudpp -I/usr/local/cudpp-2.1/include/
	nvcc -O3 $(GENCODE_FLAGS) -odir "" -M -o "$(@:%.o=%.d)" "$<"  -L$(GTKNN_PATH)lib/cudpp-2.1/lib/ -lcudpp  -I$(GTKNN_PATH)lib/cudpp-2.1/include/
	nvcc --device-c -O3 $(GENCODE_FLAGS) -x cu -o  "$@" "$<"  -L$(GTKNN_PATH)lib/cudpp-2.1/lib/ -lcudpp -I$(GTKNN_PATH)lib/cudpp-2.1/include/ 

#	@echo 'Finished building: $<'
#	@echo ' '


