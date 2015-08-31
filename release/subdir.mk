################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
./Dataset.o \
./cuLazyNN_RF.o

CPP_SRCS += \
$(LazyNN_PATH)Dataset.cpp \
$(LazyNN_PATH)cuLazyNN_RF.cpp

CPP_DEPS += \
./Dataset.d \
./cuLazyNN_RF.d 

OBJS += \
./Dataset.o \
./cuLazyNN_RF.o

# Each subdirectory must supply rules for building sources it contributes
%.o: $(LazyNN_PATH)%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -O3 $(GENCODE_FLAGS) -odir "" -M -o "$(@:%.o=%.d)" "$<"  -L$(GTKNN_PATH)lib/cudpp-2.1/lib/ -lcudpp  -I$(GTKNN_PATH)lib/cudpp-2.1/include/ -I$(GTKNN_PATH)
	nvcc --device-c -O3 $(GENCODE_FLAGS) -x cu -o  "$@" "$<"  -L$(GTKNN_PATH)lib/cudpp-2.1/lib/ -lcudpp -I$(GTKNN_PATH)lib/cudpp-2.1/include/ -I$(GTKNN_PATH)

	@echo 'Finished building: $<'
	@echo ' '


