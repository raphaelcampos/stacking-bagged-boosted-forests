################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
./Dataset.o \
./cuLazyNN_RF.o\
./cuNearestNeighbors.o 

CPP_SRCS += \
$(LazyNN_PATH)Dataset.cpp \
$(LazyNN_PATH)cuLazyNN_RF.cpp\
$(LazyNN_PATH)cuNearestNeighbors.cpp 

CPP_DEPS += \
./Dataset.d \
./cuLazyNN_RF.d \
./cuNearestNeighbors.d 

OBJS += \
./Dataset.o \
./cuLazyNN_RF.o \
./cuNearestNeighbors.o 

# Each subdirectory must supply rules for building sources it contributes
%.o: $(LazyNN_PATH)%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -O3 -Xcompiler -fopenmp -Xcompiler --fast-math -Xcompiler -lm $(GENCODE_FLAGS) -odir "" -M -o "$(@:%.o=%.d)" "$<"  -L$(GTKNN_PATH)lib/cudpp-2.1/lib/ -lcudpp  -I$(GTKNN_PATH)lib/cudpp-2.1/include/ -I$(GTKNN_PATH) -I~/opencv-3.0.0/include/ -L~/opencv-3.0.0/release/lib/ -L$(GTKNN_PATH)lib/cudpp-2.1/lib/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_stitching
	nvcc --device-c -O3 -Xcompiler -fopenmp -Xcompiler --fast-math -Xcompiler -lm $(GENCODE_FLAGS) -x cu -o  "$@" "$<"  -L$(GTKNN_PATH)lib/cudpp-2.1/lib/ -lcudpp -I$(GTKNN_PATH)lib/cudpp-2.1/include/ -I$(GTKNN_PATH) -I~/opencv-3.0.0/include/ -L~/opencv-3.0.0/release/lib/ -L$(GTKNN_PATH)lib/cudpp-2.1/lib/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_stitching

	@echo 'Finished building: $<'
	@echo ' '


