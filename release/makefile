################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

GTKNN_PATH=../lib/gtknn/
LazyNN_PATH=../

# All of the sources participating in the build are defined here
-include ../lib/gtknn/Release/sources.mk
-include subdir-gtknn.mk
-include objects-gtknn.mk

-include subdir.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(CU_DEPS)),)
-include $(CU_DEPS)
endif
ifneq ($(strip $(C++_DEPS)),)
-include $(C++_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
ifneq ($(strip $(CC_DEPS)),)
-include $(CC_DEPS)
endif
ifneq ($(strip $(CPP_DEPS)),)
-include $(CPP_DEPS)
endif
ifneq ($(strip $(CXX_DEPS)),)
-include $(CXX_DEPS)
endif
ifneq ($(strip $(C_UPPER_DEPS)),)
-include $(C_UPPER_DEPS)
endif
endif

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: lazynn_rf secondary-outputs

# Tool invocations
lazynn_rf: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: NVCC Linker'
	nvcc --relocatable-device-code=true -gencode arch=compute_20,code=sm_20 -link -o "lazynn_rf" $(LazyNN_PATH)main.cu $(OBJS) $(USER_OBJS) $(LIBS) -I$(GTKNN_PATH) -I$(GTKNN_PATH)lib/cudpp-2.1/include/ -L$(GTKNN_PATH)lib/cudpp-2.1/lib/ -lcudpp
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(CU_DEPS)$(OBJS)$(C++_DEPS)$(C_DEPS)$(CC_DEPS)$(CPP_DEPS)$(EXECUTABLES)$(CXX_DEPS)$(C_UPPER_DEPS) lazynn_rf
	-@echo ' '

secondary-outputs: $(CUBINS)

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets
