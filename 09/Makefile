.PHONY = clean all

SOURCES := $(wildcard *.cu)
OUTPUTS := $(SOURCES:.cu=)

NVCC_FLAGS := -Werror all-warnings -lm -I ../common/

all : $(OUTPUTS)

$(OUTPUTS) : % : %.cu
	nvcc $(NVCC_FLAGS) $^ --output-file $@

gpu-histogram-shared-curand : gpu-histogram-shared-curand.cu
	nvcc $(NVCC_FLAGS) -lcurand $^ --output-file $@

clean :
	rm --force $(OUTPUTS) *.png *.gif
