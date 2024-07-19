.PHONY = clean all

SOURCES := $(wildcard *.cu)
OUTPUTS := $(SOURCES:.cu=)

NVCC_FLAGS := -Werror all-warnings -lm -I ../common/

all : $(OUTPUTS)

$(OUTPUTS) : % : %.cu
	nvcc $(NVCC_FLAGS) $^ --output-file $@

ray-tracing: ray-tracing.cu
	nvcc $(NVCC_FLAGS) ../common/lodepng.cpp $^ --output-file $@

ray-tracing-constant: ray-tracing-constant.cu
	nvcc $(NVCC_FLAGS) ../common/lodepng.cpp $^ --output-file $@

clean :
	rm --force $(OUTPUTS) *.png *.gif
