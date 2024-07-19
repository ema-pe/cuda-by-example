.PHONY = clean all

SOURCES := $(wildcard *.cu)
OUTPUTS := $(SOURCES:.cu=)

NVCC_FLAGS := -Werror all-warnings -lm -I ../common/

all : $(OUTPUTS)

$(OUTPUTS) : % : %.cu
	nvcc $(NVCC_FLAGS) $^ --output-file $@

gpu-gif-threads : gpu-gif-threads.cu
	nvcc $(NVCC_FLAGS) $^ --output-file $@

julia-wrong-sync : julia-wrong-sync.cu
	nvcc $(NVCC_FLAGS) ../common/lodepng.cpp $^ --output-file $@

julia-fixed-sync : julia-fixed-sync.cu
	nvcc $(NVCC_FLAGS) ../common/lodepng.cpp $^ --output-file $@

clean :
	rm --force $(OUTPUTS) *.png *.gif
