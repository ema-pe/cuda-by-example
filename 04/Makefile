.PHONY = clean all

SOURCES := $(wildcard *.cu)
OUTPUTS := $(SOURCES:.cu=)

all : $(OUTPUTS)

$(OUTPUTS) : % : %.cu
	nvcc -Werror all-warnings $^ --output-file $@

cpu-julia : cpu-julia.cu
	nvcc -x c -Werror all-warnings -lm -I ../common/ ../common/lodepng.cpp $^ --output-file $@

gpu-julia : gpu-julia.cu
	nvcc -Werror all-warnings -lm -I ../common/ ../common/lodepng.cpp $^ --output-file $@

clean :
	rm --force $(OUTPUTS) *.nvprof *.png *.gif
