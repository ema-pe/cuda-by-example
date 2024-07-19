.PHONY = clean all

NVCC_FLAGS := -Werror all-warnings -lm -I ../common/

NVPROF_FLAGS := --force-overwrite --analysis-metrics

EXECUTABLES := bandwidthtest bench_memcpy multiple_stream multiple_stream-fixed \
	single_stream

PROFILING := bandwidthtest.nvvp bench_memcpy.nvvp multiple_stream.nvvp \
	multiple_stream-fixed.nvvp single_stream.nvvp

all : $(EXECUTABLES)

clean :
	@rm --verbose --force $(EXECUTABLES) $(PROFILING)

bandwidthtest : bandwidthtest.cu
	nvcc $(NVCC_FLAGS) $^ --output-file $@

bandwidthtest.nvvp : bandwidthtest
	nvprof $(NVPROF_FLAGS) --export-profile $@ ./$^

bench_memcpy : bench_memcpy.cu
	nvcc $(NVCC_FLAGS) $^ --output-file $@

bench_memcpy.nvvp : bench_memcpy
	nvprof $(NVPROF_FLAGS) --export-profile $@ ./$^

multiple_stream : multiple_stream.cu
	nvcc $(NVCC_FLAGS) $^ --output-file $@

multiple_stream.nvvp : multiple_stream
	nvprof $(NVPROF_FLAGS) --export-profile $@ ./$^

multiple_stream-fixed : multiple_stream-fixed.cu
	nvcc $(NVCC_FLAGS) $^ --output-file $@

multiple_stream-fixed.nvvp : multiple_stream-fixed
	nvprof $(NVPROF_FLAGS) --export-profile $@ ./$^

single_stream : single_stream.cu
	nvcc $(NVCC_FLAGS) $^ --output-file $@

single_stream.nvvp : single_stream
	nvprof $(NVPROF_FLAGS) --export-profile $@ ./$^

