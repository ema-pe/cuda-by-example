.PHONY = clean all

NVCC_FLAGS := -Werror all-warnings -lm -I ../common/

NVPROF_FLAGS := --force-overwrite --analysis-metrics --metrics all

EXECUTABLES := dot-product dot-product-deterministic dot-product-locks \
	hash-cpu hash-gpu

PROFILING := dot-product.nvvp dot-product-deterministic.nvvp \
	dot-product-locks.nvvp hash-gpu.nvvp

all : $(EXECUTABLES)

clean :
	@rm --verbose --force $(EXECUTABLES) $(PROFILING)

dot-product : dot-product.cu
	nvcc $(NVCC_FLAGS) $^ --output-file $@

dot-product.nvvp : dot-product
	nvprof $(NVPROF_FLAGS) --export-profile $@ ./$^

dot-product-deterministic : dot-product-deterministic.cu
	nvcc $(NVCC_FLAGS) $^ --output-file $@

dot-product-deterministic.nvvp : dot-product-deterministic
	nvprof $(NVPROF_FLAGS) --export-profile $@ ./$^

dot-product-locks : dot-product-locks.cu lock.h
	nvcc $(NVCC_FLAGS) $^ --output-file $@

dot-product-locks.nvvp : dot-product-locks
	nvprof $(NVPROF_FLAGS) --export-profile $@ ./$^

hash-cpu : hash-cpu.cu
	nvcc $(NVCC_FLAGS) $^ --output-file $@

hash-gpu : hash-gpu.cu lock.h
	nvcc $(NVCC_FLAGS) $^ --output-file $@

hash-gpu.nvvp : hash-gpu
	nvprof $(NVPROF_FLAGS) --export-profile $@ ./$^


