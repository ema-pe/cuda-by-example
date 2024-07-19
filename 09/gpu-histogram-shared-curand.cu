// Histogram calculation of a randomly generated data.
//
// This is the GPU version using global and shared memory atomics. This version
// also initializes input data directly on the GPU using cuRAND.
#include <stdlib.h>
#include <stdio.h>

#include <curand.h>

#include "cuda-error.h"

const size_t HISTO_SIZE = 256;

const size_t DATA_SIZE = 500 * 1024 * 1024; // 500 MB.

__global__ void histo_kernel(unsigned char *block, const size_t block_size, unsigned int *histogram) {
    // To reduce the concurrent access to the global histogram array, we use a
    // shared histogram (a copy shared for all threads of a single block). We
    // then merge the shared histograms into the global histogram.
    __shared__ unsigned int shared_hist[HISTO_SIZE];

    // This assumes each block has 256 linear threads and HISTO_SIZE == 256.
    shared_hist[threadIdx.x] = 0;

    // Wait all threads have initialized the shared histogram.
    __syncthreads();

    // Linearized offset in block.
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Total number of threads launched for this kernel.
    int tot_theads = blockDim.x * gridDim.x;

    while (i < block_size) {
        // Add atomically 1 to the histogram, it is needed because other threads
        // can access to the same address.
        atomicAdd(&(shared_hist[block[i]]), 1);

        // Move to the next block data.
        i += tot_theads;
    }

    // Wait all threads have updated the shared histogram.
    __syncthreads();

    // Each thread updates a global histogram entry. Since the kernel starts
    // with 256 threads, we cover all histogram entries.
    atomicAdd(&(histogram[threadIdx.x]), shared_hist[threadIdx.x]);
}

static void cuRANDHandleError(curandStatus_t st, const char *file, int line) {
    if (st != CURAND_STATUS_SUCCESS) {
        printf("error in %s at line %d\n", file, line );
        exit( EXIT_FAILURE );
    }
}

#define CR_HANDLE_ERROR(err) (cuRANDHandleError( err, __FILE__, __LINE__ ))

int main() {
    // Create events for timing.
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    // Allocate the data block on the device and copy the host version to it.
    unsigned char *dev_block;
    HANDLE_ERROR(cudaMalloc((void **)&dev_block, DATA_SIZE));

    // Create RNG.
    curandGenerator_t rand_gen;
    CR_HANDLE_ERROR(curandCreateGenerator(&rand_gen, CURAND_RNG_PSEUDO_DEFAULT));

    // Use RNG to fill data block on the device.
    //
    // Note we have a data block of unsigned char (8 bit), but curandGenerate
    // generate 32 bit integers.
    CR_HANDLE_ERROR(curandGenerate(rand_gen, (unsigned int *)dev_block, DATA_SIZE / 4));

    // Allocate the histogram on the device and set it to zero.
    unsigned int *dev_histogram;
    HANDLE_ERROR(cudaMalloc((void **)&dev_histogram, HISTO_SIZE * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMemset(dev_histogram, 0, HISTO_SIZE * sizeof(unsigned int)));

    // Call kernel.
    //
    // The block dimension is just a linear 256 threads, one for each histogram
    // bin.
    //
    // The grid dimension is the number of multiprocessor * 2.
    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));

    // Equivalent.
    //dim3 grid(prop.multiProcessorCount*8);
    //dim3 block(HISTO_SIZE);
    int blocks = prop.multiProcessorCount * 8;
    int threads = HISTO_SIZE;
    printf("Blocks = %d\tThreads = %d\n", blocks, threads);
    histo_kernel<<<blocks, threads>>>(dev_block, DATA_SIZE, dev_histogram);

    // Copy the histogram from the device to the host.
    unsigned int histogram[256];
    HANDLE_ERROR(cudaMemcpy(histogram, dev_histogram,
                HISTO_SIZE * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // Stop timer.
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));

    // Calculate and print elapsed time.
    float elapsed;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop));
    printf("Time to generate: %3.1f ms\n", elapsed);

    // Check the calculate histogram is correct.
    size_t total = 0;
    for (int i = 0; i < HISTO_SIZE; i++)
        total += histogram[i];

    if (total != DATA_SIZE)
        printf("There are %ld elements that doesn't match the input block data\n", total);

    // Free all allocated data on both device and host.
    CR_HANDLE_ERROR(curandDestroyGenerator(rand_gen));
    HANDLE_ERROR(cudaFree(dev_block));
    HANDLE_ERROR(cudaFree(dev_histogram));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    return EXIT_SUCCESS;
}
