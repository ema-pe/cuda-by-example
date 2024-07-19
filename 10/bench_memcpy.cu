#include <stdlib.h>
#include <stdio.h>

#include "cuda-error.h"

// Number of cudaMemcpy operations make in each test.
const unsigned int COPIES = 100;

// Size of data.
const size_t SIZE = 4 * 1024 * 1024; // 10 MB.

// Returns the time (in ms) to make several dummy cudaMemcpy copies with data of
// the given size allocated with malloc and in the given copy direction.
float cuda_malloc_test(int size, bool copyToDevice) {
    // Create events uset to track the time of the copies.
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    // Allocate host data.
    int *data = (int *)malloc(size * sizeof(data));
    HANDLE_NULL(data);

    // Allocate device data.
    int *dev_data;
    HANDLE_ERROR(cudaMalloc((void **)&dev_data, size * sizeof(data)));

    // Track the events and do the copies.
    HANDLE_ERROR(cudaEventRecord(start));
    for (unsigned int i = 0; i < COPIES; i++) {
        if (copyToDevice)
            HANDLE_ERROR(cudaMemcpy(dev_data, data, size * sizeof(data),
                        cudaMemcpyHostToDevice));
        else
            HANDLE_ERROR(cudaMemcpy(data, dev_data, size * sizeof(data),
                        cudaMemcpyDeviceToHost));
    }
    HANDLE_ERROR(cudaEventRecord(stop));
    HANDLE_ERROR(cudaEventSynchronize(stop));

    // Get elapsed time.
    float elapsed;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop));

    // Free all allocated data.
    HANDLE_ERROR(cudaFree(dev_data));
    free(data);
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    return elapsed;
}

// Returns the time (in ms) to make several dummy cudaMemcpy copies with data of
// the given size allocated with cudaHostAlloc and in the given copy direction.
//
// The cudaHostAlloc flags can be specified with alloc_flags.
float cuda_host_alloc_test(int size, bool copyToDevice, unsigned int alloc_flags) {
    // Create events uset to track the time of the copies.
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    // Allocate host data.
    int *data;
    HANDLE_ERROR(cudaHostAlloc((void **)&data, size * sizeof(data),
                alloc_flags));

    // Allocate device data.
    int *dev_data;
    HANDLE_ERROR(cudaMalloc((void **)&dev_data, size * sizeof(data)));

    // Track the events and do the copies.
    HANDLE_ERROR(cudaEventRecord(start));
    for (unsigned int i = 0; i < COPIES; i++) {
        if (copyToDevice)
            HANDLE_ERROR(cudaMemcpy(dev_data, data, size * sizeof(data),
                        cudaMemcpyHostToDevice));
        else
            HANDLE_ERROR(cudaMemcpy(data, dev_data, size * sizeof(data),
                        cudaMemcpyDeviceToHost));
    }
    HANDLE_ERROR(cudaEventRecord(stop));
    HANDLE_ERROR(cudaEventSynchronize(stop));

    // Get elapsed time.
    float elapsed;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop));

    // Free all allocated data.
    HANDLE_ERROR(cudaFree(dev_data));
    HANDLE_ERROR(cudaFreeHost(data));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    return elapsed;
}

int main() {
    float elapsed;
    float mb = (float)100 * SIZE * sizeof(int) / 1024 / 1024;

    // Benchmark with malloc and host->device copy.
    elapsed = cuda_malloc_test(SIZE, true);
    printf("Time using malloc (host->device): %4.1f ms (%4.1f MB/s)\n", elapsed,
            mb / (elapsed / 1000));

    // Benchmark with malloc and device->host copy.
    elapsed = cuda_malloc_test(SIZE, false);
    printf("Time using malloc (device->host): %4.1f ms (%4.1f MB/s)\n", elapsed,
            mb / (elapsed / 1000));

    // Benchmark with cudaHostAlloc with default flag and host->device copy.
    elapsed = cuda_host_alloc_test(SIZE, true, cudaHostAllocDefault);
    printf("Time using cudaHostAlloc with default flags (host->device): %4.1f ms (%4.1f MB/s)\n",
            elapsed, mb / (elapsed / 1000));

    // Benchmark with cudaHostAlloc with default flag and device->host copy.
    elapsed = cuda_host_alloc_test(SIZE, false, cudaHostAllocDefault);
    printf("Time using cudaHostAlloc with default flags (device->host): %4.1f ms (%4.1f MB/s)\n",
            elapsed, mb / (elapsed / 1000));

    // Benchmark with cudaHostAlloc with WriteCombined flag and host->device copy.
    elapsed = cuda_host_alloc_test(SIZE, true, cudaHostAllocDefault | cudaHostAllocWriteCombined);
    printf("Time using cudaHostAlloc with WriteCombined flag (host->device): %4.1f ms (%4.1f MB/s)\n",
            elapsed, mb / (elapsed / 1000));

    return EXIT_SUCCESS;
}
