#include <stdio.h>
#include <stdlib.h>

#include "cuda-error.h"

#define imin(a, b) (a < b ? a : b)

// Number of float numbers (36 millions).
const int N = 33 * 1024 * 1024;

// Number of threads for each block.
const int THREADS = 256;

// By default there are 32 blocks, but if the source array is shorter there may
// be fewer than 32 blocks.
const int BLOCKS = imin(32, (N + THREADS - 1) / THREADS );

__global__ void dot(float *a, float *b, float *c) {
    __shared__ float cache[THREADS];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float tmp = 0;
    while (tid < N) {
        tmp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    // Set cache value.
    cache[cacheIndex] = tmp;

    // Sync threads in the block.
    __syncthreads();

    // For reductions, threadsPerBlock must be a power of 2 because of the
    // following code
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];

        __syncthreads();
        i /= 2;
    }

    // Only one thread of the block will write to c.
    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}

float malloc_test(size_t size, float *result) {
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;
    float elapsed;

    // Create event to measure elapsed time.
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    // Allocate memory host side.
    a = (float *) malloc(sizeof(float) * size);
    b = (float *) malloc(sizeof(float) * size);
    partial_c = (float *) malloc(sizeof(float) * BLOCKS);

    // Allocate memory device side.
    HANDLE_ERROR(cudaMalloc((void **)&dev_a, sizeof(float) * size));
    HANDLE_ERROR(cudaMalloc((void **)&dev_b, sizeof(float) * size));
    HANDLE_ERROR(cudaMalloc((void **)&dev_partial_c,
                sizeof(float) * BLOCKS));

    // dev_partial_c and partial_c stores the partial results, one for each
    // block, this is why it is large sizeof(float) * BLOCKS.

    // Fill the input arrays.
    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i*2;
    }

    // Start timer.
    HANDLE_ERROR(cudaEventRecord(start));

    // Copy input array to the device.
    HANDLE_ERROR(cudaMemcpy(dev_a, a, sizeof(float) * size,
                cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, sizeof(float) * size,
                cudaMemcpyHostToDevice));

    // Run kernel.
    dot<<<BLOCKS, THREADS>>>(dev_a, dev_b, dev_partial_c);

    // Copy output array to host.
    HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c,
                sizeof(float) * BLOCKS, cudaMemcpyDeviceToHost));

    // Stop timer.
    HANDLE_ERROR(cudaEventRecord(stop));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop));

    // Make last reduction on host.
    c = 0;
    for (int i = 0; i < BLOCKS; i++)
        c += partial_c[i];
    *result = c;

    // Free memory both host and device side.
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_partial_c));
    free(a);
    free(b);
    free(partial_c);

    return elapsed;
}

float mapped_memory_test(size_t size, float *result) {
    // Create event to measure elapsed time.
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    // Allocate mapped page-locked memory on the host.
    float *a, *b, *partial_c;
    HANDLE_ERROR(cudaHostAlloc((void **)&a, size * sizeof(float),
                cudaHostAllocMapped | cudaHostAllocWriteCombined));
    HANDLE_ERROR(cudaHostAlloc((void **)&b, size * sizeof(float),
                cudaHostAllocMapped | cudaHostAllocWriteCombined));
    // partial_c stores the partial results, one for each block, this is why it
    // is large sizeof(float) * BLOCKS.
    HANDLE_ERROR(cudaHostAlloc((void **)&partial_c, BLOCKS * sizeof(float),
                cudaHostAllocMapped));

    // Fill the input arrays.
    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i*2;
    }

    // Get pointers in the device for mapped host memory.
    float *dev_a, *dev_b, *dev_partial_c;
    HANDLE_ERROR(cudaHostGetDevicePointer((void **)&dev_a, a, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer((void **)&dev_b, b, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer((void **)&dev_partial_c, partial_c, 0));

    // Start timer.
    HANDLE_ERROR(cudaEventRecord(start));

    // Run kernel.
    dot<<<BLOCKS, THREADS>>>(dev_a, dev_b, dev_partial_c);

    // Stop timer.
    HANDLE_ERROR(cudaEventRecord(stop));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsed;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop));

    // Make last reduction on host. Because we synchronized with the device, we
    // are sure to read correct value in partial_c.
    float c = 0;
    for (int i = 0; i < BLOCKS; i++)
        c += partial_c[i];
    *result = c;

    // Free memory both host and device side.
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    HANDLE_ERROR(cudaFreeHost(a));
    HANDLE_ERROR(cudaFreeHost(b));
    HANDLE_ERROR(cudaFreeHost(partial_c));

    return elapsed;
}

int main() {
    // Get current device.
    int device;
    HANDLE_ERROR(cudaGetDevice(&device));

    // Device must support mapped page-locked host memory.
    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, device));
    if (prop.canMapHostMemory != 1) {
        puts("Device cannot map host memory into the CUDA address space.");
        return EXIT_FAILURE;
    }

    // Enable mapped memory.
    HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));

    float result;
    float elapsed = malloc_test(N, &result);
    printf("Result with cudaMalloc: %.6g\n", result);
    printf("Time using cudaMalloc: %3.1f ms\n", elapsed);

    elapsed = mapped_memory_test(N, &result);
    printf("Result with cudaHostMalloc: %.6g\n", result);
    printf("Time using cudaHostMalloc: %3.1f ms\n", elapsed);

    return EXIT_SUCCESS;
}
