// Vector dot product.
//
// The difference with dot-product.cu is that the kernel computes a single
// result instead of a block of partial results that must be reduced on the
// host.
#include <stdio.h>
#include <math.h>

#include "../common/cuda-error.h"

#include "lock.h"

#define imin(a, b) (a < b ? a : b)

#define sum_squares(x) (x*(x+1)*(2*x+1)/6)

// Number of float numbers (36 millions).
const int N = 33 * 1024 * 1024;

// Number of threads for each block.
const int THREADS = 256;

// By default there are 32 blocks, but if the source array is shorter there may
// be fewer than 32 blocks.
const int BLOCKS = imin(32, (N + THREADS - 1) / THREADS );

// c is volatile because it is read and written by thread of differents blocks,
// so the compiler must not optimize the access of this variable (eg. L1 cache).
__global__ void dot(mutex_t *m, float *a, float *b, volatile float *c) {
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
    if (cacheIndex == 0) {
        mutexLock(m);
        printf("Sum of block %d\n", blockIdx.x); // DEBUG.
        *c += cache[0];
        mutexUnlock(m);
    }
}

int main() {
    float *a, *b, c;
    float *dev_a, *dev_b, *dev_c;

    // Allocate memory host side.
    a = (float *) malloc(sizeof(float) * N);
    b = (float *) malloc(sizeof(float) * N);

    // Allocate memory device side.
    HANDLE_ERROR(cudaMalloc((void **)&dev_a, sizeof(float) * N));
    HANDLE_ERROR(cudaMalloc((void **)&dev_b, sizeof(float) * N));
    HANDLE_ERROR(cudaMalloc((void **)&dev_c, sizeof(float)));
    HANDLE_ERROR(cudaMemset(dev_c, 0, sizeof(float)));

    // Fill the input arrays.
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i*2;
    }

    // Create events and start timer.
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start));

    // Copy input array to the device.
    HANDLE_ERROR(cudaMemcpy(dev_a, a, sizeof(float) * N,
                cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, sizeof(float) * N,
                cudaMemcpyHostToDevice));

    // Create mutex lock and launch kernel.
    mutex_t *mutex = mutexCreate();
    dot<<<BLOCKS, THREADS>>>(mutex, dev_a, dev_b, dev_c);

    // Copy output array to host.
    HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(float), cudaMemcpyDeviceToHost));

    // Stop timer and get elapsed time.
    HANDLE_ERROR(cudaEventRecord(stop));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsed;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop));

    printf("GPU Time: %3.1f ms\n", elapsed);
    printf("(GPU value) %.6g = %.6g (CPU value)\n", c,
            2 * sum_squares((float)(N-1)));

    // Free memory both host and device side.
    mutexDestroy(mutex);
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));
    free(a);
    free(b);

    return 0;
}
