// Vector dot product.
//
// The problem with dot-product-locks.cu is the final sum: float sums are not
// associative, so the GPU result is different from the CPU result. This version
// solves this problem by locking the result variable in a deterministic order
// based on the block number.
#include <stdio.h>
#include <math.h>

#include "../common/cuda-error.h"

#define imin(a, b) (a < b ? a : b)

#define sum_squares(x) (x*(x+1)*(2*x+1)/6)

// Number of float numbers (36 millions).
const int N = 33 * 1024 * 1024;

// Number of threads for each block.
const int THREADS = 256;

// By default there are 32 blocks, but if the source array is shorter there may
// be fewer than 32 blocks.
const int BLOCKS = imin(32, (N + THREADS - 1) / THREADS );

__global__ void dot(volatile int *lock, float *a, float *b, float *c) {
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

    // Only the first thread of all block partecipate to sum c.
    if (cacheIndex == 0) {
        // The first thread of the first block start the sum.
        if (blockIdx.x == 0) {
            *c += cache[0];
            printf("Sum of block %d\n", blockIdx.x);

            __threadfence(); // Make sure other threads will se c.

            // Pass the lock to the next block.
            atomicAdd((int *)lock, 1);

        } else {
            // Spin until the lock contains the thread's block number.
            do {
                __threadfence();
            } while (*lock != blockIdx.x);

            printf("Sum of block %d\n", blockIdx.x);
            *c += cache[0];

            __threadfence(); // Make sure other threads will se c.

            // Pass the lock to the next block.
            atomicAdd((int *)lock, 1);
        }
    }
}

int main() {
    float *a, *b, c;
    float *dev_a, *dev_b, *dev_c;
    int *dev_lock;

    // Allocate memory host side.
    a = (float *) malloc(sizeof(float) * N);
    b = (float *) malloc(sizeof(float) * N);

    // Allocate memory device side.
    HANDLE_ERROR(cudaMalloc((void **)&dev_a, sizeof(float) * N));
    HANDLE_ERROR(cudaMalloc((void **)&dev_b, sizeof(float) * N));
    HANDLE_ERROR(cudaMalloc((void **)&dev_c, sizeof(float)));
    HANDLE_ERROR(cudaMemset(dev_c, 0, sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_lock, sizeof(int)));
    HANDLE_ERROR(cudaMemset(dev_lock, 0, sizeof(int)));

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

    // Launch kernel.
    dot<<<BLOCKS, THREADS>>>(dev_lock, dev_a, dev_b, dev_c);

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
    HANDLE_ERROR(cudaFree(dev_lock));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));
    free(a);
    free(b);

    return 0;
}
