#include <stdio.h>
#include <math.h>

#include "../common/cuda-error.h"

#define imin(a, b) (a < b ? a : b)

#define sum_squares(x) (x*(x+1)*(2*x+1)/6)


const int N = 33 * 1024;

const int threadsPerBlock = 256;

// By default there are 32 blocks, but if the source array is shorter there may
// be fewer than 32 blocks.
const int blocksPerGrid = imin(32, (N+threadsPerBlock-1) / threadsPerBlock);

__global__ void dot(float *a, float *b, float *c) {
    __shared__ float cache[threadsPerBlock];
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

int main() {
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    // Allocate memory host side.
    a = (float *) malloc(sizeof(float) * N);
    b = (float *) malloc(sizeof(float) * N);
    partial_c = (float *) malloc(sizeof(float) * blocksPerGrid);

    // Allocate memory device side.
    HANDLE_ERROR(cudaMalloc((void **)&dev_a, sizeof(float) * N));
    HANDLE_ERROR(cudaMalloc((void **)&dev_b, sizeof(float) * N));
    HANDLE_ERROR(cudaMalloc((void **)&dev_partial_c,
                sizeof(float) * blocksPerGrid));

    // Fill the input arrays.
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i*2;
    }

    // Copy input array to the device.
    HANDLE_ERROR(cudaMemcpy(dev_a, a, sizeof(float) * N,
                cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, sizeof(float) * N,
                cudaMemcpyHostToDevice));

    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

    // Copy output array to host.
    HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c,
                sizeof(float) * blocksPerGrid, cudaMemcpyDeviceToHost));

    // Make last reduction on host.
    c = 0;
    for (int i = 0; i < blocksPerGrid; i++)
        c += partial_c[i];

    printf("(GPU value) %.6g = %.6g (CPU value)\n", c,
            2 * sum_squares((float)(N-1)));

    // Free memory both host and device side.
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_partial_c));
    free(a);
    free(b);
    free(partial_c);

    return 0;
}
