#include <stdio.h>

#include "../common/cuda-error.h"

#define N 1000

__global__ void add(int *a, int *b, int *c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

int main(void) {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    HANDLE_ERROR(cudaMalloc((void**)&dev_a, sizeof(int)*N));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, sizeof(int)*N));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)*N));

    // Init arrays.
    for (int i = 0; i < N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }

    // Copy host arrays to device.
    HANDLE_ERROR(cudaMemcpy(dev_a, a, sizeof(int)*N, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, sizeof(int)*N, cudaMemcpyHostToDevice));

    // 128 threads for each block.
    // Number of blocks depends on the array's size.
    add<<<(N+127)/128, 128>>>(dev_a, dev_b, dev_c);

    // Copy device result array to host.
    HANDLE_ERROR(cudaMemcpy(c, dev_c, sizeof(int)*N, cudaMemcpyDeviceToHost));

    // Display results.
    for (int i = 0; i < N; i++)
        printf("%2d + %2d = %2d\n", a[i], b[i], c[i]);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
