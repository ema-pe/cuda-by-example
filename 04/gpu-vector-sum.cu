#include <stdio.h>

#include "../common/cuda-error.h"

#define N 10

__global__ void add(int *a, int *b, int *c) {
    int tid = blockIdx.x;

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

    add<<<N, 1>>>(dev_a, dev_b, dev_c);

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
