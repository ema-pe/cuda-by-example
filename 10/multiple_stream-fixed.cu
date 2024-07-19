#include <stdio.h>
#include <stdlib.h>

#include "cuda-error.h"

const size_t CHUNK_SIZE = 1024 * 1024; // 1 million of elements.

const size_t DATA_SIZE = CHUNK_SIZE * 20; // 20 million of elements.

// Just a dummy kernel to do some calculation based on array a and b and store
// the result on c.
__global__ void kernel( int *a, int *b, int *c ) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < CHUNK_SIZE) {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;

        float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;

        c[idx] = (as + bs) / 2;
    }
}

int main() {
    // Check if the device supports concurrent copy and kernel execution.
    cudaDeviceProp prop;
    int device;
    HANDLE_ERROR(cudaGetDevice(&device));
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, device));

    // prop.deviceOverlap is deprecated.
    printf("Device supports concurrent copy and kernel execution: ");
    switch (prop.asyncEngineCount) {
        case 0:
            puts("no");
            break;
        case 1:
            puts("yes (one direction)");
            break;
        case 2:
            puts("yes (both directions)");
            break;
    }

    // Create streams.
    cudaStream_t stream1, stream2;
    HANDLE_ERROR(cudaStreamCreate(&stream1));
    HANDLE_ERROR(cudaStreamCreate(&stream2));

    // Initialize timers and start a timer.
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, stream2));

    // Allocate memory on both host and device. On the host we allocate
    // page-locked memory because cudaMemcpyAsync requires it.
    int *a, *b, *c;
    int *dev_a1, *dev_b1, *dev_c1; // Buffers for stream1.
    int *dev_a2, *dev_b2, *dev_c2; // Buffers for stream2.
    HANDLE_ERROR(cudaMalloc((void **)&dev_a1, CHUNK_SIZE * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_b1, CHUNK_SIZE * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_c1, CHUNK_SIZE * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_a2, CHUNK_SIZE * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_b2, CHUNK_SIZE * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_c2, CHUNK_SIZE * sizeof(int)));
    HANDLE_ERROR(cudaHostAlloc((void **)&a, DATA_SIZE * sizeof(int),
                cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void **)&b, DATA_SIZE * sizeof(int),
                cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void **)&c, DATA_SIZE * sizeof(int),
                cudaHostAllocDefault));

    // Initialise the two input arrays.
    for (int i = 0; i < DATA_SIZE; i++) {
        a[i] = rand();
        b[i] = rand();
    }

    // The input arrays are too big: we split them into chunks and launch the
    // cudaMemcpy and kernel for each chunk.
    //
    // We enqueue the tasks (memcpy and kernel exec) to the two streams. This
    // will improve the performance because the copy and kernel engines can work
    // without waiting too much. In any case we need to wait on the host side
    // for the stream tasks to finish.
    for (int chunk = 0; chunk < DATA_SIZE; chunk += CHUNK_SIZE*2) {
        // Copy the chunks of input array a to the device.
        HANDLE_ERROR(cudaMemcpyAsync(dev_a1, a + chunk, CHUNK_SIZE * sizeof(int),
                    cudaMemcpyHostToDevice, stream1));
        HANDLE_ERROR(cudaMemcpyAsync(dev_a2, a + chunk + CHUNK_SIZE, CHUNK_SIZE * sizeof(int),
                    cudaMemcpyHostToDevice, stream2));

        // Copy the chunks of input array b to the device.
        HANDLE_ERROR(cudaMemcpyAsync(dev_b1, b + chunk, CHUNK_SIZE * sizeof(int),
                    cudaMemcpyHostToDevice, stream1));
        HANDLE_ERROR(cudaMemcpyAsync(dev_b2, b + chunk + CHUNK_SIZE, CHUNK_SIZE * sizeof(int),
                    cudaMemcpyHostToDevice, stream2));

        // Launch the kernels.
        const int blocks = CHUNK_SIZE / 256;
        const int threads = 256;
        kernel<<<blocks, threads, 0, stream1>>>(dev_a1, dev_b1, dev_c1);
        kernel<<<blocks, threads, 0, stream2>>>(dev_a2, dev_b2, dev_c2);

        // Copy the result chunks on the host.
        HANDLE_ERROR(cudaMemcpyAsync(c + chunk, dev_c1, CHUNK_SIZE * sizeof(int),
                    cudaMemcpyDeviceToHost, stream1));
        HANDLE_ERROR(cudaMemcpyAsync(c + chunk + CHUNK_SIZE, dev_c2, CHUNK_SIZE * sizeof(int),
                    cudaMemcpyDeviceToHost, stream2));
    }
    HANDLE_ERROR(cudaStreamSynchronize(stream1));
    HANDLE_ERROR(cudaStreamSynchronize(stream2));

    // Stop timer and get elapsed time.
    HANDLE_ERROR(cudaEventRecord(stop, stream2)); 
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsed;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop));
    printf("Time taken: %3.1f ms\n", elapsed);

    // Free all allocated memory.
    HANDLE_ERROR(cudaStreamDestroy(stream1));
    HANDLE_ERROR(cudaStreamDestroy(stream2));
    HANDLE_ERROR(cudaFree(dev_a1));
    HANDLE_ERROR(cudaFree(dev_b1));
    HANDLE_ERROR(cudaFree(dev_c1));
    HANDLE_ERROR(cudaFree(dev_a2));
    HANDLE_ERROR(cudaFree(dev_b2));
    HANDLE_ERROR(cudaFree(dev_c2));
    HANDLE_ERROR(cudaFreeHost(a));
    HANDLE_ERROR(cudaFreeHost(b));
    HANDLE_ERROR(cudaFreeHost(c));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    return EXIT_SUCCESS;
}
