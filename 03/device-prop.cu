#include <stdio.h>

#include "../common/cuda-error.h"

int main(void) {
    cudaDeviceProp prop;

    int count;
    HANDLE_ERROR(cudaGetDeviceCount(&count));
    for (int i = 0; i < count; i++) {
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));

        printf("  --- General Information for device %d ---\n", i);
        printf("Name: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Clock rate: %d (KHz) \n", prop.clockRate);
        printf("Device copy overlap: ");
        switch (prop.asyncEngineCount) {
            case 0:
                printf("Not supported\n");
                break;
            case 1:
                printf("Supported (one direction)\n");
                break;
            case 2:
                printf("Supported (both directions)\n");
                break;
        }
        printf("Kernel execution timeout: %s\n",
                prop.kernelExecTimeoutEnabled ?  "Enabled" : "Disabled");
        printf("Integrated device: %s\n", prop.integrated ? "yes" : "no");
        printf("Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
        printf("ECC: %s\n", prop.ECCEnabled ? "Supported" : "Not supported");

        printf("  --- Memory Information for device %d ---\n", i);
        printf("Total global memory: %ld (bytes)\n", prop.totalGlobalMem);
        printf("Total constant memory: %ld (bytes)\n", prop.totalConstMem);
        printf("Max mem pitch: %ld (bytes)\n", prop.memPitch);
        printf("Texture alignment: %ld (bytes)\n", prop.textureAlignment);

        printf("  --- Multiprocessor Information for device %d ---\n", i);
        printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("Max shared memory per mp: %ld (bytes)\n",
                prop.sharedMemPerBlock);
        printf("Max registers per mp: %d\n", prop.regsPerBlock);
        printf("Threads in warp: %d\n", prop.warpSize);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0],
                prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0],
                prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\n");
    }

    return 0;
}
