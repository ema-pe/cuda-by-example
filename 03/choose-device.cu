#include <stdio.h>
#include <string.h>

#include "../common/cuda-error.h"

int main(void) {
    cudaDeviceProp prop;
    int device;

    HANDLE_ERROR(cudaGetDevice(&device));
    printf("ID of current CUDA device: %d\n", device);

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 7;
    prop.minor = 0;

    HANDLE_ERROR(cudaChooseDevice(&device, &prop));

    printf("ID of the closest CUDA device: %d\n", device);
    HANDLE_ERROR(cudaSetDevice(device));

    return 0;
}
