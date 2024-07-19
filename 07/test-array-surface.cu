// This is a test file to work with CUDA Array and Surface Object.
#include <stdio.h>
#include <stdlib.h>

#include "cuda-error.h"

const size_t DIM = 32;

__global__ void init_input_kernel(cudaSurfaceObject_t input) {
    // Calculate input coordinates.
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Absolute offset considering input as a linar 1D array.
    float offset = x + y * blockDim.x * gridDim.x;

    surf2Dwrite(offset, input, x * sizeof(float), y);
}

int main() {
    // Allocate the input 2D CUDA Array on the device. It will be use as texture
    // input for the kernel.
    cudaChannelFormatDesc fmt_desc;
    memset(&fmt_desc, 0, sizeof(cudaChannelFormatDesc));
    fmt_desc.f = cudaChannelFormatKindFloat;
    fmt_desc.x = 32;
    cudaArray_t dev_input;
    HANDLE_ERROR(cudaMallocArray(&dev_input, &fmt_desc, DIM, DIM, cudaArraySurfaceLoadStore));

    // Create resorce description.
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = dev_input;

    // Create the surface object for dev_input 2D CUDA Array.
    cudaSurfaceObject_t surf_input;
    cudaCreateSurfaceObject(&surf_input, &resDesc);

    // This buffer will contain the final result. It is a 2D array but it is
    // linear on the host memory.
    float *output = (float *)malloc(DIM * DIM * sizeof(float));
    HANDLE_NULL(output);

    dim3 blocks(DIM / 8, DIM / 8);
    dim3 threads(8, 8);
    init_input_kernel<<<blocks, threads>>>(surf_input);

    HANDLE_ERROR(cudaMemcpy2DFromArray(output,
                DIM * sizeof(float),
                dev_input,
                0, 0,
                DIM * sizeof(float),
                DIM,
                cudaMemcpyDeviceToHost));

    // Print output.
    for (int row = 0; row < DIM; row++) {
        for (int column = 0; column < DIM; column++) {
            printf("%5.0f ", output[column + row * DIM]);
        }
        puts("");
    }

    HANDLE_ERROR(cudaDestroySurfaceObject(surf_input));
    HANDLE_ERROR(cudaFreeArray(dev_input));
    free(output);

    return EXIT_SUCCESS;
}
