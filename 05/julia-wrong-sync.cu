#include <stdio.h>
#include <math.h>

#include "lodepng.h"
#include "cuda-error.h"

// The square dimension must be a multiple of two.
const int DIM = 1024;

const float PI = 3.1415926535897932f;

__global__ void kernel(unsigned char *image) {
    // map from threadIdx/blockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    __shared__ float shared[16][16];

    // now calculate the value at that position
    const float period = 128.0f;

    shared[threadIdx.x][threadIdx.y] =
        255 * (sinf(x*2.0f*PI/ period) + 1.0f) *
        (sinf(y*2.0f*PI/ period) + 1.0f) / 4.0f;

    // Because __syncthreads() is missing, the output won't be perfect. This
    // thread is using the shared[][] value written by another thread, but the
    // latter may not have yet finished writing that value, so there will be a
    // dirty read.
    image[offset * 4 + 0] = 0;
    image[offset * 4 + 1] = shared[15-threadIdx.x][15-threadIdx.y];
    image[offset * 4 + 2] = 0;
    image[offset * 4 + 3] = 255;
}

int main(void) {
    const char *filename = "julia-wrong-sync.png";

    // Alloc image data.
    unsigned char *image = (unsigned char *)malloc(DIM * DIM * 4);
    if (image == NULL) {
        puts("Failed to alloc memory for image\n");
        return EXIT_FAILURE;
    }

    // Alloc device image data.
    unsigned char *dev_image;
    HANDLE_ERROR(cudaMalloc((void**)&dev_image, DIM * DIM * 4));

    dim3 grid(DIM/16, DIM/16);
    dim3 threads(16, 16);
    kernel<<<grid, threads>>>(dev_image);

    HANDLE_ERROR(cudaMemcpy(image, dev_image, DIM * DIM * 4,
                cudaMemcpyDeviceToHost));

    // Save image to disk.
    unsigned error = lodepng_encode32_file(filename, image, DIM, DIM);
    if (error) {
        printf("error on saving image %u: %s\n", error, lodepng_error_text(error));
        return EXIT_FAILURE;
    }

    free(image);

    HANDLE_ERROR(cudaFree(dev_image));

    return 0;
}
