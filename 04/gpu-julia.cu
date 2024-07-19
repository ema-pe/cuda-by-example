/* GPU version to compute Julia mandelbrot and saves the image as
 * "cpu-julia.png" on disk.
 */
#include <stdio.h>
#include <stdlib.h>

#include "lodepng.h"
#include "cuda-error.h"

#define DIM 2000

struct cuComplex {
    float r, i;

    __device__ cuComplex(float a, float b) : r(a), i(b) {}

    __device__ float magnitude2(void) {
        return r * r + i * i;
    }

    __device__ cuComplex operator*(const cuComplex &a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }

    __device__ cuComplex operator+(const cuComplex &a) {
        return cuComplex(r+a.r, i+a.i);
    }
};

// Returns 1 if the given point is in the Julia set, 0 otherwise.
__device__ int julia(int x, int y) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    cuComplex c(-.8, .156); // Arbitrary constant.
    cuComplex a(jx, jy);

    for (int i = 0; i < 200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000) {
            return 0;
        }
    }

    return 1;
}

// Calculate each pixel of the image.
__global__ void kernel(unsigned char *image) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    int julia_value = julia(x, y);
    image[offset * 4 + 0] = 255 * julia_value;
    image[offset * 4 + 1] = 0;
    image[offset * 4 + 2] = 0;
    image[offset * 4 + 3] = 255;
}

int main(void) {
    const char *filename = "gpu-julia.png";

    // Alloc image data.
    unsigned char *image = (unsigned char *)malloc(DIM * DIM * 4);
    if (image == NULL) {
        puts("Failed to alloc memory for image\n");
        return EXIT_FAILURE;
    }

    // Alloc device image data.
    unsigned char *dev_image;
    HANDLE_ERROR(cudaMalloc((void**)&dev_image, DIM * DIM * 4));

    dim3 grid(DIM, DIM);
    kernel<<<grid, 1>>>(dev_image);

    HANDLE_ERROR(cudaMemcpy(image, dev_image, DIM * DIM * 4,
                cudaMemcpyDeviceToHost));

    // Save image to disk.
    unsigned error = lodepng_encode32_file(filename, image, DIM, DIM);
    if (error) {
        printf("error on saving image %u: %s\n", error, lodepng_error_text(error));
        return EXIT_FAILURE;
    }

    free(image);

    cudaFree(dev_image);

    return 0;
}
