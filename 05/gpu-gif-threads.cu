#include <math.h>

#include "gif.h"
#include "cuda-error.h"

#define DIM 1024
#define TICKS 50

__global__ void kernel(unsigned char *image, int ticks) {
    // Map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    // Now calculate the value at that position
    float fx = x - DIM/2;
    float fy = y - DIM/2;
    float d = sqrtf( fx * fx + fy * fy );
    unsigned char grey = (unsigned char)(128.0f + 127.0f *
            cos(d/10.0f - ticks/7.0f) /
            (d/10.0f + 1.0f));

    // Write value to the image.
    image[offset * 4 + 0] = grey;
    image[offset * 4 + 1] = grey;
    image[offset * 4 + 2] = grey;
    image[offset * 4 + 3] = 255;
}

int main() {
    const char *filename = "gpu-gif.gif";

    GifWriter writer;
    GifBegin(&writer, filename, DIM, DIM, 2, 8, true);

    unsigned char image[DIM * DIM * 4];
    unsigned char *dev_image;

    HANDLE_ERROR(cudaMalloc((void**)&dev_image, DIM*DIM*4));

    for (int tick = 0; tick < TICKS; tick++) {
        dim3 blocks(DIM/16, DIM/16);
        dim3 threads(16, 16);

        kernel<<<blocks, threads>>>(dev_image, tick);

        HANDLE_ERROR(cudaMemcpy(image, dev_image, DIM*DIM*4,
                    cudaMemcpyDeviceToHost));

        printf("Writing frame %d...\n", tick);
        GifWriteFrame(&writer, image, DIM, DIM, 2, 8, true);
    }

    HANDLE_ERROR(cudaFree(dev_image));

    printf("Finalizing image...\n");
    GifEnd(&writer);

    return 0;
}
