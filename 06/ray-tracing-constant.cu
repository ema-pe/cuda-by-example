#include <stdlib.h>
#include <math.h>

#include "lodepng.h"
#include "cuda-error.h"

#define INF 2e10f

#define rnd(x) (x * rand() / RAND_MAX)

// Image square dimension.
const int DIM = 2048;

// Number of spheres in the space.
const int SPHERES = 20;

struct Sphere {
    // Colors in RGB mode.
    float r, g, b;

    // Posizion in the space and radius of the sphere.
    float x, y, z, radius;

    __device__ float hit(float ray_x, float ray_y, float *n) {
        float dx = ray_x - x;
        float dy = ray_y - y;

        if (dx*dx + dy*dy < radius*radius) {
            float dz = sqrtf( radius*radius - dx*dx - dy*dy );
            *n = dz / sqrtf( radius * radius );
            return dz + z;
        }
        return -INF;
    }
};

__constant__ Sphere spheres[SPHERES];

__global__ void kernel(unsigned char *image) {
    // Map from threadIdx and blockIdx to pixel position.
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x; // Pos. in the linear buffer.

    float ray_x = (x - DIM / 2);
    float ray_y = (y - DIM / 2);

    // Iterate over all spheres to search the hit and to color the pixel.
    float r = 0, g = 0, b = 0;
    float maxz = -INF;
    for (int i = 0; i < SPHERES; i++) {
        float n;
        float t = spheres[i].hit(ray_x, ray_y, &n);

        if (t > maxz) {
            float fscale = n;
            r = spheres[i].r * fscale;
            g = spheres[i].g * fscale;
            b = spheres[i].b * fscale;
        }
    }

    // Set the pixel color. If there is no hit, the color will be black.
    image[offset * 4 + 0] = (int)(r * 255);
    image[offset * 4 + 1] = (int)(g * 255);
    image[offset * 4 + 2] = (int)(b * 255);
    image[offset * 4 + 3] = 255;
}

int main() {
    const char *filename = "ray-tracing-constant.png";

    cudaEvent_t start, stop; // Timers.
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    HANDLE_ERROR(cudaEventRecord(start, 0));

    // Alloc image data.
    unsigned char *image = (unsigned char *)malloc(DIM * DIM * 4);
    if (image == NULL) {
        puts("Failed to alloc memory for image\n");
        return EXIT_FAILURE;
    }

    // Alloc device image data.
    unsigned char *dev_image;
    HANDLE_ERROR(cudaMalloc((void **)&dev_image, DIM * DIM * 4));

    // Generate spheres on host side with a temporary memory.
    Sphere *tmp_spheres = (Sphere *)malloc(sizeof(Sphere) * SPHERES);
    for (int i = 0; i < SPHERES; i++) {
        tmp_spheres[i].r = rnd(1.0f);
        tmp_spheres[i].g = rnd(1.0f);
        tmp_spheres[i].b = rnd(1.0f);

        // Align the spheres in the center.
        tmp_spheres[i].x = rnd(2000.0f) - 1000;
        tmp_spheres[i].y = rnd(2000.0f) - 1000;
        tmp_spheres[i].z = rnd(2000.0f) - 1000;

        // Add 40 to have a minimium radius.
        tmp_spheres[i].radius = rnd(200.0f) + 40;
    }

    // Copy host spheres to device and free host buffer.
    HANDLE_ERROR(cudaMemcpyToSymbol(spheres, tmp_spheres,
                sizeof(Sphere) * SPHERES));
    free(tmp_spheres);

    // Run kernel and generate the image.
    dim3 grids(DIM/16,DIM/16);
    dim3 threads(16,16);
    kernel<<<grids,threads>>>(dev_image);

    // Copy device image data to host.
    HANDLE_ERROR(cudaMemcpy(image, dev_image, DIM * DIM * 4,
                cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    // Must syncronize the event because the kernel launch is async.
    HANDLE_ERROR(cudaEventSynchronize(stop));

    // Get elapsed time and print.
    float elapsed;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop));
    printf("Time to generate: %3.3f ms\n", elapsed);

    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
 
    // Save image to disk.
    unsigned error = lodepng_encode32_file(filename, image, DIM, DIM);
    if (error) {
        printf("error on saving image %u: %s\n", error, lodepng_error_text(error));
        return EXIT_FAILURE;
    }

    // Free host and device image data.
    HANDLE_ERROR(cudaFree(dev_image));
    free(image);

    return 0;
}
