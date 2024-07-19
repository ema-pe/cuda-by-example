/* CPU version to compute Julia mandelbrot and saves the image as
 * "cpu-julia.png" on disk.
 */
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

#include "lodepng.h"

#define DIM 2000

float magnitude2(float complex a) {
    return crealf(a) * crealf(a) + cimagf(a) * cimagf(a);
}

// Returns 1 if the given point is in the Julia set, 0 otherwise.
int julia(int x, int y) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    float complex c = -.8 + .156 * I; // Arbitrary constant.
    float complex a = jx + jy * I;

    for (int i = 0; i < 200; i++) {
        a = a * a + c;
        if (magnitude2(a) > 1000) {
            return 0;
        }
    }

    return 1;
}

// Calculate each pixel of the image.
void kernel(unsigned char *image) {
    for (int y = 0; y < DIM; y++) { // Height.
        for (int x = 0; x < DIM; x++) { // Width.
            int offset = x + y * DIM;

            int julia_value = julia(x, y);
            image[offset * 4 + 0] = 255 * julia_value;
            image[offset * 4 + 1] = 0;
            image[offset * 4 + 2] = 0;
            image[offset * 4 + 3] = 255;
        }
    }
}

int main(void) {
    const char *filename = "cpu-julia.png";

    // Alloc image data.
    unsigned char *image = (unsigned char *)malloc(DIM * DIM * 4);
    if (image == NULL) {
        puts("Failed to alloc memory for image\n");
        return EXIT_FAILURE;
    }

    // Create image (pixels).
    kernel(image);

    // Save image to disk.
    unsigned error = lodepng_encode32_file(filename, image, DIM, DIM);
    if (error) {
        printf("error on saving image %u: %s\n", error, lodepng_error_text(error));
        return EXIT_FAILURE;
    }

    free(image);

    return 0;
}
