// heat-no-texture.cu simulates a heat map diffusion and saves it as a GIF file.
// This version doesn't use texture memory.
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "gif.h"
#include "cuda-error.h"

// Image and cell matrix square dimension.
const int DIM = 1024;

// Max temperture in a cell.
const float MAX_TEMP = 1.0f;

// Min temperature in a cell.
const float MIN_TEMP = 0.0001f;

// Speed of heat propagation among cell neighbours.
const float SPEED = 0.25f;

// Time steps to simulate for each GIF frame.
const int TIME_STEPS = 90;

// Total number of GIF frame to generate.
const int GIF_FRAMES = 10;

// Gif filename.
const char *GIF_FILENAME = "heat-no-texture.gif";

// Returns the colour for the given two floats and an a hue. It is called by
// float_to_color device function.
__device__ unsigned char value(float n1, float n2, int hue) {
    // Sanitize hue value.
    if (hue > 360)
        hue -= 360;
    else if (hue < 0)
        hue += 360;

    unsigned char retval;
    if (hue < 60)
        retval = (unsigned char)(255 * (n1 + (n2-n1)*hue/60));
    if (hue < 180)
        retval = (unsigned char)(255 * n2);
    if (hue < 240)
        retval = (unsigned char)(255 * (n1 + (n2-n1)*(240-hue)/60));
    else
        retval = (unsigned char)(255 * n1);

    return retval;
}

// Converts the given heat map (data) into an image (image).
__global__ void float_to_color(unsigned char *image, const float *data) {
    // Map from threadIdx/BlockIdx to pixel and cell position.
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float l = data[offset];
    float s = 1;
    int h = (180 + (int)(360.0f * data[offset])) % 360;

    float m1, m2;
    if (l <= 0.5f)
        m2 = l * (1 + s);
    else
        m2 = l + s - l * s;
    m1 = 2 * l - m2;

    image[offset * 4 + 0] = value(m1, m2, h + 120);
    image[offset * 4 + 1] = value(m1, m2, h);
    image[offset * 4 + 2] = value(m1, m2, h - 120);
    image[offset * 4 + 3] = 255;
}

// Updates the specified input data with the specified constant data. Only cells
// with a value greater than zero in the constant data overwrite the input data,
// others are left intact.
__global__ void copy_const_kernel(float *input, const float *const_input) {
    // Map from threadIdx/BlockIdx to cell position.
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    // The const_input is the heatmap with the initial values. Just copy the
    // values greater than zero, leaving the other values intact.
    if (const_input[offset] != 0)
        input[offset] = const_input[offset];
}

// Simulates a single step of heat diffusion from the input, and writes the
// result to the output.
__global__ void blend_kernel(float *output, const float *input) {
    // Map from threadIdx/BlockIdx to cell position.
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    // We must consider the case of corner cell with no left, right, top or
    // bottom neighbours.
    int neighbours = 0;
    float neighbours_total = 0;
    if (x > 0) { // Left neighbour.
        neighbours_total += input[offset - 1];
        neighbours++;
    }
    if (x < DIM-1) { // Right neighbour.
        neighbours_total += input[offset + 1];
        neighbours++;
    }
    if (y > 0) { // Top neighbour.
        neighbours_total += input[offset - DIM]; // Cel data is a linear buf.
        neighbours++;
    }
    if (y < DIM-1) { // Bottom neighbour.
        neighbours_total += input[offset + DIM];
        neighbours++;
    }

    output[offset] = input[offset] + SPEED * (neighbours_total - neighbours * input[offset]);
}

// Initialises the given buffer with a heat map of constant data.
void init_const_data(float *buffer) {
    // Cycle over all cells.
    for (int i = 0; i < DIM*DIM; i++) {
        buffer[i] = 0;

        int x = i % DIM;
        int y = i / DIM;

        // Set a square sector with the max temp.
        if ((x > 300) && (x < 600) && (y > 310) && (y < 601))
            buffer[i] = MAX_TEMP;
    }

    buffer[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP)/2;
    buffer[DIM * 700 + 100] = MIN_TEMP;
    buffer[DIM * 300 + 300] = MIN_TEMP;
    buffer[DIM * 200 + 700] = MIN_TEMP;
    for (int y = 800; y < 900; y++) {
        for (int x = 400; x < 500; x++) {
            buffer[x + y * DIM] = MIN_TEMP;
        }
    }
}

// Initialises the given buffer with a heat map of initial data.
void init_inital_data(float *buffer) {
    for (int y=800; y<DIM; y++) {
        for (int x=0; x<200; x++) {
            buffer[x+y*DIM] = MAX_TEMP;
        }
    }
}

// FrameData contains the pointers needed by the generate_frame function to
// handle all the allocated buffers in the device.
struct FrameData {
    unsigned char *dev_image;

    float *dev_input_data;

    float *dev_output_data;

    float *dev_const_data;
};

// Produces a single frame using the buffers contained in data.
void generate_frame(FrameData *data) {
    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);

    for (int step = 0; step < TIME_STEPS; step++) {
        copy_const_kernel<<<blocks, threads>>>(data->dev_input_data, data->dev_const_data);

        blend_kernel<<<blocks, threads>>>(data->dev_output_data, data->dev_input_data);

        // I need to swap the input and output buffers because the output will
        // be the input in the next cycle.
        float *dev_tmp = data->dev_input_data;
        data->dev_input_data = data->dev_output_data;
        data->dev_output_data = dev_tmp;
    }

    // Note that we use dev_input_data because it was swapped in the cycle.
    float_to_color<<<blocks, threads>>>(data->dev_image, data->dev_input_data);
}

int main() {
    int retval = 0;
    GifWriter writer;
    GifBegin(&writer, GIF_FILENAME, DIM, DIM, 10, 8, true);
    printf("Writing GIF image file to \"%s\"\n", GIF_FILENAME);

    // Allocate image data on the host.
    unsigned char *image = (unsigned char *)malloc(DIM * DIM * 4);
    if (image == NULL) {
        puts("Failed to allocate memory for host image");
        return EXIT_FAILURE;
    }

    // Allocate image data on the device.
    unsigned char *dev_image;
    HANDLE_ERROR(cudaMalloc((void**)&dev_image, DIM*DIM*4));

    // Allocate memory for input, output and const data of the heat map. This
    // assumes that a float is 4 unsigned char (RGBA).
    assert(sizeof(float) == 4*sizeof(unsigned char));
    float *dev_input_data, *dev_output_data, *dev_const_data;
    HANDLE_ERROR(cudaMalloc((void**)&dev_input_data, DIM*DIM*4));
    HANDLE_ERROR(cudaMalloc((void**)&dev_output_data, DIM*DIM*4));
    HANDLE_ERROR(cudaMalloc((void**)&dev_const_data, DIM*DIM*4));

    // Allocate a temporary buffer to initialize the data.
    float *tmp = (float *)malloc(DIM * DIM * 4);
    if (tmp == NULL) {
        puts("Failed to allocate memory for tmp data");
        retval = EXIT_FAILURE;
        goto out_tmp;
    }

    // Init constant data and copy to device.
    init_const_data(tmp);
    HANDLE_ERROR(cudaMemcpy(dev_const_data, tmp, DIM*DIM*4,
                cudaMemcpyHostToDevice));

    // Init initial data and copy to device. It is based on constant data.
    init_inital_data(tmp);
    HANDLE_ERROR(cudaMemcpy(dev_input_data, tmp, DIM*DIM*4,
                cudaMemcpyHostToDevice));

    // We do not need anymore this buffer.
    free(tmp);

    FrameData data;
    data.dev_image = dev_image;
    data.dev_input_data = dev_input_data;
    data.dev_output_data = dev_output_data;
    data.dev_const_data = dev_const_data;

    float elapsed, total_time, frames;
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    // Generate frame by frame.
    for (int frame = 0; frame < GIF_FRAMES; frame++) {
        HANDLE_ERROR(cudaEventRecord(start, 0));

        generate_frame(&data);

        HANDLE_ERROR(cudaMemcpy(image, dev_image, DIM*DIM*4,
                    cudaMemcpyDeviceToHost));

        HANDLE_ERROR(cudaEventRecord(stop, 0));
        HANDLE_ERROR(cudaEventSynchronize(stop));

        HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop));
        total_time += elapsed;
        frames++;

        printf("Writing frame %3d (avg. time %3.2f ms)...\n", frame, total_time / frames);
        GifWriteFrame(&writer, image, DIM, DIM, 10, 8, true);
    }

    printf("Finalizing image...\n");
    GifEnd(&writer);

    // Free al allocated memory on both host and device.
out_tmp:
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    HANDLE_ERROR(cudaFree(dev_input_data));
    HANDLE_ERROR(cudaFree(dev_output_data));
    HANDLE_ERROR(cudaFree(dev_const_data));
    HANDLE_ERROR(cudaFree(dev_image));
    free(image);

    return retval;
}
