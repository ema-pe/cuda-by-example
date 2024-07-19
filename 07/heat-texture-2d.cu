// heat-no-texture.cu simulates a heat map diffusion and saves it as a GIF file.
// This version uses texture memory (2D version).
//
// Since texture references were deprecated and removed in CUDA 12, I used the
// Texture and Surface Object API. A 2D texture requires a CUDA array, so I used
// it, but I can only access a CUDA array through a texture or a surface object.
// So I also used the Surface Object API.
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

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
const char *GIF_FILENAME = "heat-texture-2d.gif";

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
__global__ void float_to_color(unsigned char *image, cudaTextureObject_t data) {
    // Map from threadIdx/BlockIdx to pixel and cell position.
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float l = tex2D<float>(data, x, y);
    float s = 1;
    int h = (180 + (int)(360.0f * l)) % 360;

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

// Updates the given surface with the given texture. It only updates cells whose
// values in the texture are greater than zero.
__global__ void copy_const_kernel(cudaSurfaceObject_t out, cudaTextureObject_t in) {
    // Map from threadIdx/BlockIdx to cell position.
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // 'in' is the heatmap with the initial values. Just copy the values greater
    // than zero, leaving the other values intact.
    float value = tex2D<float>(in, x, y);
    if (value != 0)
        surf2Dwrite(value, out, x * sizeof(float), y);
}

// Simulates a single step of heat diffusion from the input texture, and writes
// the result to the output.
__global__ void blend_kernel(cudaSurfaceObject_t out, cudaSurfaceObject_t in) {
    // Map from threadIdx/BlockIdx to cell position.
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // We must consider the case of corner cell with no left, right, top or
    // bottom neighbours.
    int neighbours = 0;
    float neighbours_total = 0, value = 0;
    if (x > 0) { // Left neighbour.
        surf2Dread(&value, in, (x - 1) * sizeof(float), y);
        neighbours_total += value;
        neighbours++;
    }
    if (x < DIM-1) { // Right neighbour.
        surf2Dread(&value, in, (x + 1) * sizeof(float), y);
        neighbours_total += value;
        neighbours++;
    }
    if (y > 0) { // Top neighbour.
        surf2Dread(&value, in, x * sizeof(float), y - 1);
        neighbours_total += value;
        neighbours++;
    }
    if (y < DIM-1) { // Bottom neighbour.
        surf2Dread(&value, in, x * sizeof(float), y + 1);
        neighbours_total += value;
        neighbours++;
    }

    float input;
    surf2Dread(&input, in, x * sizeof(float), y);
    float output = input + SPEED * (neighbours_total - neighbours * input);

    surf2Dwrite(output, out, x * sizeof(float), y);
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

    cudaTextureObject_t tex_const;

    cudaSurfaceObject_t surf_in;

    cudaSurfaceObject_t surf_out;
};

// Produces a single frame using the buffers contained in data.
void generate_frame(FrameData *data) {
    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);

    // Because the input and output are swapped in each loop, we need to keep
    // track of which data is being used as input.
    bool use_input_data = true;
    for (int step = 0; step < TIME_STEPS; step++) {
        // The output (surf_in) will be used as input in the second kernel.
        cudaSurfaceObject_t surf_in = use_input_data ? data->surf_in : data->surf_out;
        copy_const_kernel<<<blocks, threads>>>(surf_in, data->tex_const);

        cudaSurfaceObject_t surf_out = use_input_data ? data->surf_out : data->surf_in;
        surf_in = use_input_data ? data->surf_in : data->surf_out;
        blend_kernel<<<blocks, threads>>>(surf_out, surf_in);

        use_input_data = !use_input_data;
    }

    cudaSurfaceObject_t surf_in = use_input_data ? data->surf_in : data->surf_out;
    float_to_color<<<blocks, threads>>>(data->dev_image, surf_in);
}

// Allocates the CUDA 2D array on 'data' and creates the surface associated with
// the array on 'surface'.
void build_surface(cudaArray_t *data, cudaSurfaceObject_t *surface) {
    // Allocate the input 2D CUDA Array on the device. The kernel uses the array
    // via surface.
    cudaChannelFormatDesc fmt_desc;
    memset(&fmt_desc, 0, sizeof(cudaChannelFormatDesc));
    fmt_desc.f = cudaChannelFormatKindFloat;
    fmt_desc.x = 32;
    HANDLE_ERROR(cudaMallocArray(data, &fmt_desc, DIM, DIM, cudaArraySurfaceLoadStore));

    // Create resorce description.
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = *data;

    // Create the surface object for the 2D CUDA Array.
    HANDLE_ERROR(cudaCreateSurfaceObject(surface, &resDesc));
}

// Allocates a 2D CUDA array to 'data' and creates a texture associated with the
// array to 'tex'.
void build_texture(cudaArray_t *data, cudaTextureObject_t *tex) {
    // Allocate the input 2D CUDA Array on the device. The kernel uses the array
    // via texture.
    cudaChannelFormatDesc fmt_desc;
    memset(&fmt_desc, 0, sizeof(cudaChannelFormatDesc));
    fmt_desc.f = cudaChannelFormatKindFloat;
    fmt_desc.x = 32;
    HANDLE_ERROR(cudaMallocArray(data, &fmt_desc, DIM, DIM));

    // Create resource description.
    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(cudaResourceDesc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = *data;

    // Create texture description.
    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(cudaTextureDesc));

    // Create texture object.
    HANDLE_ERROR(cudaCreateTextureObject(tex, &res_desc, &tex_desc, NULL));
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

    // Allocate and create texture and surface objects for input, output and
    // const data of the heat map. This assumes that a float is 4 unsigned char
    // (RGBA).
    assert(sizeof(float) == 4*sizeof(unsigned char));
    cudaArray_t dev_input_data, dev_output_data, dev_const_data;
    cudaTextureObject_t tex_const;
    cudaSurfaceObject_t surf_in, surf_out;
    build_texture((cudaArray_t *)&dev_const_data, (cudaTextureObject_t *)&tex_const);
    build_surface((cudaArray_t *)&dev_input_data, (cudaSurfaceObject_t *)&surf_in);
    build_surface((cudaArray_t *)&dev_output_data, (cudaSurfaceObject_t *)&surf_out);

    // Allocate a temporary buffer to initialize the data.
    float *tmp = (float *)malloc(DIM * DIM * 4);
    if (tmp == NULL) {
        puts("Failed to allocate memory for tmp data");
        retval = EXIT_FAILURE;
        goto out_tmp;
    }

    // Init constant data and copy to device.
    init_const_data(tmp);
    HANDLE_ERROR(cudaMemcpy2DToArray(dev_const_data,
        0, 0, // Offsets (not used).
        tmp,
        DIM * sizeof(float), // Width in bytes plus padding (no padding).
        DIM * sizeof(float), // Width in bytes.
        DIM, // Number of rows (not bytes).
        cudaMemcpyHostToDevice));

    // Init initial data and copy to device. It is based on constant data.
    init_inital_data(tmp);
    HANDLE_ERROR(cudaMemcpy2DToArray(dev_input_data,
        0, 0, // Offsets (not used).
        tmp,
        DIM * sizeof(float), // Width in bytes plus padding (no padding).
        DIM * sizeof(float), // Width in bytes.
        DIM, // Number of rows (not bytes).
        cudaMemcpyHostToDevice));

    // We do not need anymore this buffer.
    free(tmp);

    FrameData data;
    data.dev_image = dev_image;
    data.tex_const = tex_const;
    data.surf_in = surf_in;
    data.surf_out = surf_out;

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
    HANDLE_ERROR(cudaDestroyTextureObject(tex_const));
    HANDLE_ERROR(cudaDestroySurfaceObject(surf_in));
    HANDLE_ERROR(cudaDestroySurfaceObject(surf_out));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    HANDLE_ERROR(cudaFreeArray(dev_input_data));
    HANDLE_ERROR(cudaFreeArray(dev_output_data));
    HANDLE_ERROR(cudaFreeArray(dev_const_data));
    HANDLE_ERROR(cudaFree(dev_image));
    free(image);

    return retval;
}
