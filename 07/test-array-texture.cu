// This is a test file to larn CUDA Array and Texture Object.
#include <stdio.h>
#include <stdlib.h>

// HandleError, HANDLE_ERROR, HandleNull, HANDLE_NULL are utilies to handle
// errors on CUDA calls and malloc calls.

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))

static void HandleNull(void *ptr, const char *file, int line) {
        if (ptr == NULL) {
                printf("Host memory failed in %s at line %d\n", file, line);
                exit(EXIT_FAILURE);
        }
}

#define HANDLE_NULL(a) (HandleNull(a, __FILE__, __LINE__))

// Square dimension of the matrix.
const size_t DIM = 32;

__global__ void set_value_kernel(float *output, cudaTextureObject_t input) {
    // Map from threadIdx/BlockIdx to cell position.
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float value = tex2D<float>(input, x, y);

    output[offset] = value;
}

void init_input_array(cudaArray_t dev_input) {
    // Input is a 2D array but is linear on host memory.
    float *input = (float *)malloc(DIM * DIM * sizeof(float));
    HANDLE_NULL(input);

    // Set host array values.
    for (int i = 0; i < DIM * DIM; i++)
        input[i] = i;

    // Copy host array to device array.
    HANDLE_ERROR(cudaMemcpy2DToArray(dev_input,
                0, 0, // Offsets (not used).
                input,
                DIM * sizeof(float), // Width in bytes plus padding (no padding).
                DIM * sizeof(float), // Width in bytes.
                DIM, // Number of rows (not bytes).
                cudaMemcpyHostToDevice));

    free(input);
}

int main() {
    // Allocate a CUDA 2D array on the device. This array is used to create a
    // texture object that is used as input in the kernel.
    cudaChannelFormatDesc fmt_desc;
    memset(&fmt_desc, 0, sizeof(cudaChannelFormatDesc));
    fmt_desc.f = cudaChannelFormatKindFloat;
    fmt_desc.x = 32;
    cudaArray_t dev_input;
    HANDLE_ERROR(cudaMallocArray(&dev_input, &fmt_desc, DIM, DIM, cudaArraySurfaceLoadStore));

    init_input_array(dev_input); // Initialise the array with default values.

    // Create the resource description of the texture.
    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(cudaResourceDesc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = dev_input;

    // Create the texture description (with default options).
    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(cudaTextureDesc));

    // Finally create the texture object.
    cudaTextureObject_t input_tex;
    HANDLE_ERROR(cudaCreateTextureObject(&input_tex, &res_desc, &tex_desc, NULL));

    // Allocate a linear output buffer in the device. It is used as 2D array.
    float *dev_output;
    HANDLE_ERROR(cudaMalloc((void **)&dev_output, DIM * DIM * sizeof(float)));

    // Launch the kernel with 4x4 blocks of 8x8 threads for each block. The
    // total is 1024, the dimension of the input (32x32).
    dim3 blocks(DIM / 8, DIM / 8);
    dim3 threads(8, 8);
    set_value_kernel<<<blocks, threads>>>(dev_output, input_tex);

    // Allocate a linear output buffer in the host. It is used as 2D array.
    float *output = (float *)calloc(DIM * DIM, sizeof(float));
    HANDLE_NULL(output);

    // Copy output buffer from device to host.
    HANDLE_ERROR(cudaMemcpy(output, dev_output, DIM * DIM * sizeof(float),
                cudaMemcpyDeviceToHost));

    // Print output.
    for (int row = 0; row < DIM; row++) {
        for (int column = 0; column < DIM; column++) {
            printf("%5.0f ", output[column + row * DIM]);
        }
        puts("");
    }

    // Free allocated memory.
    HANDLE_ERROR(cudaFree(dev_output));
    HANDLE_ERROR(cudaDestroyTextureObject(input_tex));
    HANDLE_ERROR(cudaFreeArray(dev_input));
    free(output);

    return EXIT_SUCCESS;
}
