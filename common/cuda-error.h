/* Utility routines to handle errors with CUDA.
 *
 * Taken from the code provided with Cuda By Example book.
 */
#ifndef CUDA_ERROR_H
#define CUDA_ERROR_H

#include <stdio.h>
#include <stdlib.h>

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

unsigned char *random_block(const size_t size) {
        unsigned char *block = (unsigned char *)malloc(size);
        HANDLE_NULL(block);

        for (int i = 0; i < size; i++)
                block[i] = rand();

        return block;
}

#endif // CUDA_ERROR_H
