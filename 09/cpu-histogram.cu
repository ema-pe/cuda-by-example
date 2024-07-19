// Histogram calculation of a randomly generated data.
//
// This is the CPU version.
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "cuda-error.h"

const size_t HISTO_SIZE = 256;

const size_t DATA_SIZE = 500 * 1024 * 1024; // 500 MB.

int main() {
    unsigned int histogram[HISTO_SIZE];
    clock_t tic = clock();

    unsigned char *block = random_block(DATA_SIZE);

    // Set initial values for histogram.
    for (int i = 0; i < HISTO_SIZE; i++)
        histogram[i] = 0;

    // Update the histogram.
    for (size_t i = 0; i < DATA_SIZE; i++)
        histogram[block[i]]++;

    clock_t toc = clock();

    printf("Elapsed: %3.3f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

    // Print and check the histogram.
    size_t total_values = 0;
    for (int i = 0; i < HISTO_SIZE; i++)
        total_values += histogram[i];
    if (total_values != DATA_SIZE)
        printf("Different histogram values with data (%ld, %ld)\n", total_values, DATA_SIZE);

    free(block);

    return EXIT_SUCCESS;
}
