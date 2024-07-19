#include <stdio.h>

#define N 10

void add(int *a, int *b, int *c) {
        int tid = 0; // CPU ID.

        while (tid < N) {
                c[tid] = a[tid] + b[tid];
                tid += 1; // Only one CPU.
        }
}

int main(void) {
        int a[N], b[N], c[N];

        // Fill the arrays.
        for (int i = 0; i < N; i++) {
                a[i] = -i;
                b[i] = i * i;
        }

        add(a, b, c);

        // Display results.
        for (int i = 0; i < N; i++) {
                printf("%2d + %2d = %2d\n", a[i], b[i], c[i]);
        }

        return 0;
}
