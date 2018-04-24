#include "matrix.h"
#include <omp.h>

#define MIN(x, y) (x < y ? x : y)

void multiply(int N, unsigned long A[][2048], unsigned long B[][2048], unsigned long C[][2048]) {
    // transpose
    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++) {
            unsigned long tmp = B[j][i];
            B[j][i] = B[i][j];
            B[i][j] = tmp;
        }
    }

    int N_THREADS = omp_get_num_threads();
    int BLOCK = (N+N_THREADS-1)/N_THREADS;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int lo = tid*BLOCK;
        int hi = MIN(N, lo+BLOCK);
        for (int i = lo, t = 0; i < hi; i++, t++) {
            for (int j = 0; j < N; j++) {
                unsigned long sum = 0;    // overflow, let it go.
                for (int k = 0; k < N; k++) {
                    sum += A[i][k] * B[j][k];
                }
                C[i][j] = sum;
            }
        }
    }
}
