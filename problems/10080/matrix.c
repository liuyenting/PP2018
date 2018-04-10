#define _GNU_SOURCE

#include <assert.h>
#include <pthread.h>
#include "matrix.h"

#define MAX_THREAD 6

#define MIN(x, y) (x < y ? x : y)

typedef struct {
    int lo, hi;
    int N;
    unsigned long (*A)[2048];
    unsigned long (*B)[2048];
    unsigned long (*C)[2048];
} thread_args_t;

void * row_worker(void *ptr) {
    thread_args_t *args = (thread_args_t *)ptr;

    int N = args->N;
    for (int i = args->lo, t = 0; i < args->hi; i++, t++) {
        for (int j = 0; j < N; j++) {
            unsigned long sum = 0;    // overflow, let it go.
            for (int k = 0; k < N; k++) {
                sum += (args->A)[i][k] * (args->B)[j][k];
            }
            (args->C)[i][j] = sum;
        }
    }
}

void multiply(int N, unsigned long A[][2048], unsigned long B[][2048], unsigned long C[][2048]) {
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        for (int i = 0; i < 6; i++) {
            CPU_SET(i, &cpuset);
        }
        assert(sched_setaffinity(0, sizeof(cpuset), &cpuset) == 0);
    }

    pthread_t threads[MAX_THREAD];
    thread_args_t args[MAX_THREAD];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    // transpose
    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++) {
            unsigned long tmp = B[j][i];
            B[j][i] = B[i][j];
            B[i][j] = tmp;
        }
    }

    int BLOCK = (N+MAX_THREAD-1)/MAX_THREAD, p = 0;
    for (int i = 0; i < N; i += BLOCK, p++) {
        args[p].lo = i;
        args[p].hi = MIN(N, i+BLOCK);
        args[p].N = N;
        args[p].A = A, args[p].B = B, args[p].C = C;
        pthread_create(&threads[p], &attr, row_worker, &args[p]);
    }
    for (int i = 0; i < p; i++) {
        pthread_join(threads[i], NULL);
    }
}
