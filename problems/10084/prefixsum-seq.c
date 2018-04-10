#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <pthread.h>
#include <assert.h>
#include "utils.h"

#define MAXN 10000005
#define MAX_THREAD 6
uint32_t prefix_sum[MAXN];

#define MIN(x, y) (x < y ? x : y)

typedef struct {
    int lo, hi;
    uint32_t *psum;
    union {
        uint32_t key;
        uint32_t bsum;
    };
} thread_args_t;

void * bsum_worker(void *ptr) {
    thread_args_t *args = (thread_args_t *)ptr;

    uint32_t *psum = args->psum;
    uint32_t sum = 0;
    for (int i = args->lo, j = 0; i <= args->hi; i++, j++) {
        sum += encrypt(i, args->key);
        psum[j] = sum;
    }
}

void * psum_worker(void *ptr) {
    thread_args_t *args = (thread_args_t *)ptr;

    uint32_t *psum = args->psum;
    for (int i = args->lo, j = 0; i <= args->hi; i++, j++) {
        psum[j] += args->bsum;
    }
}

int main() {
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

    int n;
    uint32_t key;
    while (scanf("%d %" PRIu32, &n, &key) == 2) {
        int BLOCK = (n+MAX_THREAD-1)/MAX_THREAD, p = 0;
        for (int i = 1; i <= n; i += BLOCK, p++) {
            args[p].lo = i;
            args[p].hi = MIN(n, i+BLOCK-1);
            args[p].psum = prefix_sum+i;
            args[p].key = key;
            pthread_create(&threads[p], &attr, bsum_worker, &args[p]);
        }
        for (int i = 0; i < p; i++) {
            pthread_join(threads[i], NULL);
        }

        p = 0;
        uint32_t bsum = 0;
        for (int i = 1; i <= n; i += BLOCK, p++) {
            args[p].lo = i;
            args[p].hi = MIN(n, i+BLOCK-1);
            args[p].psum = prefix_sum+i;
            args[p].bsum = bsum;
            pthread_create(&threads[p], &attr, psum_worker, &args[p]);
            bsum += prefix_sum[MIN(n, i+BLOCK-1)];
        }
        for (int i = 0; i < p; i++) {
            pthread_join(threads[i], NULL);
        }

        output(prefix_sum, n);
    }

    pthread_attr_destroy(&attr);

    return 0;
}
