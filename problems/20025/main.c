#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define P 10

int main(void) {
    int N;
    scanf("%d", &N);

    int *arr = malloc(N*sizeof(int));
    int *buf = malloc(N*sizeof(int));
    for (int i = 0; i < N; i++) {
        scanf("%d", arr+i);
        buf[i] = 0;
    }

    int iter = (int)ceil(log2(N));
    for (int i = 0; i < iter; i++) {
        int offset = (int)pow(2, i);
        for (int j = 0; j < N; ) {
            for (int p = 0; p < P; p++, j++) {
                if (j < offset) {
                    buf[j] = arr[j];
                } else {
                    buf[j] = arr[j] + arr[j-offset];
                }
            }
        }

        int *tmp = buf;
        buf = arr;
        arr = tmp;
    }

    for (int j = 0; j < N; j++) {
        printf("%d%c", arr[j], (j == N-1) ? '\n' : ' ');
    }
}
