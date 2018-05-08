#include <omp.h>
#include <stdio.h>
#include <stdint.h>

#define MAX_N 16

char BLOCK[MAX_N*MAX_N];
int BOARD[MAX_N];

inline int abs(int x) {
    // mask of sign bit
    uint32_t y = x >> 31;
    // toggle the sign bit
    x ^= y;
    // add 1 if x is negative (2's complement)
    x += y & 1;
    return x;
}

int place(int N, int row, int col) {
    if (BLOCK[row*n + col] == '*') {
        // blockage
        return 0;
    }
    for (int i = 0; i < row; i++) {
        if (BOARD[i] == col) {
            // column conflict
            return 0;
        } else if (abs(BOARD[i] - col) == abs(i-row)) {
            // diagonal conflict
            return 0;
        }
    }
    return 1;
}

int queen(int row, int N, int count) {
    for (int c = 0; c < N; c++) {
        //printf("r=%d, c=%d\n", row, c);
        if (place(N, row, c)) {
            BOARD[row] = c;
            if (row == N-1) {
                //printf(".. found!\n");
                count++;
            } else {
                //printf(".. deeper\n");
                count = queen(row+1, N, count);
            }
        }
    }
    return count;
}

int main(void) {
    int N, n_case = 0;
    while (scanf("%d", &N) != EOF) {
        n_case++;
        printf("N=%d\n", N);

        for (int i = 0; i < N*N; i += N) {
            scanf("%s", &BLOCK[i]);
        }

        for (int i = 0; i < N*N; i++) {
            printf("%c ", BLOCK[i]);
            if ((i+1)%N == 0) {
                printf("\n");
            }
        }

        printf("Case %d: %d\n", n_case, queen(0, N, 0));
    }

    return 0;
}
