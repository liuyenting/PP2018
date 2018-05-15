#include <omp.h>
#include <stdio.h>
#include <stdint.h>

#define MAX_N 16

char BLOCK[MAX_N*MAX_N];

inline int abs(int x) {
    // mask of sign bit
    uint32_t y = x >> 31;
    // toggle the sign bit
    x ^= y;
    // add 1 if x is negative (2's complement)
    x += y & 1;
    return x;
}

int place(int board[], int N, int row, int col) {
    if (BLOCK[row*N + col] == '*') {
        // blockage
        return 0;
    }
    for (int i = 0; i < row; i++) {
        if (board[i] == col) {
            // column conflict
            return 0;
        } else if (abs(board[i] - col) == abs(i-row)) {
            // diagonal conflict
            return 0;
        }
    }
    return 1;
}

int _queen(int board[], int r, int N, int count) {
    for (int c = 0; c < N; c++) {
        //printf("r=%d, c=%d\n", row, c);
        if (place(board, N, r, c)) {
            board[r] = c;
            if (r == N-1) {
                //printf(".. found!\n");
                count++;
            } else {
                //printf(".. deeper\n");
                count = _queen(board, r+1, N, count);
            }
        }
    }
    return count;
}

int queen(int N) {
    int count = 0;
    int board[MAX_N] = {0};

    #pragma omp parallel for collapse(3) \
                             firstprivate(board) \
                             reduction(+ : count) \
                             schedule(dynamic)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                if (!place(board, N, 0, i)) {
                    continue;
                }
                board[0] = i;

                if (!place(board, N, 1, j)) {
                    continue;
                }
                board[1] = j;

                if (!place(board, N, 2, k)) {
                    continue;
                }
                board[2] = k;

                count += _queen(board, 3, N, 0);
            }
        }
    }

    return count;

    //return _queen(0, N, 0);
}

int main(void) {
    int N, n_case = 0;
    while (scanf("%d", &N) != EOF) {
        n_case++;

        for (int i = 0; i < N*N; i += N) {
            scanf("%s", &BLOCK[i]);
        }

        /*
        for (int i = 0; i < N*N; i++) {
            printf("%c ", BLOCK[i]);
            if ((i+1)%N == 0) {
                printf("\n");
            }
        }
        */

        printf("Case %d: %d\n", n_case, queen(N));
    }

    return 0;
}
