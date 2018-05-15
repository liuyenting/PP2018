/* header */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SIDE 2048

#define nLiveNeighbor(A, i, j) \
    A[i + 1][j] + A[i - 1][j] + A[i][j + 1] + \
    A[i][j - 1] + A[i + 1][j + 1] + A[i + 1][j - 1] + \
    A[i - 1][j + 1] + A[i - 1][j - 1]

char A[SIDE][SIDE];
char B[SIDE][SIDE];

void print(char A[SIDE][SIDE], int n) {
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++)
            printf("%d", A[i][j]);
        printf("\n");
    }
}

int main() {
    int n, generation, cell;
    scanf("%d%d", &n, &generation);
    for (int i = 1; i <= n; i++) {
        scanf("%s", &A[i][1]);
        for (int j = 1; j <= n; j++) {
            A[i][j] = (A[i][j] == '0') ? 0 : 1;
        }
    }

    int nln;
    for (int g = 0; g < generation; g++) {
        if (g % 2 == 0) {
             /*  from A to B */
#pragma omp parallel for collapse(2) \
                         schedule(static, 128)
            for (int i = 1; i <= n; i++) {
                for (int j = 1; j <= n; j++) {
                    nln = nLiveNeighbor(A, i, j);
                    B[i][j] = ((A[i][j] == 0 && nln == 3) ||
                        (A[i][j] == 1 && (nln == 2 || nln == 3)));
                }
            }
        } else {
            /*  from B to A */
#pragma omp parallel for collapse(2) \
                         schedule(static, 128)
            for (int i = 1; i <= n; i++) {
                for (int j = 1; j <= n; j++) {
                    nln = nLiveNeighbor(B, i, j);
                    A[i][j] = ((B[i][j] == 0 && nln == 3) ||
                        (B[i][j] == 1 && (nln == 2 || nln == 3)));
                }
            }
        }
    }

    if (generation % 2 == 0) {
        print(A, n);
    } else {
        print(B, n);
    }

    return 0;
}
