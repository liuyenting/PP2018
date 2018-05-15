/* header */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SIDE 2048

#define nLiveNeighbor(G, f, i, j) \
    G[f][i + 1][j] + G[f][i - 1][j] + G[f][i][j + 1] + \
    G[f][i][j - 1] + G[f][i + 1][j + 1] + G[f][i + 1][j - 1] + \
    G[f][i - 1][j + 1] + G[f][i - 1][j - 1]

char G[2][SIDE][SIDE];

void print(char G[2][SIDE][SIDE], int flag, int n) {
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++)
            printf("%d", G[flag][i][j]);
        printf("\n");
    }
}

int main() {
    int n, generation, cell;
    scanf("%d%d", &n, &generation);
    for (int i = 1; i <= n; i++) {
        scanf("%s", &G[0][i][1]);
        for (int j = 1; j <= n; j++) {
            G[0][i][j] -= '0';
        }
    }

#pragma omp parallel
    for (int g = 0, flag = 0; g < generation; g++, flag = 1-flag) {
#pragma omp for
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                int nln = nLiveNeighbor(G, flag, i, j);
                G[1-flag][i][j] = ((G[flag][i][j] == 0 && nln == 3) ||
                    (G[flag][i][j] == 1 && (nln == 2 || nln == 3)));
            }
        }
    }
    print(G, generation%2, n);

    return 0;
}
