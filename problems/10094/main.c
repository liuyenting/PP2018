#include <omp.h>
#include <stdio.h>
#include <string.h>

#define MAX(a, b) (a > b) ? a : b

#define MAX_N 10000
#define MAX_M 1000000

int W[MAX_N], V[MAX_N];
int DP[2][MAX_M+1]; // [0][] = skip, [1][] = selected

int main(void) {
    int N, M;
    scanf("%d %d", &N, &M);

    for (int i = 0; i < N; i++) {
        scanf("%d %d", &W[i], &V[i]);
    }
    memset(DP, 0, sizeof(int)*(M+1)*2);

    int i_in = 0, i_out;
    for (int i = 0; i < N; i++) {
        i_out = 1 - i_in;
        int W_curr = W[i], V_curr = V[i];
        #pragma omp parallel
        {
            #pragma omp for
            for (int i = W_curr; i <= M; i++) {
                DP[i_out][i] = MAX(DP[i_in][i-W_curr]+V_curr, DP[i_in][i]);
            }
            #pragma omp for
            for (int i = 0; i < W_curr; i++) {
                DP[i_out][i] = DP[i_in][i];
            }
        }
        i_in = i_out;
    }
    printf("%d\n", DP[i_out][M]);

    return 0;
}
