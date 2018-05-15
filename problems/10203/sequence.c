#include <stdio.h>

#define MAXN 2048
#define INF (1LL<<60)

int N;
long long dp[MAXN*MAXN], SZ[MAXN+1];

int main() {
    while (scanf("%d", &N) == 1) {
        for (int i = 0; i <= N; i++)
            scanf("%lld", &SZ[i]);
        for (int i = 0; i < N; i++)
            dp[i*N+i] = 0;

#pragma omp parallel
        for (int i = 1; i < N; i++) {
#pragma omp for
            for (int j = 0; j < N-i; j++) {
                int l = j, r = j+i;
                int *dpptr = &dp[l*N+l];
                long long local = INF, sz = SZ[l]*SZ[r+1];
                for (int k = l; k < r; k++, dpptr++) {
                    long long t = *dpptr + dp[(k+1)*N+r] + sz*SZ[k+1];
                    if (t < local)
                        local = t;
                }
                *(dpptr++) = local;
            }
        }
        printf("%lld\n", dp[0*N+N-1]);
    }
    return 0;
}
