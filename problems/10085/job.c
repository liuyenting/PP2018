#include "utils.h"

#define OFFSET 32

int ret[128];
int run(int n, int key) {
    int sum = 0;
    f(n, key, ret, ret+OFFSET, ret+OFFSET*2, ret+OFFSET*3);
    for (int i = 0; i < 4; i++)
        sum += ret[OFFSET*i];
    return sum;
}
