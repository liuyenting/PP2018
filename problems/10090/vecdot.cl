#include <stdint.h>

#include "utils.h"

__kernel void vecmul(
    __global uint32_t key1,
    __global uint32_t key2,
    __global uint32_t* prod
) {
    int i = get_global_id(0);
    prod[i] = encrypt(i, key1) * encrypt(i, key2);
}
