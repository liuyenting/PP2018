#include "VSQRT.h"

#include <math.h>
#include <x86intrin.h>

void sqrt2(float *begin, float *end) {
    for (; begin+8 <= end; begin += 8)
        _mm256_store_ps(begin, _mm256_sqrt_ps(_mm256_load_ps(begin)));
    for (; begin+4 <= end; begin += 4)
        _mm_store_ps(begin, _mm_sqrt_ps(_mm_load_ps(begin)));
    for (; begin != end; begin++)
        *begin = sqrt(*begin);
}
