#include <stdio.h>
#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <x86intrin.h>

static inline uint32_t rotate_left(uint32_t x, uint32_t n) {
    return  (x << n) | (x >> (32-n));
}

static inline uint32_t encrypt(uint32_t m, uint32_t key) {
    return (_rotl(m, key&31) + key)^key;
}

static inline __m128i rotate_left_128(__m128i x, __m128i n) {
    return _mm_or_si128(
        _mm_sll_epi32(x, n),
        _mm_srl_epi32(x, _mm_sub_epi32(_mm_set_epi32(0, 0, 0, 32), n))
    );
}

static inline __m128i encrypt_128(__m128i m, __m128i key, __m128i mask) {
    return _mm_xor_si128(
        _mm_add_epi32(rotate_left_128(m, mask), key),
        key
    );
}

static uint32_t f(int N, int off, uint32_t key1, uint32_t key2) {
    uint32_t sum = 0;
    int i = 0, j = off;
    __m128i psum = _mm_set1_epi32(0);
    __m128i j_128 = _mm_set_epi32(off, off+1, off+2, off+3);
    const __m128i incr = _mm_set1_epi32(4);
    const __m128i key1_128 = _mm_set1_epi32(key1);
    const __m128i mask1_128 = _mm_set_epi32(0, 0, 0, key1&31);
    const __m128i key2_128 = _mm_set1_epi32(key2);
    const __m128i mask2_128 = _mm_set_epi32(0, 0, 0, key2&31);
    for (; i+4 < N; i+=4, j+=4) {
        psum = _mm_add_epi32(
            psum,
            _mm_mullo_epi32(
                encrypt_128(j_128, key1_128, mask1_128),
                encrypt_128(j_128, key2_128, mask2_128)
            )
        );
        j_128 = _mm_add_epi32(j_128, incr);
    }
    psum = _mm_add_epi32(psum, _mm_srli_si128(psum, 8));
    psum = _mm_add_epi32(psum, _mm_srli_si128(psum, 4));
    sum += (uint32_t)_mm_cvtsi128_si32(psum);
    for (; i < N; i++, j++)
        sum += encrypt(j, key1) * encrypt(j, key2);
    return sum;
}

int main() {
    int N;
    uint32_t key1, key2;
    while (scanf("%d %" PRIu32 " %" PRIu32, &N, &key1, &key2) == 3) {
        uint32_t sum = f(N, 0, key1, key2);
        printf("%" PRIu32 "\n", sum);
    }
    return 0;
}
