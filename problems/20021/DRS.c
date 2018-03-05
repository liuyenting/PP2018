#include <x86intrin.h>
#include "DRS.h"

int32_t search_range(Rect rect, int32_t x[], int32_t y[],
        int32_t w[], int32_t n) {
    int32_t ret = 0;
    __m128i pret = _mm_set1_epi32(0);
    const __m128i lx_128 = _mm_set1_epi32(rect.lx);
    const __m128i rx_128 = _mm_set1_epi32(rect.rx);
    const __m128i ly_128 = _mm_set1_epi32(rect.ly);
    const __m128i ry_128 = _mm_set1_epi32(rect.ry);
    __m128i mask = _mm_set1_epi32(0);
    const __m128i incr = _mm_set1_epi32(4);
    int i = 0;
    for (; i+4 < n; i+=4) {
        __m128i x_128 = _mm_load_si128((__m128i *)(&x[i]));
        __m128i y_128 = _mm_load_si128((__m128i *)(&y[i]));

        mask = _mm_and_si128(
            _mm_and_si128(
                _mm_or_si128(
                    _mm_cmplt_epi32(lx_128, x_128),
                    _mm_cmpeq_epi32(lx_128, x_128)
                ),
                _mm_or_si128(
                    _mm_cmplt_epi32(x_128, rx_128),
                    _mm_cmpeq_epi32(x_128, rx_128)
                )
            ),
            _mm_and_si128(
                _mm_or_si128(
                    _mm_cmplt_epi32(ly_128, y_128),
                    _mm_cmpeq_epi32(ly_128, y_128)
                ),
                _mm_or_si128(
                    _mm_cmplt_epi32(y_128, ry_128),
                    _mm_cmpeq_epi32(y_128, ry_128)
                )
            )
        );

        pret = _mm_add_epi32(pret, _mm_maskload_epi32(&w[i], mask));
    }
    pret = _mm_add_epi32(pret, _mm_srli_si128(pret, 8));
    pret = _mm_add_epi32(pret, _mm_srli_si128(pret, 4));
    ret += _mm_cvtsi128_si32(pret);
    for (; i < n; i++) {
        if (rect.lx <= x[i] && x[i] <= rect.rx &&
            rect.ly <= y[i] && y[i] <= rect.ry) {
            ret += w[i];
        }
    }
    return ret;
}
