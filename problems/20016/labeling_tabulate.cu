#include "labeling.h"

#include <thrust/tabulate.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

struct mark_spaces {
    const char *str;

    mark_spaces(const char *str)
        : str(str) {
    }

    __device__
    int operator()(int index) {
        // https://thrust.github.io/doc/group__transformations.html#ga7a231d3ed7e33397e36a20f788a0548c
        return (str[index] > ' ') ? -1 : index;
    }
};

struct sub_offset {
    int *offset;

    sub_offset(int *offset)
        : offset(offset) {
    }

    __device__
    int operator()(int index) {
        return index-offset[index];
    }
};

void labeling(const char *cuStr, int *cuPos, int strLen) {
    /*
        _ _ a  _ _ v  y  _ _ l  u  _  _  r  a  h
        0 1 -1 3 4 -1 -1 7 8 -1 -1 11 12 -1 -1 -1
     */
    thrust::tabulate(
        thrust::device,
        cuPos,          // beginning of the input sequence
        cuPos+strLen,   // end of the input sequence
        mark_spaces(cuStr)
    );

    /*
        _ _ a _ _ v y _ _ l u _  _  r  a  h
        0 1 1 3 4 4 4 7 8 8 8 11 12 12 12 12
     */
    thrust::inclusive_scan(
        thrust::device,
        cuPos,          // beginning of the input sequence
        cuPos+strLen,   // end of the input sequence
        cuPos,          // beginning of the output sequence, inplace
        thrust::maximum<int>()
    );

    /*
        _ _ a _ _ v y _ _ l u  _  _  r  a  h
        0 0 1 3 4 4 4 7 8 8 8  11 12 12 12 12
        0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
        0 0 1 0 0 1 2 0 0 1 2  0  0  1  2  3
     */
    thrust::tabulate(
        thrust::device,
        cuPos,          // beginning of the input sequence
        cuPos+strLen,   // end of the input sequence
        sub_offset(cuPos)
    );
}
