#include "labeling.h"

#include <thrust/tabulate.h>
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

void labeling(const char *cuStr, int *cuPos, int strLen) {
    thrust::tabulate(
        thrust::device,
        cuPos,          // beginning of the input sequence
        cuPos+strLen,   // end of the input sequence
        mark_spaces(cuStr)
    );

}
