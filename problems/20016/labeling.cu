#include "labeling.h"

#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

struct is_alphabet {
    __device__
    int operator()(const char c) const {
        return (c != '\n') ? 1 : 0;
    }
};

void CountPosition1(const char *text, int *pos, int text_size)
{
    thrust::transform(
        thrust::device,
        text,
        text + text_size,
        pos,
        is_alphabet()
    );

    thrust::inclusive_scan_by_key(
        thrust::device,
        pos,
        pos + text_size,
        pos,
        pos
    );
}

__global__
void count_position_kernel(
    const char *input,
    int *output,
    const int n
) {
    for (
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        i < n;
        i += blockDim.x * gridDim.x
    ) {
        if ((input[i] != ' ') && ((i == 0) || (input[i-1] == ' '))) {
            int j = i, c = 1;
            do {
                output[j++] = c++;
            } while ((input[j] != ' ') && (j < n));
        }
    }
}

void labeling(const char *text, int *pos, int text_size) {
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    cudaMemset(pos, 0, text_size*sizeof(int));
    count_position_kernel<<<32*numSMs, 256>>>(text, pos, text_size);
}
