#include "labeling.h"

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
