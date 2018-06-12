#include "labeling.h"

#define BLOCK_SIZE 512

__global__
void labling_kernel(const char *cuStr, int *cuPos, const int strLen) {
    __shared__ int local_pos[BLOCK_SIZE];

    int pos_index = threadIdx.x + blockIdx.x*blockDim.x;
    int index = threadIdx.x;

    if (pos_index >= strLen) {
        return;
    }

    // thrust::tabulate, mark_spaces
    local_pos[index] = (cuStr[pos_index] != ' ') ? -1 : index;
    __syncthreads();

    // thrust::inclusive_scan, thrust::maximum<int>
    for (int offset = 1; offset <= index; offset *= 2) {
        if (local_pos[index] < local_pos[index-offset]) {
            local_pos[index] = local_pos[index-offset];
        }
        __syncthreads();
    }

    // thrust::tabulate, sub_offset
    cuPos[pos_index] = index - local_pos[index];
}

__global__
void patch_kernel(int *cuPos, const int strLen) {
    int pos_index = threadIdx.x + blockIdx.x*blockDim.x;
    int index = threadIdx.x;

    if (pos_index >= strLen) {
        return;
    }

    // cross blocks
    if (blockIdx.x > 0 && cuPos[pos_index] == (index+1)) {
        cuPos[pos_index] += cuPos[blockIdx.x*blockDim.x-1];
    }
}

void labeling(const char *cuStr, int *cuPos, int strLen) {
    int n_blocks = (strLen + BLOCK_SIZE-1) / BLOCK_SIZE;
    labling_kernel<<<n_blocks, BLOCK_SIZE>>>(cuStr, cuPos, strLen);
    patch_kernel<<<n_blocks, BLOCK_SIZE>>>(cuPos, strLen);
}
