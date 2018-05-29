#define BLK_SIZE 256    // 8 groups, 32*8=256

#define uint32_t unsigned int

static inline uint32_t rotate_left(uint32_t x, uint32_t n) {
    return  (x << n) | (x >> (32-n));
}

static inline uint32_t encrypt(uint32_t m, uint32_t key) {
    return (rotate_left(m, key&31) + key)^key;
}

__kernel void vecdot(
    int N,
    uint32_t key1, uint32_t key2,
    __global uint32_t* out_buf
) {
    __local uint32_t buf[BLK_SIZE+1];
    uint32_t sum;

    // determine range
    int offset = get_global_id(0);
    int lo = offset * BLK_SIZE;
    int hi = lo + BLK_SIZE;
    if (hi > N) {
        hi = N;
    }

    // the actual dot product
    int ind = get_local_id(0);
    for (int i = lo; i < hi; ) {
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            sum += encrypt(i, key1) * encrypt(i, key2);
            i++;
        }
    }
    buf[ind] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    // local partial sum
    #pragma unroll
    for (int u = 0; u <= 7; u++) {
        if (ind < (1<<(7-u))) {
            buf[ind] += buf[ind + (1<<(7-u))];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (ind == 0) {
        out_buf[get_group_id(0)] = buf[0];
    }
}
