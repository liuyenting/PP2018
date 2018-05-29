#define BLK_SIZE 256    // 8 groups, 32*8=256

#define uint32_t unsigned int

static inline uint32_t rotate_left(uint32_t x, uint32_t n) {
    return  (x << n) | (x >> (32-n));
}

static inline uint32_t encrypt(uint32_t m, uint32_t key) {
    return (rotate_left(m, key&31) + key)^key;
}

__kernel void vecmul(
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
        for (int j = 0; j < 8; j++) {
            sum += encrypt(i, key1) * encrypt(i, key2);
            i++;
        }
    }
    buf[ind] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (ind < 128) {
        buf[ind] += buf[ind + 128];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (ind < 64) {
        buf[ind] += buf[ind + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (ind < 32) {
        // local partial sum
        #pragma unroll
        for (int j = 0; j < 6; j++) {
            buf[ind] += buf[ind + 1<<(5-j)];
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // global partial sum
        if (ind == 0) {
            atomic_add(&out_buf[get_group_id(0) & 7], buf[0]);
        }
    }
}
