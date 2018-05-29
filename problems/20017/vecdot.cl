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
    __local uint32_t buf[BLK_SIZE];
    uint32_t sum = 0;

    // determine range
    int gid = get_global_id(0);
    int lo = gid * BLK_SIZE;
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
    for (int u = BLK_SIZE; u > 1; u >>= 1) {
        int offset = u/2;
        if (ind < offset) {
            buf[ind] += buf[ind+offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (ind == 0) {
        atomic_add(&out_buf[get_group_id(0)&7], buf[0]);
    }
}
