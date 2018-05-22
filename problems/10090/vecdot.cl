static inline unsigned int rotate_left(unsigned int x, unsigned int n) {
    return  (x << n) | (x >> (32-n));
}

static inline unsigned int encrypt(unsigned int m, unsigned int key) {
    return (rotate_left(m, key&31) + key)^key;
}

__kernel void vecmul(
    const unsigned int key1,
    const unsigned int key2,
    __global unsigned int* prod,
    const int block_size,
    const int N
) {
    int offset = get_global_id(0);
    unsigned int psum = 0;
    for (int i = offset; i < N; i += block_size) {
        psum += encrypt(i, key1) * encrypt(i, key2);
    }
    prod[offset] = psum;
}
