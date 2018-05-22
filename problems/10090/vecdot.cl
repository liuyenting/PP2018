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
    const int N
) {
    int i = get_global_id(0);
    if (i >= N) {
        return;
    }
    prod[i] = encrypt(i, key1) * encrypt(i, key2);
}
