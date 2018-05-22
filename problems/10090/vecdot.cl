static inline cl_uint rotate_left(cl_uint x, cl_uint n) {
    return  (x << n) | (x >> (32-n));
}

static inline cl_uint encrypt(cl_uint m, cl_uint key) {
    return (rotate_left(m, key&31) + key)^key;
}

__kernel void vecmul(
    const cl_uint key1,
    const cl_uint key2,
    __global cl_uint* prod,
    const cl_uint N
) {
    int i = get_global_id(0);
    if (i >= N) {
        return;
    }
    prod[i] = encrypt(i, key1) * encrypt(i, key2);
}
