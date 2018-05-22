#ifndef _UTILS_H
#define _UTILS_H

static inline cl_uint rotate_left(cl_uint x, cl_uint n) {
    return  (x << n) | (x >> (32-n));
}
static inline cl_uint encrypt(cl_uint m, cl_uint key) {
    return (rotate_left(m, key&31) + key)^key;
}

#endif
