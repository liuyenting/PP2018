#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

//#define DEBUG

#ifdef DEBUG
#define dprintf(_f, ...) printf((_f), ##__VA_ARGS__)
#else
#define dprintf(_f, ...) {}
#endif

typedef struct {
    int w, h;
    int *x;
} image_t;

void read_image(image_t *im) {
    int h = im->h, w = im->w;
    im->x = calloc(h*w, sizeof(int));
    for (int i = 0; i < h*w; i++) {
        scanf("%d", im->x + i);
    }
}

void print_image(image_t *im) {
    int h = im->h, w = im->w;
    dprintf("h=%d, w=%d\n", h, w);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            dprintf("%d ", im->x[i*w+j]);
        }
        dprintf("\n");
    }
}

void free_image(image_t *im) {
    if (im->x != 0) {
        free(im->x);
        im->x = 0;
    }
}

long long diff_image(image_t *ref, image_t *test, int oh, int ow) {
    int wr = ref->w;
    int ht = test->h, wt = test->w;

    long long diff = 0;
    //#pragma omp parallel for reduction(+ : diff)
    for (int i = 0; i < ht; i++) {
        for(int j = 0; j < wt; j++) {
            dprintf("r[%d][%d]=%d, t[%d][%d]=%d\n",
                   i+oh, j+ow, ref->x[(i+oh)*wr+(j+ow)], i, j, test->x[i*wt+j]);
            long long v = ref->x[(i+oh)*wr+(j+ow)] - test->x[i*wt+j];
            diff += v*v;
        }
    }
    return diff;
}

int main(void) {
    #ifdef DEBUG
    omp_set_num_threads(1);
    #endif

    image_t a, b;
    int c = 0;
    while(scanf("%d %d %d %d", &a.h, &a.w, &b.h, &b.w) != EOF) {
        read_image(&a);
        read_image(&b);

        dprintf("[a]\n");
        print_image(&a);
        dprintf("[b]\n");
        print_image(&b);

        long long diff = -1, x, y;
        #pragma omp parallel for
        for (int oh = 0; oh <= a.h-b.h; oh++) {
            for (int ow = 0; ow <= a.w-b.w; ow++) {
                long long t = diff_image(&a, &b, oh, ow);
                dprintf("oh=%d, ow=%d, diff=%lld\n", oh, ow, t);
                #pragma omp critical
                {
                    if ((diff < 0) ||
                        (t < diff) ||
                        ((t == diff) && ((oh < y) || ((oh == y) && (ow < x))))
                    ) {
                        dprintf("update!\n");
                        diff = t;
                        x = ow, y = oh;
                    }
                }
            }
        }
        dprintf("[result]\n");
        printf("%d %d\n", y+1, x+1);

        free_image(&a);
        free_image(&b);

        dprintf("\n");
    }

    return 0;
}
