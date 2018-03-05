#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#define N           10000       // number of particles
#define CONST_G     1.0f
#define DT          0.5f    // time step
#define T           5.0f    // duration

struct vec3 {
    union {
        float val[3];
        struct {
            float x, y, z;
        };
    };
};

struct particle {
    float m;            // unit mass
    struct vec3 pos;    // position
    struct vec3 acc;    // acceleration
};

float rand_normalized() {
    return (float)rand() / (float)RAND_MAX;
}

void init_particles(int n, struct particle *list) {
    for (int i = 0; i < n; i++) {
        list[i].m = rand_normalized();
        for (int j = 0; j < 3; j++) {
            list[i].pos.val[j] = rand_normalized();
            list[i].acc.val[j] = 0.0f;
        }
    }
}

void print_particles(int n, struct particle *list) {
    for (int i = 0; i < n; i++) {
        printf("[%d]\n m=%f\n", i, list[i].m);
        printf(" pos=(%f, %f, %f)\n",
               list[i].pos.x, list[i].pos.y, list[i].pos.z);
        printf(" acc=(%f, %f, %f)\n",
               list[i].acc.x, list[i].acc.y, list[i].acc.z);
    }
    printf("\n");
}

float calculate_norm2(struct vec3 v1, struct vec3 v2) {
    return (v1.x-v2.x)*(v1.x-v2.x) +
           (v1.y-v2.y)*(v1.y-v2.y) +
           (v1.z-v2.z)*(v1.z-v2.z);
}

float calculate_acceleration(struct particle *p0, struct particle *p) {
    return CONST_G * p->m / calculate_norm2(p0->pos, p->pos);
}

void update_acceleration(int n, struct particle *list, int id) {
    struct particle *p0 = &list[id];
    for (int i = 0; i < n; i++) {
        // ignore current id
        if (i == id) {
            continue;
        }
        float acc = calculate_acceleration(p0, &list[i]);
        float r = sqrtf(calculate_norm2(p0->pos, list[i].pos));
        p0->acc.x = acc * (list[i].pos.x - p0->pos.x) / r;
        p0->acc.y = acc * (list[i].pos.y - p0->pos.y) / r;
        p0->acc.z = acc * (list[i].pos.z - p0->pos.z) / r;
    }
}

void update_position(struct particle *p) {
    p->pos.x += p->acc.x * DT;
    p->pos.y += p->acc.y * DT;
    p->pos.z += p->acc.z * DT;
}

void update_particles(int n, struct particle *list) {
    for (int i = 0; i < n; i++) {
        update_acceleration(n, list, i);
    }

    for (int i = 0; i < n; i++) {
        update_position(&list[i]);
    }
}

int main(int argc, char **argv) {
    struct particle list[N];
    printf("%d particles\n", N);

    srand(time(NULL));

    init_particles(N, &list[0]);
    printf("--- [INITIAL POSITION] ---\n");
    print_particles(N, &list[0]);

    for (float t = 0.0f; t < T; t += DT) {
        printf("t = %f\n", t);
        update_particles(N, &list[0]);
    }
    printf("\n");

    printf("--- [FINAL POSITION] ---\n");
    print_particles(N, &list[0]);
}
