#include <assert.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <inttypes.h>

#include <CL/cl.h>

#include "utils.h"

#define MAXN 16777216

static char *
load_program_source(const char *filename) {
    struct stat statbuf;
    FILE *fh;
    char *source;

    fh = fopen(filename, "r");
    if (fh == 0)
        return 0;

    stat(filename, &statbuf);
    source = (char *)malloc(statbuf.st_size + 1);
    int err = fread(source, statbuf.st_size, 1, fh);
    source[statbuf.st_size] = '\0';

    return source;
}

int main(int argc, char *argv[]) {
    int err;
    cl_mem buffer;
    uint32_t *h_buffer;

    /* query platform and device id */
    cl_platform_id platform_id;
    status = clGetPlatformIDs(1, &platform_id, NULL);
    assert(err == CL_SUCCESS);

    cl_device_id device_id;
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    assert (err == CL_SUCCESS);

    /* load and build program */
    char *source = load_program_source("vecdot.cl");
    if (!source) {
        fprintf(stderr, "failed to load program from file\n");
        return EXIT_FAILURE;
    }

    /* create context */
    cl_context context;
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context) {
        fprintf(stderr, "failed to create a compute context\n");
        return EXIT_FAILURE;
    }

    /* create command queue */
    cl_command_queue command;
    command = clCreateCommandQueue(context, device_id, 0, &err);
    if (!command) {
        fprintf(stderr, "failed to create a command commands\n");
        return EXIT_FAILURE;
    }

    /* create buffer */
    size_t buf_size = sizeof(uint32_t) * MAXN;
    buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, buf_size, NULL, NULL);
    if (!buffer) {
        fprintf(stderr, "failed to allocate result buffer on device\n");
        return EXIT_FAILURE;
    }
    h_buffer = malloc(buf_size);

    /* create and build program */
    cl_program program =
        clCreateProgramWithSource(context, 1, (const char **)&source, NULL, &err);
    if (!program || err != CL_SUCCESS) {
        fprintf(stderr, "failed to create compute program\n");
        return EXIT_FAILURE;
    }

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char log_buf[2048];
        fprintf(stderr, "%s\n", source);
        fprintf(stderr, "failed to build program executable\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(log_buf), log_buf, &len);
        printf("%s\n", log_buf);
        return EXIT_FAILURE;
    }

    /* create kernel */
    cl_kernel kernel = clCreateKernel(program, "vecmul", &err);
    if (!kernel || err != CL_SUCCESS) {
        fprintf(stderr, "failed to create compute kernel\n");
        return EXIT_FAILURE;
    }

    free(source);

    int N;
    uint32_t key1, key2;
    while (scanf("%d %" PRIu32 " %" PRIu32, &N, &key1, &key2) == 3) {
        for (int i = 0; i < N; i++) {
            err = CL_SUCCESS;
            err |= clSetKernelArg(kernel, 0, sizeof(uint32_t), &key1);
            err |= clSetKernelArg(kernel, 1, sizeof(uint32_t), &key2);
            err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer);
            err |= clSetKernelArg(kernel, 3, sizeof(int), &N);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "failed to set kernel arguments\n");
                return EXIT_FAILURE;
            }

            size_t global = (N + 31)/32;
            size_t local = 32;

            err = CL_SUCCESS;
            err |= clEnqueueNDRangeKernel(
                command,
                kernel,
                1,              // work dimension
                NULL,           // global work offset
                &global,        // global work size
                &local,         // local work size
                0, NULL, NULL   // events
            );
            if (err != CL_SUCCESS) {
                fprintf(stderr, "failed to execute kernel\n");
                return EXIT_FAILURE;
            }
        }

        err = clFinish(command);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "failed to wait for command queue to finish, %d\n", err);
            return EXIT_FAILURE;
        }

        /* read back the result */
        err = clEnqueueReadBuffer(commands, buffer, CL_TRUE, 0, buf_size, h_buffer, 0, NULL, NULL);
        if (err) {
            fprintf(stderr, "failed to read back results from the device\n");
            return EXIT_FAILURE;
        }

        uint32_t sum = 0;
#pragma omp parallel for schedule(static) \
                         reduction(+: sum)
        for (int i = 0; i < N; i++) {
            sum += h_buffer[i];
        }

        printf("%" PRIu32 "\n", sum);
    }

    /* release resources */
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(buffer);
    clReleaseCommandQueue(command);
    clReleaseContext(context);

    free(h_buffer);

    return 0;
}
