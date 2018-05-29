#include <assert.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#include <CL/cl.h>

#define BLK_SIZE    256     // local buffer size
#define BATCH_SIZE  256     // single submission

static char *
load_program_source(const char *filename) {
    struct stat statbuf;
    FILE *fh;
    char *source;

    fh = fopen(filename, "r");
    if (fh == 0) {
        return 0;
    }

    stat(filename, &statbuf);
    source = (char *)malloc(statbuf.st_size + 1);
    int status = fread(source, statbuf.st_size, 1, fh);
    source[statbuf.st_size] = '\0';

    return source;
}

int main(int argc, char *argv[]) {
    int status;

    /*
     * ===== INITIALIZE BEGIN =====
     */

    // query platform and device id
    cl_platform_id platform_id;
    status = clGetPlatformIDs(1, &platform_id, NULL);
    assert(status == CL_SUCCESS);

    cl_device_id device_id;
    status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    assert(status == CL_SUCCESS);

    // create context
    cl_context context;
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &status);
    assert(status == CL_SUCCESS);

    // create command queue
    cl_command_queue command;
    command = clCreateCommandQueue(context, device_id, 0, &status);
    assert(status == CL_SUCCESS);

    // load and build program
    char *source = load_program_source("vecdot.cl");
    assert(source != 0);

    cl_program program =
        clCreateProgramWithSource(context, 1, (const char **)&source, NULL, &status);
    if (!program || status != CL_SUCCESS) {
        fprintf(stderr, "failed to create compute program\n");
        return EXIT_FAILURE;
    }

    status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (status != CL_SUCCESS) {
        // retrieve build log
        size_t len;
        char *log_buf;
        status =
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        log_buf = calloc(len+1, sizeof(char));
        status =
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, log_buf, NULL);
        assert(status == CL_SUCCESS);
        printf("%s\n", log_buf);
        free(log_buf);
        return EXIT_FAILURE;
    }

    // create kernel
    cl_kernel kernel = clCreateKernel(program, "vecmul", &status);
    assert(kernel != 0 && status == CL_SUCCESS);

    free(source);

    /*
     * ===== INITIALIZE END =====
     */

    /* create buffers */
    cl_mem d_buf;
    d_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uint32_t)*8, NULL, &status);
    assert(status == CL_SUCCESS);

    const int ZERO = 0;
    int N;
    uint32_t key1, key2;
    while (scanf("%d %" PRIu32 " %" PRIu32, &N, &key1, &key2) == 3) {
        status = CL_SUCCESS;
        status |= clSetKernelArg(kernel, 0, sizeof(int), &N);
        status |= clSetKernelArg(kernel, 1, sizeof(uint32_t), &key1);
        status |= clSetKernelArg(kernel, 2, sizeof(uint32_t), &key2);
        status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_buf);
        assert(status == CL_SUCCESS);

        // execute kernel
        N = (N + BATCH_SIZE*BLK_SIZE - 1) / (BATCH_SIZE*BLK_SIZE) * BLK_SIZE;

        status = clEnqueueFillBuffer(
            command, d_buf,
            &ZERO,              // pattern
            sizeof(int),        // pattern size
            0,                  // buffer offset
            sizeof(uint32_t)*8, // buffer size
            0, NULL, NULL       // events
        );
        assert(status == CL_SUCCESS);

        size_t global_size[] = { N };
        size_t local_size[] = { BLK_SIZE };
        status = clEnqueueNDRangeKernel(
            command, kernel,
            1,              // work dimension
            NULL,           // global offset
            global_size,    // global size
            local_size,     // local size
            0, NULL, NULL   // events
        );
        assert(status == CL_SUCCESS);

        // read out
        uint32_t sum[8];
        status = clEnqueueReadBuffer(command, d_buf, CL_TRUE, 0, sizeof(uint32_t)*8, &sum, 0, NULL, NULL);
        assert(status == CL_SUCCESS);

        // final summed up
        for (int i = 1; i < 8; i++) {
            sum[0] += sum[i];
        }
        printf("%" PRIu32 "\n", sum[0]);
    }

    // release resources
    clReleaseMemObject(d_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command);
    clReleaseContext(context);

    return 0;
}
