#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include <CL/cl.h>
#include <omp.h>

#define MAX_GPU     4       // maximum allowed gpu numbers
#define MAX_CASES   10000   // maximum cases to process
#define BLK_SIZE    256     // local buffer size
#define BATCH_SIZE  256     // single submission

const char *
clGetErrorString(cl_int error) {
    switch(error){
        // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
}

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

    cl_device_id device_id[MAX_GPU];
    int n_gpu;
    status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, MAX_GPU, device_id, &n_gpu);
    assert(status == CL_SUCCESS);

    cl_context context[MAX_GPU];
    cl_command_queue command[MAX_GPU];
    cl_program program[MAX_GPU];
    cl_kernel kernel[MAX_GPU];
    cl_mem d_buf[MAX_GPU];
    // load program
    char *source = load_program_source("vecdot.cl");
    assert(source != 0);
    // build program for all valid devices
    int compile_status = CL_SUCCESS;
    #pragma omp parallel for reduction(| : compile_status)
    for (int i = 0; i < n_gpu; i++) {
        // create context
        context[i] = clCreateContext(0, 1, device_id+i, NULL, NULL, &status);
        assert(status == CL_SUCCESS);

        // create command queue
        command[i] = clCreateCommandQueue(context[i], device_id[i], 0, &status);
        assert(status == CL_SUCCESS);

        // build program
        program[i] =
            clCreateProgramWithSource(context[i], 1, (const char **)&source, NULL, &status);
        if (!program || status != CL_SUCCESS) {
            fprintf(stderr, "failed to create compute program\n");
            compile_status |= status;
            continue;
        }

        // program associates with context, only 1 device is allowed
        status = clBuildProgram(program[i], 1, device_id+i, NULL, NULL, NULL);
        if (status != CL_SUCCESS) {
            // retrieve build log
            size_t len;
            char *log_buf;
            status =
                clGetProgramBuildInfo(program[i], device_id[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
            log_buf = calloc(len+1, sizeof(char));
            status =
                clGetProgramBuildInfo(program[i], device_id[i], CL_PROGRAM_BUILD_LOG, len, log_buf, NULL);
            assert(status == CL_SUCCESS);
            printf("=== dev %d ===\n%s\n", i, log_buf);
            free(log_buf);
            compile_status |= status;
            continue;
        }

        // create kernel
        kernel[i] = clCreateKernel(program[i], "vecdot", &status);
        assert(kernel != 0 && status == CL_SUCCESS);

        // create buffers
        d_buf[i] = clCreateBuffer(context[i], CL_MEM_WRITE_ONLY, sizeof(uint32_t)*8, NULL, &status);
        assert(status == CL_SUCCESS);
    }
    free(source);
    assert(compile_status == CL_SUCCESS);

    /*
     * ===== INITIALIZE END =====
     */

    // 1 thread per gpu
    omp_set_num_threads(n_gpu);

    const int ZERO = 0;
    int n_cases, N[MAX_CASES];
    uint32_t key1[MAX_CASES], key2[MAX_CASES], result[MAX_CASES];
    // read all the data
    for (n_cases = 0;
         scanf("%d %" PRIu32 " %" PRIu32, N+n_cases, key1+n_cases, key2+n_cases) == 3 &&
         n_cases < MAX_CASES;
         n_cases++) {
    }
    // distribute the tasks
    #pragma omp parallel for schedule(dynamic, 8) private(status)
    for (int i = 0; i < n_cases; i++) {
        // 1 thread per gpu
        int tid = omp_get_thread_num();

        status = CL_SUCCESS;
        status |= clSetKernelArg(kernel[tid], 0, sizeof(int), N+i);
        status |= clSetKernelArg(kernel[tid], 1, sizeof(uint32_t), key1+i);
        status |= clSetKernelArg(kernel[tid], 2, sizeof(uint32_t), key2+i);
        status |= clSetKernelArg(kernel[tid], 3, sizeof(cl_mem), d_buf+tid);
        assert(status == CL_SUCCESS);

        // execute kernel
        status = clEnqueueFillBuffer(
            command[tid], d_buf[tid],
            &ZERO,              // pattern
            sizeof(int),        // pattern size
            0,                  // buffer offset
            sizeof(uint32_t)*8, // buffer size
            0, NULL, NULL       // events
        );
        assert(status == CL_SUCCESS);

        size_t global_size[] = {
            ((((N[i]+BATCH_SIZE - 1) / BATCH_SIZE)+BLK_SIZE - 1) / BLK_SIZE) * BLK_SIZE
        };
        size_t local_size[] = { BLK_SIZE };
        status = clEnqueueNDRangeKernel(
            command[tid], kernel[tid],
            1,              // work dimension
            NULL,           // global offset
            global_size,    // global size
            local_size,     // local size
            0, NULL, NULL   // events
        );
        assert(status == CL_SUCCESS);

        // read out
        uint32_t sum[8];
        status = clEnqueueReadBuffer(
            command[tid], d_buf[tid],
            CL_TRUE,            // blocking read
            0,                  // read offset
            sizeof(uint32_t)*8, // bytes to read
            sum,                // host memory
            0, NULL, NULL       // events
        );
        assert(status == CL_SUCCESS);

        // final summed up
        for (int i = 1; i < 8; i++) {
            sum[0] += sum[i];
        }
        result[i] = sum[0];
    }
    // print the result
    for (int i = 0; i < n_cases; i++) {
        printf("%" PRIu32 "\n", result[i]);
    }

    // release resources
    for (int i = 0; i < n_gpu; i++) {
        clReleaseMemObject(d_buf[i]);
        clReleaseKernel(kernel[i]);
        clReleaseProgram(program[i]);
        clReleaseCommandQueue(command[i]);
        clReleaseContext(context[i]);
    }

    return 0;
}
