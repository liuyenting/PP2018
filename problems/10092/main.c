#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

int main(void) {
    cl_int status;

    // query platforms and devices
    cl_platform_id platform_id;
    status = clGetPlatformIDs(1, &platform_id, NULL);
    assert(status == CL_SUCCESS);

    cl_device_id device_id;
    status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);
    assert(status == CL_SUCCESS);

    // create context
    const cl_context_properties properties[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id,
        0
    };
    cl_context context = clCreateContext(properties, 1, &device_id, NULL, NULL, &status);
    assert(status == CL_SUCCESS);

    char filename[30];
    scanf("%s", filename);

    // create program
    unsigned char *program_file = NULL;
    size_t program_size = 0;
    FILE *fp = fopen(filename, "rb");
    assert(fp != 0);
    fseek(fp, 0, SEEK_END);
    program_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    program_file = malloc(program_size);
    if (!program_file) {
        fclose(fp);
        assert(program_file != 0);
    }
    fread(program_file, program_size, 1, fp);
    fclose(fp);

    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&program_file, &program_size, &status);
    assert(status == CL_SUCCESS);

    // retrieve build log
    size_t len = 0;
    status = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    char *build_log = calloc(len, sizeof(char));
    status = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, build_log, NULL);
    printf("%s", build_log);
    free(build_log);

    // release resources
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseDevice(device);

    return 0;
}
