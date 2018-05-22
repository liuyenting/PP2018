#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define MAXFILENAME 30

int main(void) {
    cl_int status;

    /* query platform and device id */
    cl_platform_id platform_id;
    status = clGetPlatformIDs(1, &platform_id, NULL);
    assert(status == CL_SUCCESS);

    cl_device_id device_id;
    status =
        clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    assert(status == CL_SUCCESS);

    /* create context */
    cl_context context =
        clCreateContext(NULL, 1, &device_id, NULL, NULL, &status);
    assert(status == CL_SUCCESS);

    /* load program from file */
    char filename[MAXFILENAME];
    status = scanf("%s", filename);
    assert(status > 0);

    FILE *fp = fopen(filename, "r");
    assert(fp != NULL);

    fseek(fp, 0, SEEK_END);
    size_t len = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char *buffer = malloc(len);
    status = fread(buffer, len, 1, fp);
    cl_program program =
      clCreateProgramWithSource(context, 1, (const char **)&buffer, &len, &status);
    assert(status == CL_SUCCESS);
    free(buffer);

    /* build program */
    status = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    //NOTE program will fail due to requirement
    //assert(status == CL_SUCCESS);

    /* retrieve build log */
    status =
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    buffer = malloc(len);
    status =
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
    assert(status == CL_SUCCESS);
    printf("%s", buffer);

    /* release resources */
    clReleaseProgram(program);
    clReleaseContext(context);

    return 0;
}
