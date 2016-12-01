#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <ctype.h>
#include <CL/opencl.h>

#define cl_ret_on_err(errstring,errnum,ret){ \
  if (errnum != CL_SUCCESS) { \
    fprintf(stderr, "%s returned %d at line %d\n", errstring, errnum, __LINE__); \
    return ret; \
  } \
}

#define TRUE  1
#define FALSE 0

static int target_device = 0;
static int detail = FALSE;

void help ()
{
  fprintf (stderr, "Usage:\n");
  fprintf (stderr, "-g: Use the discreet GPU\n");
  fprintf (stderr, "-d: Detailed messages\n");
}

int parse_params(int argc, char *argv[])
{
  int c;
  while ((c = getopt (argc, argv, "gdh")) != -1) {
    switch (c) {
      case 'g':
        target_device = 1;
        break;
      case 'd':
        detail = TRUE;
        break;
      case '?':
        if (isprint (optopt))
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf (stderr,
                   "Unknown option character `\\x%x'.\n",
                   optopt);
      case 'h':
        help();
        return FALSE;
      default:
        abort ();
    }
  }
  fprintf(stdout, "Done\n");
  return TRUE;
}


int main(int argc, char *argv[])
{
  cl_platform_id    platform;
  cl_device_id    devices[2], device;
  cl_context    context;
  cl_command_queue  gpu_queue;
  cl_kernel kernel;
  cl_program program;
  cl_mem input;
  cl_mem output;
  cl_mem output2;
  cl_mem output3;
  size_t global[2];
  cl_int err;
  unsigned int rgbstride,uvstride,ystride;
  unsigned char *xrgb,*y,*u,*v;
  int i;
  struct timeval base,aux,diff;
  FILE *fd;
  size_t size;
  char *prog_char;
  int width = 1290, height = 1080;
  cl_uint numdevices = 0;
  int iterations = 100;

  if (!parse_params(argc, argv))
    return -1;

  fprintf(stdout, "Malloc data structs\n");
  xrgb = malloc(width*height*4);
  if (!xrgb){
    perror("malloc");
    return -1;
  }
  y = malloc(width * height);
  if (!y){
    perror("malloc");
    return -1;
  }
  u = malloc(width/2 * height/2);
  if (!u){
    perror("malloc");
    return -1;
  }
  v = malloc(width/2 * height/2);
  if (!v){
    perror("malloc");
    return -1;
  }

  for (i=0;i<width*height;i++)
    memset(&xrgb[i*4],i&0xff,4);

  mlockall(MCL_CURRENT);

  /* Initialize the platform, context and queues */
  fprintf(stdout, "Initialize context\n");
  err = clGetPlatformIDs(1, &platform, NULL);
  cl_ret_on_err("clGetPlatformIDs", err, -1);

  /* Get the CPU device */
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 2, devices, &numdevices);
  cl_ret_on_err("clGetDeviceIDs", err, -1);

  if (!numdevices) {
    fprintf(stderr, "No GPU devices found\n");
    return -1;
  }

  fprintf(stdout, "Found %d GPUs\n", numdevices);

  if (target_device >= numdevices)
    target_device = numdevices-1;
  fprintf(stdout, "Using GPU %d\n", target_device);
  device = devices[target_device];

  /* Create OpenCL context */
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  cl_ret_on_err("clClreateContext", err, -1);

  /* Create command queue */
  gpu_queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
  cl_ret_on_err("clClreateCommandQueue", err, -1);

  fprintf(stdout, "Creating program\n");
  /* Create program */
  fd = fopen("i420-8.cl","r");
  if (!fd){
    perror("i420-8.cl");
    return -1;
  }
  fseek(fd, 0L, SEEK_END);
  size = ftell(fd);
  fseek(fd, 0L, SEEK_SET);

  prog_char = malloc(size+1);
  if (!prog_char){
    perror("malloc");
    return -1;
  }

  if (fread(prog_char, size,1,fd)!=1){
    perror("i420-8.cl");
    return -1;
  }
  fclose(fd);

  prog_char[size] = '\0';


  program = clCreateProgramWithSource(context, 1, (const char **) & prog_char, NULL, &err);
  cl_ret_on_err("clCreateProgramWithSource", err, -1);

  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t buildlog_length;
    char *buildlog;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &buildlog_length);

    buildlog = (char *)malloc(buildlog_length + 1);

    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, buildlog_length, buildlog, NULL);
    fprintf (stderr,"Program build errors:\n%s", buildlog);

    free(buildlog);
    cl_ret_on_err("clBuildProgram", err, -1);
  }

  kernel = clCreateKernel(program, "xrgb_to_i420_kernel", &err);
  cl_ret_on_err("clCreateKernel", err, -1);

  gettimeofday(&base,NULL);

  fprintf(stdout, "Running computation for %d times\n", iterations);
  for (i=0;i<iterations;i++){
    if (detail)
      fprintf(stdout, "iteration %d\n", i+1);
    input = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,  sizeof(*xrgb) * width* height * 4, xrgb, &err);
    cl_ret_on_err("clCreateBuffer", err, -1);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY| CL_MEM_USE_HOST_PTR, sizeof(*y) * width*height, y, &err);
    cl_ret_on_err("clCreateBuffer", err, -1);
    output2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY| CL_MEM_USE_HOST_PTR, sizeof(*u) * width/2*height/2, u, &err);
    cl_ret_on_err("clCreateBuffer", err, -1);
    output3 = clCreateBuffer(context, CL_MEM_WRITE_ONLY| CL_MEM_USE_HOST_PTR, sizeof(*v) * width/2*height/2, v, &err);
    cl_ret_on_err("clCreateBuffer", err, -1);

    gettimeofday(&aux,NULL);
    timersub(&aux,&base,&diff);
    //fprintf(stderr, "Create %8ld usec\n", diff.tv_usec);

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    cl_ret_on_err("clSetKernelArg", err, -1);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    cl_ret_on_err("clSetKernelArg", err, -1);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &output2);
    cl_ret_on_err("clSetKernelArg", err, -1);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &output3);
    cl_ret_on_err("clSetKernelArg", err, -1);
    rgbstride = width;
    err = clSetKernelArg(kernel, 4, sizeof(unsigned int), &rgbstride);
    cl_ret_on_err("clSetKernelArg", err, -1);
    ystride = width /8;
    err = clSetKernelArg(kernel, 5, sizeof(unsigned int), &ystride);
    uvstride = width /8;
    err = clSetKernelArg(kernel, 6, sizeof(unsigned int), &uvstride);
    cl_ret_on_err("clSetKernelArg", err, -1);

    gettimeofday(&base,NULL);

    clEnqueueMapBuffer( gpu_queue, input, CL_TRUE, CL_MAP_READ, 0, sizeof(*xrgb)*width*height*4, 0, NULL, NULL,&err );

    cl_ret_on_err("clEnqueueMapBuffer", err, -1);
    gettimeofday(&aux,NULL);
    timersub(&aux,&base,&diff);
    //fprintf(stderr, "Enque  %8ld usec\n", diff.tv_usec);

    global[0] = width/8;
    global[1] = height;
    err = clEnqueueNDRangeKernel(gpu_queue, kernel, 2, NULL, global,NULL, 0, NULL, NULL);
    cl_ret_on_err("clEnqueueNDRangeKernel", err, -1);
    //clFinish(gpu_queue);

    gettimeofday(&aux,NULL);
    timersub(&aux,&base,&diff);
    //fprintf(stderr, "kernel %8ld usec\n", diff.tv_usec);

    gettimeofday(&base,NULL);
    clEnqueueMapBuffer( gpu_queue, output, CL_FALSE, CL_MAP_WRITE, 0, sizeof(*y)*width*height, 0, NULL, NULL,&err );
    cl_ret_on_err("clEnqueueMapBuffer", err, -1);
    clEnqueueMapBuffer( gpu_queue, output2, CL_FALSE, CL_MAP_WRITE, 0, sizeof(*u)*width/2*height/2, 0, NULL, NULL,&err );
    cl_ret_on_err("clEnqueueMapBuffer", err, -1);
    clEnqueueMapBuffer( gpu_queue, output3, CL_TRUE, CL_MAP_WRITE, 0, sizeof(*v)*width/2*height/2, 0, NULL, NULL,&err );
    cl_ret_on_err("clEnqueueMapBuffer", err, -1);
    gettimeofday(&aux,NULL);
    timersub(&aux,&base,&diff);

    //fprintf(stderr, "Readbu %8ld usec\n", diff.tv_usec);


    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseMemObject(output2);
    clReleaseMemObject(output3);
  }
  fprintf(stdout, "Computation complete\n");
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(gpu_queue);
  clReleaseDevice(device);
  clReleaseContext(context);
  return 0;
}
