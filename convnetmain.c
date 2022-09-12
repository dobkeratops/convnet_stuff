#include <stdio.h>
#include <stdlib.h>
#include <OpenCL/opencl.h>

#include "SDL2/SDL.h"
#include "SDL_image.h"


const char* g_kernel="	\n\
__kernel void vector_add(__global const int *A, __global const int *B, __global int *C) {	\n\
 									\n\
    // Get the index of the current element to be processed		\n\
    int i = get_global_id(0);						\n\
 									\n\
    // Do the operation							\n\
    C[i] = A[i] + B[i];							\n\
}									\n\
\0";

cl_device_id g_cl_device=0;
cl_context gcl;
cl_command_queue gclq;

void opencl_init() {
    cl_uint num_devices, i;
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);

    cl_device_id* devices = calloc(sizeof(cl_device_id), num_devices);
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

    char buf[128];
    for (i = 0; i < num_devices; i++) {
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 128, buf, NULL);
        fprintf(stdout, "Device %s supports ", buf);

        clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, 128, buf, NULL);
	// just use the last we find?
	g_cl_device = devices[i];
        fprintf(stdout, "%s\n", buf);
    }

    free(devices);

	cl_int ret;
	gcl = clCreateContext(NULL,1,&g_cl_device, NULL, NULL, &ret);
	gclq = clCreateCommandQueue(gcl,g_cl_device,0,&ret);

}

#define MALLOCS(TYPE,NUM) ((TYPE*)malloc(sizeof(TYPE)*(NUM)))

// does it work?
void test_opencl() {
	cl_int ret;
	int testsize=64;
	cl_mem array_a =  clCreateBuffer(gcl,CL_MEM_READ_ONLY, testsize*sizeof(int), NULL, &ret);
	cl_mem array_b =  clCreateBuffer(gcl,CL_MEM_READ_ONLY, testsize*sizeof(int), NULL, &ret);
	cl_mem array_c =  clCreateBuffer(gcl,CL_MEM_WRITE_ONLY, testsize*sizeof(int), NULL, &ret);

	int* data_a=MALLOCS(int, testsize);
	int* data_b=MALLOCS(int, testsize);
	int* data_c=MALLOCS(int, testsize);
	int i;
	for (i=0; i<testsize; i++) {
		data_a[i]=i;
		data_b[i]=i*10000;
		data_c[i]=0;
	}
	
	ret=clEnqueueWriteBuffer(gclq, array_a, CL_TRUE,0, testsize*sizeof(int), data_a, 0, NULL,NULL);
	ret=clEnqueueWriteBuffer(gclq, array_b, CL_TRUE,0, testsize*sizeof(int), data_b, 0, NULL,NULL);

	size_t srclen=strlen(&g_kernel);
	printf("create program\n");
	cl_program  prg = clCreateProgramWithSource(gcl, 1, (const char**) &g_kernel,(const size_t*)&srclen, &ret);
	printf("build program\n");
	ret= clBuildProgram(prg, 1, &g_cl_device, NULL,NULL,NULL);
	
	printf("compiled cl program %x\n",ret);


	cl_kernel kernel= clCreateKernel(prg, "vector add", &ret);	
	printf("created kernel %x %x",kernel,ret);
	printf("set kernel args\n");
	ret=clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&array_a);
	ret=clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&array_b);
	ret=clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&array_c);

	size_t global_item_size = testsize;
	size_t local_item_size = 64;
	printf("trigger kernel\n");
	ret=clEnqueueNDRangeKernel(gclq, kernel, 1, NULL, &global_item_size,&local_item_size, 0, NULL,NULL);
	printf("finished dispatch..");
	clEnqueueReadBuffer(gclq, array_c, CL_TRUE, 0, sizeof(int)*testsize, data_c, 0, NULL,NULL);
	printf("finished read\n");
	clFlush(gclq);
	clFinish(gclq);
	
	printf("values back from opencl device kernel invocation?:-\n");
	for (i=0; i<testsize; i++) {
		printf("[%d/%d] %d + %d = %d\n", i,testsize, data_a[i],data_b[i],data_c[i]);
	}
	printf("finish..\n");
	clReleaseKernel(kernel);
	clReleaseProgram(prg); 
	clReleaseMemObject(array_a);
	clReleaseMemObject(array_b);
	clReleaseMemObject(array_c);
	free(data_a);
	free(data_b);
	free(data_c);
}
void opencl_shutdown() {
	clReleaseCommandQueue(gclq); gclq=0;
	clReleaseContext(gcl); gcl=0;
	
}

int SCREEN_HEIGHT = 800;
int SCREEN_WIDTH = 600;
int main() {
	printf(g_kernel);
	SDL_Init(SDL_INIT_VIDEO);
	SDL_Window *window = SDL_CreateWindow("SDL Game", 0, 0, 
	SCREEN_HEIGHT, SCREEN_WIDTH, SDL_WINDOW_HIDDEN);
	SDL_ShowWindow(window);
	SDL_Event event;

	opencl_init();
	test_opencl();
	opencl_shutdown();

	int running = 1;
	while(running) {
		while(SDL_PollEvent(&event)) {
			if(event.type == SDL_QUIT) {
				running = 0;
			}
		}
		SDL_Delay( 32 );
	}
	SDL_DestroyWindow(window);
	SDL_Quit();
	return 0;
}