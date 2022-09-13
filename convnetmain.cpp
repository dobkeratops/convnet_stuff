#include <stdio.h>
#include <stdlib.h>
#include <OpenCL/opencl.h>

#include "SDL2/SDL.h"
#include "SDL_image.h"


cl_device_id g_cl_device=0;
cl_context gcl=0;
cl_command_queue gclq=0;

void cl_verify(cl_int errcode, const char*srcfile ,int line,const char* msg){
	if (errcode==0) {return;}
	const char* err="unknown error";
	#define ERRCODE(X) case X: err=(const char*)#X; break;

	switch (errcode) {
		ERRCODE(CL_INVALID_PROGRAM_EXECUTABLE)
		ERRCODE(CL_INVALID_COMMAND_QUEUE)
		ERRCODE(CL_INVALID_KERNEL)
		ERRCODE(CL_INVALID_CONTEXT)
		ERRCODE(CL_INVALID_KERNEL_ARGS)
		ERRCODE(CL_INVALID_WORK_DIMENSION)
		ERRCODE(CL_INVALID_WORK_GROUP_SIZE)
		ERRCODE(CL_INVALID_WORK_ITEM_SIZE)
		ERRCODE(CL_INVALID_GLOBAL_OFFSET)
		ERRCODE(CL_OUT_OF_RESOURCES)
		ERRCODE(CL_MEM_OBJECT_ALLOCATION_FAILURE)
		ERRCODE(CL_INVALID_EVENT_WAIT_LIST)
		ERRCODE(CL_OUT_OF_HOST_MEMORY)
		ERRCODE(CL_INVALID_VALUE)
		ERRCODE(CL_INVALID_HOST_PTR)
		ERRCODE(CL_INVALID_OPERATION)
		ERRCODE(CL_MEM_COPY_OVERLAP)
		ERRCODE(CL_INVALID_MEM_OBJECT)
	
	}
	#undef ERRCODE
	printf("%s:%d\nopencl error %x\n%s\n",srcfile,line,errcode,msg?msg:"");
}
#define CL_VERIFY(ERR) cl_verify(ERR, __FILE__, __LINE__, (const char*)0);

void opencl_init() {
	cl_uint num_devices, i;
	clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);

	auto devices = (cl_device_id*) calloc(sizeof(cl_device_id), num_devices);
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

	cl_int ret=0;
	gcl = clCreateContext(NULL,1,&g_cl_device, NULL, NULL, &ret); CL_VERIFY(ret);
	gclq = clCreateCommandQueue(gcl,g_cl_device,0,&ret); CL_VERIFY(ret);

}

#define MALLOCS(TYPE,NUM) ((TYPE*)malloc(sizeof(TYPE)*(NUM)))

// does it work?

cl_program cl_load_program(const char* prgname) {
	FILE* fp = fopen(prgname,"rb");
	if (!fp) {
		printf("can't load %s\n",prgname);
		exit(0);
	}
	cl_int ret=0;
	size_t srclen=0;
	fseek(fp,0,SEEK_END); srclen=ftell(fp); fseek(fp,0,SEEK_SET);
	const char* kernel_src = (const char*) malloc(srclen);

	fread((void*) kernel_src,1,srclen,fp);
	fclose(fp);
	printf(kernel_src);

	cl_program  prg = clCreateProgramWithSource(gcl, 1, (const char**) &kernel_src,(const size_t*)&srclen, &ret);
	CL_VERIFY(ret);
	return prg;
}

cl_mem cl_create_and_load_buffer(size_t elem_size,int num_elems,void* src_data) {
	cl_int ret;
	cl_mem buffer = clCreateBuffer(gcl,CL_MEM_READ_ONLY, elem_size*num_elems, NULL,&ret); CL_VERIFY(ret);
	if (src_data!=0) {
		ret = clEnqueueWriteBuffer(gclq, buffer, CL_TRUE, 0, elem_size*num_elems, src_data, 0, NULL,NULL);
	}
	CL_VERIFY(ret);
}




void opencl_test_vec_add() {
	cl_int ret;
	int testsize=64;
	cl_mem buffer_a =  clCreateBuffer(gcl,CL_MEM_READ_ONLY, testsize*sizeof(int), NULL, &ret); CL_VERIFY(ret);
	cl_mem buffer_b =  clCreateBuffer(gcl,CL_MEM_READ_ONLY, testsize*sizeof(int), NULL, &ret); CL_VERIFY(ret);
	cl_mem buffer_c =  clCreateBuffer(gcl,CL_MEM_WRITE_ONLY, testsize*sizeof(int), NULL, &ret); CL_VERIFY(ret);


	float* data_a=MALLOCS(float, testsize);
	float* data_b=MALLOCS(float, testsize);
	float* data_c=MALLOCS(float, testsize);
	int i;
	for (i=0; i<testsize; i++) {
		data_a[i]=(float)i;
		data_b[i]=(float)i*10000.0f;
		data_c[i]=0.0f;
	}
	
	ret=clEnqueueWriteBuffer(gclq, buffer_a, CL_TRUE,0, testsize*sizeof(float), data_a, 0, NULL,NULL);
	ret=clEnqueueWriteBuffer(gclq, buffer_b, CL_TRUE,0, testsize*sizeof(float), data_b, 0, NULL,NULL);

	size_t srclen=0;
	printf("create program\n");

	cl_program prg= cl_load_program("kernel.cl");

	printf("build program\n");
	ret= clBuildProgram(prg, 1, &g_cl_device, NULL,NULL,NULL);
	CL_VERIFY(ret);


	cl_kernel kernel= clCreateKernel(prg, "vector_add", &ret);	
	CL_VERIFY(ret);
	printf("set kernel args\n");
	int size[4]={1,2,3,4};
	ret=clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffer_a); CL_VERIFY(ret);
	ret=clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffer_b); CL_VERIFY(ret);
	ret=clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buffer_c); CL_VERIFY(ret);
	//ret=clSetKernelArg(kernel, 3, sizeof(size), (void*)&size[0]); CL_VERIFY(ret);

	size_t global_item_size = testsize;
	size_t local_item_size = 64;
	printf("trigger kernel\n");
	ret=clEnqueueNDRangeKernel(gclq, kernel, 1, NULL, &global_item_size,&local_item_size, 0, NULL,NULL);
	printf("finished dispatch..");
	clEnqueueReadBuffer(gclq, buffer_c, CL_TRUE, 0, sizeof(float)*testsize, data_c, 0, NULL,NULL);

	printf("finished read\n");
	clFlush(gclq);
	clFinish(gclq);
	
	printf("values back from opencl device kernel invocation?:-\n");
	for (i=0; i<testsize; i++) {
		printf("[%d/%d] %.3f+ %.3f = %.3f\n", i,testsize, data_a[i],data_b[i],data_c[i]);
	}
	printf("finish..\n");
	clReleaseKernel(kernel);
	clReleaseProgram(prg); 
	clReleaseMemObject(buffer_a);
	clReleaseMemObject(buffer_b);
	clReleaseMemObject(buffer_c);
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

	SDL_Init(SDL_INIT_VIDEO);
	SDL_Window *window = SDL_CreateWindow("SDL Game", 0, 0, 
		SCREEN_HEIGHT, SCREEN_WIDTH, SDL_WINDOW_HIDDEN);
	SDL_ShowWindow(window);
	SDL_Event event;

	opencl_init();
	opencl_test_vec_add();
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
