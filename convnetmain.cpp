#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <array>
#include <iostream>
#include <OpenCL/opencl.h>

#include "SDL2/SDL.h"
#include "SDL_image.h"


cl_device_id g_cl_device=0;
cl_context gcl=0;
cl_command_queue gclq=0;

#define TRACE printf("%s:%d\n",__FILE__,__LINE__);

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
	printf("%s",kernel_src);

    
	cl_program  prg = clCreateProgramWithSource(gcl, 1, (const char**) &kernel_src,(const size_t*)&srclen, &ret);
    
	CL_VERIFY(ret);

   	ret= clBuildProgram(prg, 1, &g_cl_device, NULL,NULL,NULL);
    
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
    return buffer;
}


struct Int4 {
    int x,y,z,w;
    Int4(){x=0;y=0;z=0;w=0;}
    Int4(int x,int y,int z,int w){this->x=x;this->y=y;this->z=z;this->w=w;}
    int hmul()const{return x*y*z*w;}
};
template<typename T> T& operator<<(T& dst, const Int4& src){return dst<<"("<<src.x<<","<<src.y<<","<<src.z<<","<<src.w<<")";}

template<typename T=float>
struct Buffer {
    Int4 shape;
    T* data=nullptr;
    cl_mem device_buffer=0;
    Buffer(Int4 shape, const T* src=nullptr, cl_int mode=CL_MEM_READ_ONLY) {
        this->shape=shape;
        std::cout<<"creating buffer: ["<<this->shape<<"\n";
        
        this->data = new T[shape.hmul()];
        cl_int ret;
        
        this->device_buffer =clCreateBuffer(gcl, mode, sizeof(T)*this->total_size() , NULL, &ret); CL_VERIFY(ret);
        if (src!=nullptr){
            for (int i=0; i<this->total_size(); i++) {
                this->data[i] = src[i];
            }
        }
        if (src) {
            this->to_device();
        }

    }
    Buffer(Buffer&& src) {
        this->shape =src.shape;
        src.shape=Int4();
        this->data=src.data;
        src.data=nullptr;
        this->device_buffer = src.device_buffer;
        src.device_buffer=0;
    }
    ~Buffer() {
        if (this->data){delete[] this->data;}
        if (this->device_buffer) {
            clReleaseMemObject(this->device_buffer);
        }
    }
    void set_arg_of(cl_kernel kernel, int arg_index) {
        auto ret=clSetKernelArg(kernel, arg_index, sizeof(cl_mem), (void*)&this->device_buffer); CL_VERIFY(ret);
    }
    
    int total_size() const{return shape.hmul();}
    void to_device() {
        auto ret=clEnqueueWriteBuffer(gclq, this->device_buffer, CL_TRUE,0, this->total_size()*sizeof(T), (void*) this->data, 0, NULL,NULL); CL_VERIFY(ret);
    }
    void from_device() {
        auto ret=clEnqueueReadBuffer(gclq, this->device_buffer, CL_TRUE, 0, sizeof(T)*this->total_size(), this->data, 0, NULL,NULL);
        CL_VERIFY(ret);
    }
    const T& operator[](int i) const{return this->data[i];}
    T& operator[](int i){return this->data[i];}
};

struct Kernel {
    cl_kernel kernel;
    Kernel(cl_program prg, const char* entry) {
        cl_int ret;
        this->kernel = clCreateKernel(prg, entry, &ret);	CL_VERIFY(ret);
    }
    Kernel(kernel) = delete;
    Kernel(Kernel&& src){this->kernel  =src.kernel;src.kernel=0;}
    ~Kernel(){
        if (this->kernel){
            clReleaseKernel(this->kernel);
        }
    }
};

void opencl_test_conv() {
	cl_int ret;
	int testsize=64;
    auto size=Int4(testsize,1,1,1);
    auto buffer_a = Buffer(size);
    auto buffer_b = Buffer(size); 
    auto buffer_c = Buffer(size, (float*)nullptr, CL_MEM_READ_WRITE);
	for (int i=0; i<testsize; i++) {
		buffer_a[i]=(float)i+(float)i*100.0;
		buffer_b[i]=(float)i*10000.0f;
		buffer_c[i]=0.0f;
	}
    buffer_a.to_device();
    buffer_b.to_device();
    
	cl_program prg= cl_load_program("kernel.cl");
	//printf("build program\n");
	//ret= clBuildProgram(prg, 1, &g_cl_device, NULL,NULL,NULL);
	//CL_VERIFY(ret);

	auto kernel=Kernel(prg,"vector_add");
	 
	printf("set kernel args\n");

    buffer_a.set_arg_of(kernel,0);
    buffer_b.set_arg_of(kernel,1);
    buffer_c.set_arg_of(kernel,2);

	size_t global_item_size = testsize;
	size_t local_item_size = 64;
	printf("trigger kernel\n");
	ret=clEnqueueNDRangeKernel(gclq, kernel, 1, NULL, &global_item_size,&local_item_size, 0, NULL,NULL);
	printf("finished dispatch..");
	//clEnqueueReadBuffer(gclq, buffer_c, CL_TRUE, 0, sizeof(float)*testsize, &data_c[0], 0, NULL,NULL);
    buffer_c.from_device();

	printf("finished read\n");
	clFlush(gclq);
	clFinish(gclq);
	
	printf("values back from opencl device kernel invocation?:-\n");
	for (int i=0; i<testsize; i++) {
		printf("[%d/%d] %.3f+ %.3f = %.3f\n", i,testsize, buffer_a[i],buffer_b[i],buffer_c[i]);
	}
	printf("finish..\n");
	
	clReleaseProgram(prg); 
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
	opencl_test_conv();
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
