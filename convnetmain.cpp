#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
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
        ERRCODE(CL_INVALID_ARG_SIZE)
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

// todo .. decide if we should just use std::array for this..
struct Int2 {
    int x,y;Int2(){x=0;y=0;}Int2(int x,int y){this->x=x;this->y=y;}int hmul()const{return x*y;}
    template<typename T>
    operator std::array<T,2>() const {return std::array<T,2>({(T)this->x,(T)this->y});}
};
struct Int3 {
    int x,y,z;Int3(){x=0;y=0,z=0;}Int3(int x,int y,int z){this->x=x;this->y=y;this->z=z;}int hmul()const{return x*y*z;}
    template<typename T>    
    operator std::array<T,3>() const {return std::array<T,3>({(T)this->x,(T)this->y,(T)this->z});}
};
struct Int4 {
    int x,y,z,w;    Int4(){x=0;y=0;z=0;w=0;}    Int4(int x,int y,int z,int w){this->x=x;this->y=y;this->z=z;this->w=w;} int hmul()const{return x*y*z*w;}
    template<typename T>
    operator std::array<T,4>() const {return std::array<T,4>({(T)this->x,(T)this->y,(T)this->z,(T)this->w});}
};

template<typename T> T& operator<<(T& dst, const Int4& src){return dst<<"("<<src.x<<","<<src.y<<","<<src.z<<","<<src.w<<")";}

template<typename T=float>
struct Buffer {
    Int4 shape;
    T* data=nullptr;
    cl_mem device_buffer=0;
    
    Buffer(Int4 shape, std::function<T(int)> generate=[](int i){return T();}, cl_int mode=CL_MEM_READ_ONLY) {
        this->shape=shape;
        std::cout<<"creating buffer: ["<<this->shape<<"\n";
        
        this->data = new T[shape.hmul()];
        cl_int ret;
        this->device_buffer =clCreateBuffer(gcl, mode, sizeof(T)*this->total_size() , NULL, &ret); CL_VERIFY(ret);
        for (int i=0; i<this->total_size(); i++) {
            this->data[i] = generate(i);
        }
        if (mode!=CL_MEM_WRITE_ONLY){
            this->to_device();
        }
    }
    Buffer(Int4 shape, const T* src, cl_int mode=CL_MEM_READ_ONLY) : Buffer(shape, [&](int i){return src[i];},mode){
        
    }
    Buffer(Buffer<T>&& src) {
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
    void set_arg_of(cl_kernel kernel, int arg_index) const {
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

template<typename F,typename T>
F& operator<<(F& dst, const Buffer<T>& src) {
    dst<<src.shape<<"\n";
    int numy=src.shape.y*src.shape.z*src.shape.w;
    // jst print as 2d, todo..
    dst<<"[\n";
    for (int j=0; j<numy; j++) {
        if (src.shape.x<16) {
            dst<<"\t\t[";
            for (int i=0; i<src.shape.x; i++) {
                dst <<src[i+j*src.shape.x] << "\t";
            }
            dst<<"]\n";
        } else {
            dst<<"\t[\n";
            for (int i=0; i<src.shape.x; i++) {
                dst <<"\t\t"<<src[i+j*src.shape.x] << "\n";
            }
            dst<<"\t]\n";
        }
    }
    dst<<"]\n";
    return dst;
}

struct Program {
    cl_program prog=0;
    Program(){this->prog=0;}
    Program(const char* filename){
        this->prog= cl_load_program(filename);
    }
    Program(Program&& src){this->prog=src.prog;src.prog=0;}
    ~Program(){
        if (this->prog){
            clReleaseProgram(this->prog); 
            this->prog=0;
        }
    }
};

struct Kernel {
    cl_kernel kernel;
    cl_int num_args=0;
    cl_int arg_set=0;
    Kernel(const Program& prg, const char* entry) {
        assert(prg.prog);
        cl_int ret;
        this->kernel = clCreateKernel(prg.prog, entry, &ret);	CL_VERIFY(ret);
        size_t sz;
        ret=clGetKernelInfo(this->kernel,CL_KERNEL_NUM_ARGS,sizeof(this->num_args),(void*)&this->num_args,&sz);
        
    }

    Kernel(const Kernel&) = delete;
    Kernel(Kernel&& src){this->kernel  =src.kernel;src.kernel=0;}
    ~Kernel(){
        if (this->kernel){
            clReleaseKernel(this->kernel);
        }
    }
    void verify_args()const{
        assert(this->arg_set == (1<<this->num_args)-1);
    }
    void enqueue_range(size_t globalsize,size_t localsize) {
        verify_args();
        auto ret=clEnqueueNDRangeKernel(gclq, this->kernel, 1, NULL, &globalsize,&localsize, 0, NULL,NULL); CL_VERIFY(ret);
    }
    void enqueue_range_2d(Int2 _globalsize,Int2 _localsize) {
        auto globalsize=(std::array<size_t,2>)_globalsize;
        auto localsize=(std::array<size_t,2>)_localsize;
        verify_args();
        auto ret=clEnqueueNDRangeKernel(gclq, this->kernel, 2, NULL, &globalsize[0],&localsize[0], 0, NULL,NULL); CL_VERIFY(ret);
    }

    // setting a buffer is specialization.. ths looks horrid after rust.
    template<>
    template<typename T>
    void set_arg(int arg_index, Buffer<T>& x){
        x.set_arg_of(this->kernel, arg_index);
        assert(arg_index<this->num_args);
        this->arg_set|=1<<arg_index;
    }
    template<typename T>
    void set_arg(int arg_index, const T& val){  
        assert(arg_index<this->num_args);
        auto ret=clSetKernelArg(this->kernel, arg_index, (size_t) sizeof(T), (const void*)&val); CL_VERIFY(ret);
        this->arg_set|=1<<arg_index;
    }

    template<typename A,typename B>
    void set_args(Buffer<A>& a, Buffer<B>& b){
        this->set_arg(0,a);this->set_arg(1,b);
    }
    template<typename A,typename B,typename C>
    void set_args(Buffer<A>& a, Buffer<B>& b, Buffer<C>& c){
        this->set_arg(0,a);this->set_arg(1,b);this->set_arg(2,c);
    }
    // todo -figure out variadic template mixing Buffer<X>& and const Y&
    template<typename A,typename B,typename C,typename... D>
    void set_args(Buffer<A>& a, Buffer<B>& b, Buffer<C>& c,D... rest){
        this->set_arg(0,a);this->set_arg(1,b);this->set_arg(2,c);
        this->set_args_at(3,rest...);
    }
    template<typename X>
    void set_args_at(int index,const X& x){
        this->set_arg(index,x);
    }
    template<typename X,typename... Rest>
    void set_args_at(int index,const X& x,Rest... rest){
        this->set_arg(index,x);
        this->set_args_at(index+1,rest...);
    }


};

void opencl_test_conv() {
	cl_int ret;
	int testsize=64;
    auto size=Int4(testsize,1,1,1);
    auto buffer_a = Buffer<float>(size,[&](auto i){return (float)i;});
    auto buffer_b = Buffer<float>(size,[&](auto i){return (float)(testsize-i);}); 
    auto buffer_c = Buffer<float>(size,[&](auto i){return 0.0f;}, CL_MEM_READ_WRITE);
	
    Program prg("kernel.cl");
	auto kernel=Kernel(prg,"vector_add_scaled");
    kernel.set_args(buffer_a,buffer_b,buffer_c, 1000.0f, 1.0f);

    kernel.enqueue_range(testsize,64);
	
    buffer_c.from_device();

	clFlush(gclq);
	clFinish(gclq);

	printf("values back from opencl device kernel invocation?:-\n");
    std::cout<< buffer_c;	
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
