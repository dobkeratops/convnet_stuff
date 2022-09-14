#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <map>
#include <sys/stat.h>
#include <vector>
#include <array>
#include <iostream>
#include <memory>
#include <OpenCL/opencl.h>

#include "SDL2/SDL.h"
//#include "SDL_image.h"


cl_device_id g_cl_device=0;
cl_context gcl=0;
cl_command_queue gclq=0;
#ifndef TRACE
#define TRACE printf("%s:%d %s()\n",__FILE__,__LINE__,__FUNCTION__);
#endif

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
#ifndef CL_VERIFY
#define CL_VERIFY(ERR) cl_verify(ERR, __FILE__, __LINE__, (const char*)0);
#endif

void opencl_init() {
    TRACE
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


// does it work?

cl_program cl_load_program(const char* prgname) {
    struct stat s; stat(prgname,&s); 
	FILE* fp = fopen(prgname,"rb");
	if (!fp) {
		printf("can't load %s\n",prgname);
		exit(0);
	}
	cl_int ret=0;
	
    const char* kernel_src=(const char*) malloc(s.st_size);

	fread((void*) kernel_src,1,s.st_size,fp);
	fclose(fp);
    printf("loaded %s %d bytes\n",prgname,(int)s.st_size);

	cl_program  prg = clCreateProgramWithSource(gcl, 1, (const char**) &kernel_src,(const size_t*)&s.st_size, &ret);
    free((void*)kernel_src);
    
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
    bool operator==(const Int2& other)const{return x==other.x&&y==other.y;}
    template<typename T>
    operator std::array<T,2>() const {return std::array<T,2>({(T)this->x,(T)this->y});}
};
struct Int3 {
    int x,y,z;Int3(){x=0;y=0,z=0;}Int3(int x,int y,int z){this->x=x;this->y=y;this->z=z;}int hmul()const{return x*y*z;}
    bool operator==(const Int3& other)const{return x==other.x&&y==other.y&&z==other.z;}
    template<typename T>    
    operator std::array<T,3>() const {return std::array<T,3>({(T)this->x,(T)this->y,(T)this->z});}
    Int2 xy()const{return Int2(x,y);}
};
struct Int4 {
    int x,y,z,w;    Int4(){x=0;y=0;z=0;w=0;}    Int4(int x,int y,int z,int w){this->x=x;this->y=y;this->z=z;this->w=w;} int hmul()const{return x*y*z*w;}
    bool operator==(const Int4& other)const{return x==other.x&&y==other.y&&z==other.z&&w==other.w;}
    template<typename T>
    operator std::array<T,4>() const {return std::array<T,4>({(T)this->x,(T)this->y,(T)this->z,(T)this->w});}
    Int2 xy()const{return Int2(x,y);}
    Int3 xyz()const{return Int3(x,y,z);}
};

template<typename T> T& operator<<(T& dst, const Int2& src){return dst<<"("<<src.x<<","<<src.y<<")";}
template<typename T> T& operator<<(T& dst, const Int4& src){return dst<<"("<<src.x<<","<<src.y<<","<<src.z<<","<<src.w<<")";}

template<typename T=float, const int INTERLEAVEZ=1>
struct Buffer {
    Int4 shape=Int4(0,0,0,0);
    T* data=nullptr;
    cl_mem device_buffer=0;
    void set_size(Int4 shape, std::function<T(int)> generate = [](int x){return T();}, cl_int mode = CL_MEM_READ_WRITE) {
        assert(data==nullptr);
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
    Buffer() {}
    
    Buffer(Int4 shape, std::function<T(int)> generate=[](int i){return T();}, cl_int mode=CL_MEM_READ_ONLY) {
        this->set_size(shape, generate,mode);
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
    // TODO distinguish debug print
    dst<<src.shape<<"\n";
    int numz=src.shape.z*src.shape.w;
    // jst print as 2d, todo..
    dst<<"[\n";
    for (int k=0; k<numz; k++) {
        dst<<"\t[\n";
        for (int j=0; j<src.shape.y; j++) {
            int num_to_show=src.shape.x<16?src.shape.x:16;
            
            dst<<"\t\t[";
            for (int i=0; i<num_to_show; i++) {
                dst <<src[i+j*src.shape.x] << "\t";
            }
            if (src.shape.x>num_to_show){dst<<"...";}
            dst<<"\t]\n";
        }
        dst<<"\t]\n";
    }
    dst<<"]\n";
    return dst;
}

struct Program {
    std::string progname;
    cl_program prog=0;
    Program(){this->prog=0;}
    Program(const char* filename){
        this->progname=filename;
        this->prog= cl_load_program(filename);
    }
    Program(Program&& src){this->prog=src.prog;src.prog=0;}
    ~Program(){
        std::cout<<"releasing program " <<this->progname<<"\n";
        if (this->prog){
            clReleaseProgram(this->prog); 
            this->prog=0;
        }
    }
};

struct Kernel {
    std::shared_ptr<Program> program;
    std::string name;
    cl_kernel kernel=0;
    cl_int num_args=0;
    cl_int arg_set=0;
    Kernel(){}
    Kernel(std::shared_ptr<Program>  prg, const char* entry) {
        name=entry;
        assert(prg->prog);
        this->program=prg;
        cl_int ret;
        this->kernel = clCreateKernel(prg->prog, entry, &ret);	CL_VERIFY(ret);
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
    //template<>
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

void opencl_test_basic() {
    TRACE
	cl_int ret;
	int testsize=64;
    auto size=Int4(testsize,1,1,1);
    auto buffer_a = Buffer<float>(size,[&](auto i){return (float)i;});
    auto buffer_b = Buffer<float>(size,[&](auto i){return (float)(testsize-i);}); 
    auto buffer_c = Buffer<float>(size,[&](auto i){return 0.0f;}, CL_MEM_READ_WRITE);
	
    auto prg = std::make_shared<Program>("kernel.cl");
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


typedef int nodeid;

struct Node;
struct NeuralNet {
    friend Node;
    std::vector<Node*> nodes;
    std::shared_ptr<Program> prg = std::make_shared<Program>("kernel.cl");
    std::map<std::string,std::shared_ptr<Kernel>> used_kernels;
    
    void push_node(Node* n);
    ~NeuralNet() noexcept;
    void dump();
    std::shared_ptr<Kernel> get_kernel(const char* entrypt) {
        auto strname=std::string(entrypt);
        if (used_kernels.contains(strname)) {
            return used_kernels[strname];
        } else {
            std::shared_ptr<Kernel> ret=std::make_shared<Kernel>(this->prg,entrypt);
            used_kernels.insert(std::make_pair(strname, ret));
            return ret;
        }
    }
};

struct Node {
    friend NeuralNet;
    NeuralNet* net;
    Buffer<float> activations;
    std::shared_ptr<Kernel> kernel;
    // todo: smallvector, node inptu counts are 0,1,2
    const char* kernel_name() const{return kernel?kernel->name.c_str():"";}
    void dump_base() {
        
        auto shape=this->activations.shape;
        printf("node [%d]\ttype=%s\tshape=[%d %d %d %d]\t%s()\n",this->index,this->name(), shape.x,shape.y,shape.z,shape.w, this->kernel_name());
        if (this->inputs.size()>0){
            if (this->inputs.size()==1) {printf("input=node[%d]\n",this->inputs[0]);}
            else {
                printf("inputs=[");
                for (int i=0; i<this->inputs.size(); i++){
                    printf("%d ",this->inputs[i]);
                }
                printf("]\n");
            }
        }
        
    }
    int channels() const{return activations.shape.z;}
    int width() const {return activations.shape.x;}
    int height() const {return activations.shape.y;}

protected:
    nodeid index;
    std::vector<nodeid> inputs;
    void set_size(Int3 size){
        activations.set_size(Int4(size.x,size.y,size.z,1));
    }
    virtual const char* name() const=0;
    Node(NeuralNet* _net, const char* kernel_entrypt) {
        this->net = _net;
        this->index=(nodeid)_net->nodes.size();
        this->net->nodes.push_back(this);
        if (kernel_entrypt) {
            
            this->kernel = net->get_kernel(kernel_entrypt);
        }
    }
    Node(NeuralNet* net, const char* entrypt,nodeid _input) : Node(net,entrypt){inputs.resize(1);inputs[0] = _input<0?this->index+_input:_input;}    
    Node(NeuralNet* net,const char* entrypt,nodeid src0,nodeid src1) : Node(net,entrypt){inputs.resize(2);inputs[0] = src0<0?this->index+src0:src0;inputs[1] = src1<0?this->index+src1:src1;}
    Node* input_node(int i)const{return net->nodes[this->inputs[i]];}
    virtual ~Node();
    virtual void dump(){};
};

NeuralNet::~NeuralNet() noexcept{
    for (auto x : this->nodes) {
        assert(x->net==this);
        delete x;
    }
}

void NeuralNet::dump() {
    for (auto n :nodes) {
        n->dump_base();
        n->dump();
    }
}
Node::~Node() {printf("destructing node %d\n",this->index);}
class Conv2d : public Node{
    Buffer<float> filter;
    const char* name() const override{return "Conv2d";};

    void dump() override {
        printf("filter_shape=[%d %d %d %d]\n",filter.shape.x,filter.shape.y,filter.shape.z,filter.shape.w);
    }
public:    
    Conv2d(NeuralNet* owner, int _input, Int2 _size, int _channels_out, int stride=1) :Node(owner,"conv2d",_input){
        auto inp=input_node(0);
        int input_channels=inp->channels();
        this->set_size( Int3(inp->width()/stride,inp->height()/stride, _channels_out) );
        filter.set_size(Int4(_size.x,_size.y, input_channels,_channels_out));
    }
};
class Concat : public Node {
    const char* name() const override{return "Concat";}
public:
    Concat(NeuralNet* owner, int src0, int src1) :Node(owner,"concat",src0,src1) {
        auto in0=this->input_node(0),in1=this->input_node(1);
        assert(in0->width()==in1->width() && in0->height()==in1->height());
        this->set_size(Int3(in0->width(),in0->height(), in0->channels()+in1->channels()));
    }
};

class Add : public Node {
    const char* name() const override{return "Concat";}
public:
    Add(NeuralNet* owner, int src0, int src1) :Node(owner,"vector_add",src0,src1) {
        auto in0=this->input_node(0),in1=this->input_node(1);
        assert(in0->activations.shape==in1->activations.shape);
        this->set_size(Int3(in0->width(),in0->height(), in0->channels()));
    }
};

class AvPool2x2 : public Node {
    const char* name() const override{return "AvPool2x2";}
public:
    AvPool2x2(NeuralNet* owner,int _input=-1) : Node(owner,"avpool2x2",_input){
        auto inp=input_node(0);
        this->set_size( Int3(inp->width()/2,inp->height()/2,inp->channels()));
    }
};
class MaxPool2x2 : public Node {
    const char* name() const override{return "MaxPool2x2";}
public:
    MaxPool2x2(NeuralNet* owner,int _input=-1) : Node(owner,"maxpool2x2",_input){
        auto inp=input_node(0);
        this->set_size( Int3(inp->width()/2,inp->height()/2,inp->channels()));
    }
};

class Expand2x2 : public Node {
    const char* name() const override{return "AvPool2x2";}
public:
    Expand2x2(NeuralNet* owner,int _input=-1) : Node(owner,"expand2x2",_input){
        auto inp=input_node(0);
        this->set_size( Int3(inp->width()*2,inp->height()*2,inp->channels()));
    }
};
class FlattenToZ : public Node {
    const char* name() const override{return "FlattenZ";}
public:
    FlattenToZ(NeuralNet* owner,int _input=-1) : Node(owner,"flatten_to_z",_input){
        auto inp=input_node(0);
        this->set_size( Int3(1,1, inp->width()*inp->height()*inp->channels(),1) )
    }
}

class InputImage : Node{
    const char* name() const override{return "InputImage";};
    void dump() override {}
public:
    InputImage(NeuralNet* net, Int3 _size) : Node(net,nullptr) {
        this->set_size(_size);
    }
};

void test_setup_convnet() {
    TRACE
    NeuralNet net;
    new InputImage(&net, Int3(256,256,3));
    new Conv2d(&net,-1 , Int2(3,3), 16, 1);
    new AvPool2x2(&net);
    new Conv2d(&net,-1 , Int2(3,3), 32, 1);
    new AvPool2x2(&net);
    new Conv2d(&net,-1 , Int2(3,3), 64, 1);
    new Conv2d(&net,-1 , Int2(3,3), 64, 1);
    new Concat(&net,-1,-2);
    new AvPool2x2(&net);

    net.dump();
}


int SCREEN_HEIGHT = 800;
int SCREEN_WIDTH = 600;
void run_window_main_loop() {
    TRACE
	SDL_Event event;
	SDL_Init(SDL_INIT_VIDEO);
	SDL_Window *window = SDL_CreateWindow("SDL Game", 0, 0, 
		SCREEN_HEIGHT, SCREEN_WIDTH, SDL_WINDOW_HIDDEN);
	SDL_ShowWindow(window);
    SDL_Renderer* rs=SDL_CreateRenderer(window,-1,0);
    
    int frame=0;
	int running = 1;
	while(running) {
		while(SDL_PollEvent(&event)) {
			if(event.type == SDL_QUIT) {
				running = 0;
			}
		}
        frame+=1;
        SDL_SetRenderDrawColor(rs, frame&255, frame&1?128:0, 0, 255);

        SDL_RenderClear(rs);
        
        SDL_RenderPresent(rs);

	}
	SDL_DestroyWindow(window);
	SDL_Quit();
}

int main() {


	opencl_init();
	opencl_test_basic();
    test_setup_convnet();
    run_window_main_loop();
	opencl_shutdown();

	return 0;
}
