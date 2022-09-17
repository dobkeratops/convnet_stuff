#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <map>
#include <time.h>
#include <sys/stat.h>
#include <vector>
#include <array>
#include <iostream>
#include <memory>
#include <OpenCL/opencl.h>

#include "SDL2/SDL.h"
#include "SDL_image.h"


cl_device_id g_cl_device=0;
cl_context gcl=0;
cl_command_queue gclq=0;
#ifndef TRACE
#define TRACE printf("%s:%d %s()\n",__FILE__,__LINE__,__FUNCTION__);
#endif

void cl_verify(cl_int errcode, const char*srcfile ,int line,const char* msg){
	if (errcode==0) {return;}
	const char* err="unknown error";
	#define ERRCODE(X) if (errcode==X) {err=(const char*)#X;}


    ERRCODE(CL_INVALID_PROGRAM_EXECUTABLE)
    ERRCODE(CL_INVALID_COMMAND_QUEUE)
    ERRCODE(CL_INVALID_KERNEL)
    ERRCODE(CL_INVALID_CONTEXT)
    ERRCODE(CL_INVALID_KERNEL_ARGS)
    ERRCODE(CL_INVALID_KERNEL_NAME)
    ERRCODE(CL_INVALID_KERNEL_DEFINITION)
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

	
	#undef ERRCODE
	printf("%s:%d\nopencl error %d\t%s\t%s\n",srcfile,line,errcode,err,msg?msg:"");
}
#ifndef CL_VERIFY
#define CL_VERIFY(ERR) cl_verify(ERR, __FILE__, __LINE__, (const char*)0);
#endif

struct ClDeviceInfo {
    char extensions[2040];
    char device_name[512]; //CL_DEVICE_NAME
    char device_type[512];
    cl_uint local_mem_size;
    cl_ulong global_mem_cache_size;
    size_t max_workgroup_size;//CL_DEVICE_MAX_WORK_GROUP_SIZE
    size_t max_work_item_sizes[3];
};
ClDeviceInfo gDeviceInfo;
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

     
    size_t s;    
    #define CL_GET_INFO(id,ITEM, str) clGetDeviceInfo(g_cl_device,id, sizeof(gDeviceInfo.ITEM),(void*)&gDeviceInfo.ITEM,&s); printf("\t%s=" str "\n", #ITEM, gDeviceInfo.ITEM);
    CL_GET_INFO(CL_DEVICE_EXTENSIONS, extensions, "%s");
    CL_GET_INFO(CL_DEVICE_NAME, device_name, "%s");
    CL_GET_INFO(CL_DEVICE_TYPE,device_type, "%s");
    CL_GET_INFO(CL_DEVICE_LOCAL_MEM_SIZE,local_mem_size, "%lu8");
    
   
    CL_GET_INFO(CL_DEVICE_MAX_WORK_GROUP_SIZE,max_workgroup_size,"%lu");
    CL_GET_INFO(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,global_mem_cache_size,"%lu");
    #undef CL_GET_INFO

    {
        int val; size_t rets=0;
        
        clGetDeviceInfo(g_cl_device, CL_DEVICE_HALF_FP_CONFIG, sizeof(val),(void*)&val,&rets);   
        printf("half float=%d %lu\n",val,rets);

        int half;
    }

    clGetDeviceInfo(g_cl_device,CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(gDeviceInfo.max_work_item_sizes),(void*)&gDeviceInfo.max_work_item_sizes,&s);
    for (int i=0; i<3; i++) {printf("max item dim[%d]=%lul\n", i,gDeviceInfo.max_work_item_sizes[i]);}

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
template<typename T>
struct Vec2 {
    T x,y;
    Vec2(){x=0;y=0;}
    Vec2(T x,T y){this->x=x;this->y=y;}
    int hmul()const{return x*y;}
    bool operator==(const Vec2& other)const{return x==other.x&&y==other.y;}
    template<typename B>
    operator std::array<B,2>() const {return std::array<B,2>({(B)this->x,(B)this->y});}
};
template<typename T>
struct Vec3 {
    T x,y,z;
    Vec3(){x=0;y=0,z=0;}
    Vec3(T x,T y,T z){this->x=x;this->y=y;this->z=z;}
    T hmul()const{return x*y*z;}
    bool operator==(const Vec3& other)const{return x==other.x&&y==other.y&&z==other.z;}
    template<typename B>    
    operator std::array<B,3>() const {return std::array<B,3>({(B)this->x,(B)this->y,(B)this->z});}
    auto operator/(const Vec3& src)const {return Vec3(x/src.x,y/src.y,z/src.z);}
    auto operator*(const Vec3& src)const {return Vec3(x*src.x,y*src.y,z*src.z);}
    auto operator*(const T& src)const {return Vec3(x*src,y*src,z*src);}
    auto operator/(const T& src)const {return Vec3(x/src,y/src,z/src);}
    auto operator+(const Vec3& src)const {return Vec3(x+src.x,y+src.y,z+src.z);}
    auto operator-(const Vec3& src)const {return Vec3(x-src.x,y-src.y,z-src.z);}
    auto xy()const{return Vec2(x,y);}
    auto min(const Vec3& src)const{
        return VEc3(std::min(x,src.x),std::min(y,src.y),std::min(z,src.z));
    }
};

template<typename T>
struct Vec4 {
    T x,y,z,w;
    Vec4(){x=0;y=0;z=0;w=0;}   
    Vec4(T x,T y,T z,T w){this->x=x;this->y=y;this->z=z;this->w=w;}
    T hmul()const{return x*y*z*w;}
    bool operator==(const Vec4& other)const{return x==other.x&&y==other.y&&z==other.z&&w==other.w;}
    template<typename B>
    operator std::array<B,4>() const {return std::array<B,4>({(B)this->x,(B)this->y,(B)this->z,(B)this->w});}
    auto operator/(const Vec4& src)const {return Vec4(x/src.x,y/src.y,z/src.z,w/src.w);}
    auto operator*(const Vec4& src)const {return Vec4(x*src.x,y*src.y,z*src.z,w*src.w);}
    auto operator*(const T& src)const {return Vec4(x*src,y*src,z*src,w*src);}
    auto operator/(const T& src)const {return Vec4(x/src,y/src,z/src,w/src);}
    auto operator+(const Vec4& src)const {return Vec4(x+src.x,y+src.y,z+src.z,w+src.w);}
    auto operator-(const Vec4& src)const {return Vec4(x-src.x,y-src.y,z-src.z,w-src.w);}

    auto xy()const{return Vec2(x,y);}
    auto xyz()const{return Vec3(x,y,z);}
};
typedef Vec2<int32_t> Int2;
typedef Vec3<int32_t> Int3;
typedef Vec4<int32_t> Int4;

template<typename T> T& operator<<(T& dst, const Int2& src){return dst<<"["<<src.x<<","<<src.y<<"]";}
template<typename T> T& operator<<(T& dst, const Int3& src){return dst<<"["<<src.x<<","<<src.y<<","<<src.z<<"]";}
template<typename T> T& operator<<(T& dst, const Int4& src){return dst<<"["<<src.x<<","<<src.y<<","<<src.z<<","<<src.w<<"]";}

float frands(){ int x=rand();return  (1.0/(float)0x8000)*(float)((x&0xffff)-0x8000);}
template<typename T=float> 
struct Buffer {
    // TODO: try INTERLEAVEZ=4 for unrolling in kernels?
    // layout [z&3][x][y][z/4][w]

    Int4 shape=Int4(0,0,0,0);
    Int4 padding=Int4(0,0,0,0);   // so our filters can overstep.

    std::vector<T> data;
    cl_mem device_buffer=0;
    void generate_with(std::function<T(Int4)> genf) {
        for (int l=0; l<this->shape.w; l++) {
            for (int k=0; k<this->shape.z; k++) {
                for (int j=0; j<this->shape.y; j++) {
                    for (int i=0; i<this->shape.x; i++) {
                        auto pos=Int4(i,j,k,l);
                        this->operator[](pos) = genf(pos);
                    }
                }
            }
        }
    }
    void set_size(Int4 shape, std::function<T(Int4)> generate_f = [](Int4 pos){return T();}, cl_int mode = CL_MEM_READ_WRITE) {
        assert(data.size()==0 && "resize not supported yet");
        this->shape=shape;
        std::cout<<"creating buffer: ["<<this->shape<<"\n";
        
        this->data.resize(total_elems_padded());
        cl_int ret;
        this->device_buffer =clCreateBuffer(gcl, mode, this->total_bytes() , NULL, &ret); CL_VERIFY(ret);
        this->generate_with(generate_f);
        if (mode!=CL_MEM_WRITE_ONLY){
            this->to_device();
        }
    }
    // image array interface. x,y = indexes width,height. 'z' is used for channels eg r,g,b  'w'=slices,layers.
    // even when we move to interleaved channels, we will keep these index names.
    template<int D> std::array<T,D> get_pixel(int x,int y, int layer=0) const{
        assert(this->shape.z>=D);
        assert(x<this->shape.x);
        assert(y<this->shape.y);
        std::array<T,D> ret;
        for (int c=0; c<D; c++){
            auto i= flatten_index(Int4(x,y,c,layer));
            if (i>=0 && i<this->data.size()){   
                ret[c] =  this->data[i];
            }
            
        }
        return ret;
    }
    template<int D> void set_pixel(int x,int y, int layer, const std::array<T,D>& src) {
        assert(this->shape.z==D);
        for (int c=0; c<D; c++){
            auto i= flatten_index(this->shape, Int4(x,y,c,layer));
            this->data[i]=src[c];
            
        }
    }
    inline size_t flatten_index(const Int4& pos) const {
        auto p=pos+this->padding;
        auto shapepadded =shape + this->padding*2;
        return p.z + shapepadded.z*(p.x +shapepadded.x*(p.y+ shapepadded.y*p.w));
    }

    void init_random(Int4 shape){this->set_size(shape, [](Int4 pos){return frands();});}
    Buffer() {}
    
    Buffer(Int4 shape, std::function<T(Int4)> generate_f=[](Int4){return T();}, cl_int mode=CL_MEM_READ_WRITE) {
        this->set_size(shape, generate_f,mode);
    }
    Buffer(Int4 shape, const T* src, cl_int mode=CL_MEM_READ_WRITE) 
        : Buffer(shape, 
            [&](Int4 pos){
                return src[this->flatten_index(shape,pos)];},mode)
    {    
    }
    Buffer(Buffer<T>&& src) {
        this->shape =src.shape;
        src.shape=Int4();
        this->data=std::move(src.data);
        this->device_buffer = src.device_buffer;
        src.device_buffer=0;
    }
    ~Buffer() {
        if (this->device_buffer) {
            clReleaseMemObject(this->device_buffer);
        }
    }
    
    size_t total_elems_padded() const{return (shape+padding*2).hmul();}
    size_t total_bytes() const{return total_elems_padded()*sizeof(T);}
    void to_device() {
        auto ret=clEnqueueWriteBuffer(gclq, this->device_buffer, CL_TRUE,0, this->total_bytes(), (void*) &this->data[0], 0, NULL,NULL); CL_VERIFY(ret);
    }
    void from_device() {
        auto ret=clEnqueueReadBuffer(gclq, this->device_buffer, CL_TRUE, 0, this->total_bytes(), &this->data[0], 0, NULL,NULL);
        CL_VERIFY(ret);
    }
    // both linear and 4d indices
    //const T& operator[](int i) const{return this->data[i];}
    //T& operator[](int i){return this->data[i];}
    T& operator[](Int4 pos){return this->data[this->flatten_index(pos)];}
    const T& operator[](Int4 pos) const{return this->data[this->flatten_index(pos)];}
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
                dst <<src[Int4(i,j,k,0)] << "\t";
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
        this->kernel = clCreateKernel(prg->prog, entry, &ret);	cl_verify(ret,__FILE__,__LINE__,entry);
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
    void enqueue_range_3d(Int3 _globalsize,Int3 _localsize) {
        auto globalsize=(std::array<size_t,3>)_globalsize;
        auto localsize=(std::array<size_t,3>)_localsize;
        verify_args();
        auto ret=clEnqueueNDRangeKernel(gclq, this->kernel, 3, NULL, &globalsize[0],&localsize[0], 0, NULL,NULL); CL_VERIFY(ret);
        if (ret ==CL_INVALID_WORK_GROUP_SIZE) {
            std::cout<<_globalsize <<" "<<_localsize<<"\n";
            exit(0);
        }
    }

    template<typename T>
    void set_arg_buffer_shape(int arg_index, Buffer<T>& src){
        this->set_arg(arg_index,src);
        this->set_arg(arg_index+1,src.shape);
        
        assert(arg_index+1<this->num_args);
        this->arg_set|=3<<arg_index;
    }

    // setting a buffer is specialization.. ths looks horrid after rust.
    //template<>
    template<typename T>
    auto set_arg(int arg_index, const Buffer<T>& x)->decltype(*this)&{
        if (arg_index>=this->num_args) {
            std::cout<<arg_index<<" "<<this->num_args<<"\n";
            assert(arg_index<this->num_args);
        }
        auto ret=clSetKernelArg(this->kernel, arg_index, sizeof(cl_mem), (void*)&x.device_buffer); CL_VERIFY(ret);
        this->arg_set|=1<<arg_index;
        return *this;
    }
    template<typename T>
    auto set_arg(int arg_index, const T& val)->decltype(*this){  
        assert(arg_index<this->num_args);
        auto ret=clSetKernelArg(this->kernel, arg_index, (size_t) sizeof(T), (const void*)&val); CL_VERIFY(ret);
        this->arg_set|=1<<arg_index;
        return *this;
    }
};

void opencl_test_basic() {
    TRACE
	cl_int ret;
	int testsize=64;
    auto size=Int4(testsize,1,1,1);
    auto buffer_a = Buffer<float>(size,[&](Int4 pos){return (float)pos.x;});
    auto buffer_b = Buffer<float>(size,[&](Int4 pos){return (float)(testsize-pos.x);}); 
    auto buffer_c = Buffer<float>(size,[&](Int4 pos){return 0.0f;});
	
    auto prg = std::make_shared<Program>("kernel.cl");
	auto kernel=Kernel(prg,"vector_add_scaled");
    kernel.set_arg(0,buffer_c).set_arg(1,buffer_a).set_arg(2,buffer_b).set_arg(3,1000.0f).set_arg(4,1.0f);

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


struct NeuralNet {
    struct Node;
    friend Node;
    struct Cost {size_t fmadds=0;int parameters=0; int activations=0;};
    std::vector<Node*> nodes;
    // todo propper singleton or whatever, global management
    std::shared_ptr<Program> prg = std::make_shared<Program>("kernel.cl");
    std::map<std::string,std::shared_ptr<Kernel>> used_kernels;
    Node* last_node(){assert(nodes.size()>0);return nodes[nodes.size()-1];}
    Node* first_node(){assert(nodes.size()>0);return nodes[0];}
    void push_node(Node* n);
    ~NeuralNet() noexcept;
    void dump();
    Cost estimate_cost() const;
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
    void eval();
};

struct NeuralNet::Node {
    friend NeuralNet;
    NeuralNet* net=nullptr;
    Buffer<float> activations;
    std::shared_ptr<Kernel> kernel;
    Int3 output_dilation=Int3(1,1,1);
    
    // todo: smallvector, node inptu counts are 0,1,2
    const char* kernel_name() const{return kernel?kernel->name.c_str():"";}
    void dump_base() const {
        
        auto shape=this->activations.shape;
        printf("\t\t\"index\":%d,\t\"type\":\"%s\",\t\"shape\":[%d,%d,%d,%d],\t\"function\":\"%s\",\n",
                this->index,
                this->name(),
                shape.x,shape.y,shape.z,shape.w,
                this->kernel_name());
        if (this->inputs.size()>0){
            printf("\t\t\"inputs\":[");
            for (int i=0; i<this->inputs.size(); i++){
                printf("%d,",this->inputs[i]);

            }
            printf("],\n");
        }
        
        
    }
    virtual void eval();
    virtual void set_extra_args(int basearg) {}
    virtual void estimate_cost(NeuralNet::Cost*) const{};
    int channels() const{return activations.shape.z;}
    int width() const {return activations.shape.x;}
    int height() const {return activations.shape.y;}

protected:
    nodeid index;
    std::vector<nodeid> inputs;
    void set_size(Int3 size){
        activations.init_random(Int4(size.x,size.y,size.z,1));
    }
    virtual const char* name() const=0;
    Node(NeuralNet* _net, const char* kernel_entrypt) {
        assert(_net!=nullptr && "must create by passing a NeuralNet that will take ownership of this");
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
    void set_kernel_buffer_args();
    virtual ~Node();
    virtual void dump_extra(){};
};

NeuralNet::Cost NeuralNet::estimate_cost() const{
    NeuralNet::Cost c;
    for (auto& n:nodes){n->estimate_cost(&c); c.activations+=n->activations.shape.hmul();}
    return c;
}

NeuralNet::~NeuralNet() noexcept{
    printf("destructing neuralnet");
    for (auto x : this->nodes) {
        assert(x->net==this);
        x->net=nullptr;
        delete x;
    }
}

void NeuralNet::dump() {
    printf("{\"nodes\":[\n");
    for (auto n :nodes) {
        printf("\t{\n");
        n->dump_base();
        n->dump_extra();
        printf("\t},\n");
    }
    printf("],\n");
    auto tmp=this->estimate_cost();
    printf("\"cost\":{\n\t\"parameters\":%de6\n",tmp.parameters/1000000);
    printf("\t\"fmadds\":%lue9\n",tmp.fmadds/1000000000);
    printf("\t\"activations\":%de6\n",tmp.activations/1000000);
    printf("\t}\n");
    printf("}\n");
}
void NeuralNet::eval() {
    
    for (auto& node : this->nodes) {
        for (auto& x:node->inputs){
            assert(x < node->index && "directed node graph, sources must preceed dests");
        }
        node->eval();
    }
    


}

void NeuralNet::Node::set_kernel_buffer_args(){
    assert(this->kernel!=nullptr);
    // conventoin expected by kernel code: first arg is destination
    // alternate buffer and shape data
    this->kernel->set_arg_buffer_shape(0,this->activations);
    // then list sources
    for (size_t i=0; i<this->inputs.size(); i++) {
        this->kernel->set_arg_buffer_shape((i+1)*2, this->input_node(i)->activations);
    }
    // kernel custom args follow
    
}
bool can_inc_dim(int current, int max){
    return ((current*2) <=max) && (max %(current*2))==0;
}
Int3 get_workgroup_size_for(const Int3& worksize){
    auto ret=Int3(1,1,1);
    int maxsize=gDeviceInfo.max_workgroup_size;

    while ((ret.hmul()*2)<= maxsize) {
        bool can_inc_z= can_inc_dim(ret.z, worksize.z) ;
        // try to increase width or height then depth
        if (ret.x<ret.y && (ret.x<=ret.z || !can_inc_z) && can_inc_dim(ret.x, worksize.x)) {
            ret.x*=2;
        } else if ((ret.y<=ret.z  || !can_inc_z) && can_inc_dim(ret.y,worksize.y) ) {
            ret.y*=2;
        }
        else if (can_inc_z){
            ret.z*=2;
        } else {
            break;
        }
    }

    
    return ret;
}
void NeuralNet::Node::eval() {
    //printf("eval node:%s{\n",this->name());
    if (this->kernel==nullptr) {return;}
    this->set_kernel_buffer_args();
    this->set_extra_args((this->inputs.size()+1)*2);
    // todo - tweaking of ß
    assert(this->activations.shape.w==1 && "node sizes must be 3d");
    
    auto worksize=this->activations.shape.xyz()/this->output_dilation;
    auto grpsize=get_workgroup_size_for(worksize);
    // 
    if (worksize.z% grpsize.z!=0) {
        grpsize.z = worksize.z; // TODO better.
    }

    //std::cout<<grpsize<<" "<<this->activations.shape.xyz()<<"\n";
    
    this->kernel->enqueue_range_3d( worksize, grpsize);
    //printf("}\n");
}

NeuralNet::Node::~Node() {assert(this->net==0 && "must only be manipulated by owning NeuralNet, dont store on stack etc");printf("destructing node %d\n",this->index);}

class Conv2d : public NeuralNet::Node{
    Buffer<float> filter;
    const char* name() const override{return "Conv2d";};
    Int2 stride=Int2(1,1);
    
    float negfactor=0.0f;

    void dump_extra() override {
        printf("\t\t\"filter_shape\":[%d,%d,%d,%d],\n",filter.shape.x,filter.shape.y,filter.shape.z,filter.shape.w);
    }
    void set_extra_args(int argid) override{
        assert(argid==4);
        this->kernel->set_arg_buffer_shape(argid,filter);
        this->kernel->set_arg(argid+2, this->stride);
        this->kernel->set_arg(argid+3, this->negfactor);
    }
    void estimate_cost(NeuralNet::Cost* dst) const override {
        dst->parameters+=filter.shape.hmul();
        dst->fmadds+=(size_t)filter.shape.hmul() * (size_t)activations.shape.x* (size_t)activations.shape.y;
    }
public:    
    Conv2d(NeuralNet* owner, int _input, Int2 _size, int _channels_out, int _stride=1,float _negf=0.0f) :Node(owner,"conv2d_nhwc",_input), stride(_stride,_stride),negfactor(_negf){
        
        auto inp=input_node(0);
        int input_channels=inp->channels();
        assert(
            (input_channels &3)==0 && 
            (_channels_out&3)==0 && "channel sizes must be multiple of 4, use RGBA etc.");
        this->set_size( Int3(inp->width()/stride.x,inp->height()/stride.y, _channels_out) );
        filter.init_random(Int4(_size.x,_size.y, input_channels,_channels_out));
    }
};

//todo rename DILATED convolution.
class ConvDilated2x : public NeuralNet::Node{
    Buffer<float> filter;
    const char* name() const override{return "ConvDilated2x";};
    float negfactor=0.0f;
    void dump_extra() override {
        printf("\t\t\"filter_shape\":[%d,%d,%d,%d],\n",filter.shape.x,filter.shape.y,filter.shape.z,filter.shape.w);
    }
    void set_extra_args(int argid) override{
        assert(argid==4);
        this->kernel->set_arg_buffer_shape(argid,filter);
        this->kernel->set_arg(argid+2, this->negfactor);
    }
    void estimate_cost(NeuralNet::Cost* dst) const override {
        dst->parameters+=filter.shape.hmul();
        dst->fmadds+=filter.shape.hmul() * activations.shape.x* activations.shape.y / 4;
    }
public:    

    ConvDilated2x(NeuralNet* owner, int _input, Int2 _filter_size, int _channels_out, float _negf=0.0f) :Node(owner,"deconv_xy_2x_nhwc",_input), negfactor(_negf){
        assert((_filter_size.x&1)==0 && (_filter_size.y&1)==0 && "filter be multiple of 2");
        assert(_filter_size.x==6 && "we use hardcoded 3xdilation2= 6x6 kernel optimized unrolled loop");
        auto inp=input_node(0);
        int input_channels=inp->channels();
        assert((input_channels &3)==0 && (_channels_out&3)==0 && "channel sizes must be multiple of 4, use RGBA etc.");
        this->output_dilation=Int3(2,2,1);
        this->set_size( output_dilation*Int3(inp->width(),inp->height(), _channels_out) );
        filter.init_random(Int4(_filter_size.x,_filter_size.y, input_channels,_channels_out));
    }
};



class FullyConnected : public NeuralNet::Node {
    Buffer<float> matrix_weights;
    const char* name() const override{return "FullyConnected";};
public:
    void estimate_cost(NeuralNet::Cost* c)const override{
        c->fmadds+=matrix_weights.shape.hmul();
        c->parameters+=matrix_weights.shape.hmul();
    }
    FullyConnected(NeuralNet* owner, int _input, int _channels_out) : NeuralNet::Node(owner,"matmul_on_z",_input) {
        auto inp=input_node(0);
        auto s=inp->activations.shape.hmul();
        assert(s==inp->activations.shape.z && "input to fully connected layer must be flattened to Z, assumptions for interleave..");
        this->matrix_weights.set_size(Int4(1,1, s, _channels_out));
        this->set_size( Int3(1,1, _channels_out));
    }
    void set_extra_args(int argid) override{
        assert(argid==4);
        this->kernel->set_arg_buffer_shape(argid,matrix_weights);
    }

};
class ConcatZ : public NeuralNet::Node {
    const char* name() const override{return "ConcatZ";}
public:
    ConcatZ(NeuralNet* owner, int src0, int src1) : NeuralNet::Node(owner,"concat_z",src0,src1) {
        auto in0=this->input_node(0),in1=this->input_node(1);
        assert(in0->width()==in1->width() && in0->height()==in1->height());
        this->set_size(Int3(in0->width(),in0->height(), in0->channels()+in1->channels()));
    }
};

class Add : public NeuralNet::Node {
    const char* name() const override{return "Add";}
public:
    Add(NeuralNet* owner, int src0, int src1) : NeuralNet::Node(owner,"vector_add",src0,src1) {
        auto in0=this->input_node(0),in1=this->input_node(1);
        assert(in0->activations.shape==in1->activations.shape);
        this->set_size(Int3(in0->width(),in0->height(), in0->channels()));
    }
};

class AvPool2x2 : public NeuralNet::Node {
    const char* name() const override{return "AvPool2x2";}
public:
    AvPool2x2(NeuralNet* owner,int _input=-1) : NeuralNet::Node(owner,"avpool2x2_nhwc",_input){
        auto inp=input_node(0);
        this->set_size( Int3(inp->width()/2,inp->height()/2,inp->channels()));
    }
};
class DebugFill : public NeuralNet::Node {
    const char* name() const override{return "DebugFill";}
    float val;
public:
    DebugFill(NeuralNet* owner,float _val,int _input=-1) : NeuralNet::Node(owner,"debug_fill",_input), val(_val){
        auto inp=input_node(0);
        this->set_size( input_node(0)->activations.shape.xyz() );
    }
    void set_extra_args(int argid) override{
        this->kernel->set_arg(argid,val);
    }

};


class MaxPool2x2 : public NeuralNet::Node {
    const char* name() const override{return "MaxPool2x2";}
public:
    MaxPool2x2(NeuralNet* owner,int _input=-1) : NeuralNet::Node(owner,"maxpool2x2",_input){
        auto inp=input_node(0);
        this->set_size( Int3(inp->width()/2,inp->height()/2,inp->channels()));
    }
};

// hacky, probably not useful.
class Expand2x2 : public NeuralNet::Node {
    const char* name() const override{return "Expand2x2";}
public:
    Expand2x2(NeuralNet* owner,int _input=-1) : Node(owner,"expand2x2",_input){
        auto inp=input_node(0);
        this->set_size( Int3(inp->width()*2,inp->height()*2,inp->channels()));
    }
};

class FlattenToZ : public NeuralNet::Node {
    const char* name() const override{return "FlattenToZ";}
public:
    FlattenToZ(NeuralNet* owner,int _input=-1) : Node(owner,"flatten_to_z",_input){
        auto inp=input_node(0);
        this->set_size( Int3(1,1, inp->width()*inp->height()*inp->channels()) );
    }
    void eval() override {
        // NOP until we have interleave
    }
};

class InputImage : NeuralNet::Node{
    const char* name() const override{return "InputImage";};
public:
    InputImage(NeuralNet* net, Int3 _size) : NeuralNet::Node(net,nullptr) {
        this->set_size(_size);
    }
};

std::unique_ptr<NeuralNet> make_trivial_edgedetector_convnet() {
    std::unique_ptr<NeuralNet> thenet= std::make_unique<NeuralNet>();
    NeuralNet* net=thenet.get();
    new InputImage(net, Int3(256,256,4));
    new Conv2d(net,-1 , Int2(3,3), 1, 1);
    new AvPool2x2(net);
    new ConvDilated2x(net,-1 , Int2(6,6), 4);
    return thenet;
}

std::unique_ptr<NeuralNet> make_example_convnet() {
    std::unique_ptr<NeuralNet> thenet= std::make_unique<NeuralNet>();
    NeuralNet* net=thenet.get();

    new InputImage(net, Int3(256,256,4));

    new Conv2d(net,-1 , Int2(3,3), 16, 1); 
    new Conv2d(net,-1 , Int2(3,3), 24, 2);  // stride 2 to downsample->128x128
    new Conv2d(net,-1 , Int2(3,3), 32, 1);
    new Conv2d(net,-1 , Int2(3,3), 32, 2);  // 64x64 x 32
    new Conv2d(net,-1 , Int2(3,3), 64, 1);
    new Conv2d(net,-1 , Int2(3,3), 64, 2);  // 32x32 x 64
    new Conv2d(net,-1 , Int2(3,3), 128, 1);
    new Conv2d(net,-1 , Int2(3,3), 128, 2); // -> 16x16 x 128
    new Conv2d(net,-1 , Int2(3,3), 256, 1); // 16x16 x 256 = deepest latent representation
    new ConvDilated2x(net,-1 , Int2(6,6), 128); // now deconvs expand (=dilated convolution)
    new ConvDilated2x(net,-1 , Int2(6,6), 128);
    new ConvDilated2x(net,-1 , Int2(6,6), 64);
    new ConvDilated2x(net,-1 , Int2(6,6), 32);
    
    new ConvDilated2x(net,-1 , Int2(6,6), 16);
    new ConvDilated2x(net,-1 , Int2(6,6), 4);

    return thenet;
}

void test_setup_convnet() {
    TRACE
    auto net = make_example_convnet();

    //new ConvDilated2x(&net,-1 , Int2(6,6), 3);


/*
    new FlattenToZ(&net);
    new FullyConnected(&net,-1, 128);
    new FullyConnected(&net,-1, 32);
*/
    net->dump();
    //exit(0);
    TRACE
    
    int num_iter=1;
    printf("run %d iterations..\n",num_iter);
    for (int i=0; i<num_iter; i++) {
        net->eval();   
    }
    net->last_node()->activations.from_device();

    std::cout<<"output:\n"; 
    if (net->last_node()->activations.shape.hmul()<1024) {
        
        std::cout<<net->last_node()->activations;
    } else {
        std::cout<<"(too big to print..)\n"; 
    }
    TRACE
}


int SCREEN_HEIGHT = 256;
int SCREEN_WIDTH = 512;
void run_window_main_loop(std::function<void(SDL_Surface*,int frame)> generate_image) {
    TRACE
	SDL_Event event;
	SDL_Init(SDL_INIT_VIDEO);
	SDL_Window *window = SDL_CreateWindow("SDL window", 0, 0, 
		SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_HIDDEN);
	SDL_ShowWindow(window);
    
    int frame=0;
	int running = 1;
	while(running) {
		while(SDL_PollEvent(&event)) {
			if(event.type == SDL_QUIT) {
				running = 0;
			}
		}
        frame+=1;
        
        SDL_Surface* sfc= SDL_GetWindowSurface(window);
        SDL_LockSurface(sfc);
        generate_image(sfc,frame);
        SDL_UnlockSurface(sfc);
        SDL_UpdateWindowSurface(window);

	}
	SDL_DestroyWindow(window);
	SDL_Quit();
}

void scrolling_window_test(SDL_Surface* sfc,int frame){
    for (int y=0; y<sfc->h; y++) {
        for (int x=0; x<sfc->w; x++){
            auto pixel=(((uint8_t*)sfc->pixels)+x*sfc->format->BytesPerPixel+y * sfc->pitch);
            pixel[0]=x-frame;
            pixel[1]=x+frame;
            pixel[2]=y-frame;
        }
    }
}

void fill_buffer_from_sdl_surface(Buffer<float>& buffer, SDL_Surface* src){
    if (!src) return;
    TRACE
    int w=std::min(buffer.shape.x, src->w);
    int h=std::min(buffer.shape.y, src->h);
    int numc=std::min(buffer.shape.z,(int)src->format->
        BytesPerPixel);
    //SDL_LockSurface(src);
    auto pixels=(uint8_t*)src->pixels;
    for (int y=0; y<h; y++) {
        for (int x=0; x<w; x++){
            for (int z=0; z<numc; z++){
                buffer.operator[](Int4(x,y,z,0)) = 
                    (float)pixels[x*src->format->BytesPerPixel+y*src->pitch + z] * (1.0/255.0);
            }
        }
    }
    //SDL_UnlockSurface(src);
    buffer.to_device();
    TRACE
}

void copy_sdl_surface_from_buffer(SDL_Surface* sfc, int x0,int y0, int w,int h, Buffer<float>* src,float scale) {

    if (!sfc) return;
    w= std::min(std::min((int)(sfc->w-x0),w),(int)src->shape.x);
    h= std::min(std::min((int)(sfc->h-y0),h),(int)src->shape.y);

    for (int y=0; y< 256; y++){
        for (int x=0; x< 256; x++) {
            if (x<0 || x>=sfc->w || y<0 || y>=sfc->h) continue;
            auto dstpixel=(((uint8_t*)sfc->pixels)+(x+x0)*sfc->format->BytesPerPixel+(y+y0) * sfc->pitch);

            std::array<float,3> sp= src->get_pixel<3>(x,y);
            dstpixel[0] = (uint8_t) (sp[0]*scale) ;
            dstpixel[1] = (uint8_t) (sp[1]*scale) ;
            dstpixel[2] = (uint8_t) (sp[2]*scale) ;
            
        }
    }

}
void run_neural_net_test(SDL_Surface* input) {
    //auto net = make_trivial_edgedetector_convnet();
    auto net = make_example_convnet();
    

    fill_buffer_from_sdl_surface(net->first_node()->activations, input);

    run_window_main_loop([&](SDL_Surface* sfc, int frame) {
        
        net->eval();
        
        
        NeuralNet::Node* last=net->last_node();
        NeuralNet::Node* first=net->first_node();
        
        last->activations.from_device();

        copy_sdl_surface_from_buffer(sfc, frame&255,0,sfc->w/2,sfc->h, &first->activations,256.0);

         copy_sdl_surface_from_buffer(sfc, sfc->w/2,0,sfc->w/2,sfc->h, &last->activations,1024.0);
        
    });
}


int main(int argc, const char** argv) {
    SDL_Surface* input_image=nullptr;
    if (argc>1) {
        printf("loading image %s\n",argv[1]);
        input_image=IMG_Load(argv[1]);
    } else{
        input_image=IMG_Load("01mame.png");
    }

	opencl_init();
	opencl_test_basic();
    test_setup_convnet();
//    run_window_main_loop(scrolling_window_test);
    run_neural_net_test(input_image);
    
	opencl_shutdown();

	return 0;
}
