__kernel void vector_add( __global float *dst, __global const float *src0, __global const float *src1) {
    int i = get_global_id(0);					
    dst[i] = src0[i]+ src1[i];
}
// todo - calculate indices per line etc.
int lin_index(int4 shape, int x,int y,int z){
    return x+shape.x*(y + shape.y* z);
}
float get3df(__global const float* src, int4 srcshape, int x,int y,int z) {
    return src[x + srcshape.x*(y + srcshape.y * z)];
}
void set3df(__global float* dst, int4 dstshape, int x, int y, int z, float value) {
    dst[x + dstshape.x*(y + dstshape.y * z)] = value;
}

__kernel void concat_z( __global float *dst, int4 dstsize, __global const float *src0,int4 src0size, __global const float *src1,int4 src1size) {
    int i = get_global_id(0),j = get_global_id(1),k = get_global_id(2);
    float val=(k<src0size.z)? get3df(src0,src0size,i,j,k) : get3df(src1,src1size,i,j,k-src0size.z);
    set3df(dst,dstsize, i,j,k, val);
}

__kernel void matmul_on_z(__global float *dst, int4 dstsize, __global const float *src0,int4 src0size, __global const float* matrix, int4 matrixsize) {
    // confusing, take care!
    // Z axis of 3d arrays is considered as input & output of the matrix multiply,
    // besides that 'w' is matrix weights
    int x=get_global_id(0);
    int y=get_global_id(1);
    int vec_index=get_global_id(2);
    int matrix_width=src0size.z;
    
    int rowi=matrix_width * vec_index;
    float sum=0.0f;
    
    for (int i=0; i<src0size.z; i++) {
        sum+= src0[i] * matrix[rowi];
        rowi+=1;
    }
    
    dst[vec_index] = sum;
}
__kernel void debug_fill(__global float* dst,int4 dstsize, __global const float* src0, int4 src0size){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    set3df(dst,dstsize,i,j,k, (float)(i%100)+(float)(j%100)*100.0+(float)(k%100)*10000.0);
}

__kernel void avpool2x2(__global float* dst,int4 dstsize, __global const float* src0, int4 src0size){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    int si=lin_index(src0size,i*2,j*2,k);
    float val=0.25f*(src0[si] + src0[si+1] + src0[si+src0size.x]+src0[si+src0size.x+1]);
    //float val  = get3df(src0,src0size, i*2,j*2,k);
    set3df(dst,dstsize,i,j,k, val);
}

__kernel void vector_add_scaled(__global float *dst, __global const float *src0, __global const float *src1, float f0, float f1) {
    int i = get_global_id(0);					
    dst[i] = src0[i]*f0 + src1[i]*f1;
}

__kernel void vector_clamp(__global float *dst, __global const float *src0,  float fmin,float fmax) {
    int i = get_global_id(0);
    float f=src0[i];
    if (f<fmin) {f=fmin;} else if (f>fmax) {f=fmax;}
    dst[i] = f;
}									

__kernel void vector_mul(__global float *dst, __global const float *src0, __global const float *src1 ) {
    int i = get_global_id(0);
    dst[i] = src0[i]* src1[i];
}									

// rescale values
__kernel void vector_mul_add_clamp(__global float *dst, __global const float *src0,  float scale, float ofs, float min, float max) {
    int i = get_global_id(0);
    float f=src0[i]*scale + ofs;
    if (f<min) f=min; else if (f>max) f=max;
    dst[i] = f;
}									

// todo .. bias & custom saturation.
// todo - interleaving 'z' probably better (width,height,channels vs ..)

__kernel void conv2d_planar(
        __global float* dst,                 // 3d array wdith,height, dstchannels
        int4 dst_shape,
        __global const float* src,  // 3d array width,height,srcchannels
        int4 src_shape,
        __global const float* filter, // 4D array, width,height,srcchannels,dstchannels
        int4 filter_shape,
        int2 src_stride,
        float negfactor)    // set 0.0 for relu, 1.0 for nothing, 0.1 for modified relu
{
    int ix=get_global_id(0); // dest x
    int iy=get_global_id(1); // dest y
    int dst_channel=get_global_id(2); // dest channel

    int sx = ix*src_stride.x;
    int sy = iy*src_stride.y;

    float sum=0.0;

    int layerofs = dst_channel * filter_shape.z;
    // requires shape0.z == shape1.z
    for (int kz=0; kz<filter_shape.z; kz++) {
        
        for (int ky=0; ky<filter_shape.y; ky++){
            
            for (int kx=0; kx<filter_shape.x; kx++) {

                float s=get3df(src,src_shape,sx+kx, sy+ky, kz);
                float f=get3df(filter, filter_shape, kx,ky,kz + layerofs);
                sum+=s*f;
            }
            
        }        
    }
    if (sum<0.0) {sum*=negfactor;}

    set3df(dst, dst_shape, ix,iy,dst_channel, sum);
}
