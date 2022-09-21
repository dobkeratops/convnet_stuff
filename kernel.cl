#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef float val_t;
typedef float4 val4_t;

inline val_t vec4hadd(val4_t a){
    return (a.x+a.y)+(a.z+a.w);
}

// crude retrofitable tiling, 4x4 , doesn't need to know whole size TODO more elegane totalsize aware alternative
// .. no effect, so disabled.
#define DETILE(X,Y) //{const int tsize=1;int tx=X/tsize; int ty=Y/tsize; int subx=X&(tsize-1); int suby=Y&(tsize-1); X=tx*tsize+suby; Y=ty*tsize+subx;}


__kernel void vector_add(__global float *dst, __global const float *src0, __global const float *src1) {
    int i = get_global_id(0);					
    dst[i] = src0[i]+ src1[i];
}
// todo - calculate indices per line etc.
// wzyx
int lin_index(int4 shape, int x,int y,int z){
    return x+shape.x*(y + shape.y* z);
}
// wyxz
int lin_index_nhwc(int4 shape, int x,int y,int z){
    return z+shape.z*(x + shape.x* y);
}

float leaky_relu(float x, float f){return x<0.0f?(x*f):x;}

float4 leaky_relu4(float4 v,float f) {
    return (float4)(leaky_relu(v.x,f),leaky_relu(v.y,f),leaky_relu(v.z,f),leaky_relu(v.w,f));
}

struct int2x2 { int m00,m01, m10, m11};

struct int2x2 lin_index_nhwc_2x2(int4 shape, int x,int y,int z){
    
    struct int2x2 ret;
    ret.m00= z + shape.z*(x + shape.x* y);
    ret.m10= ret.m00 + shape.z;
    ret.m01= ret.m00 + shape.z*shape.x;
    ret.m11 = ret.m01 + shape.z;
    return ret;  
}

struct int2x2 lin_index_nhwc_2x2_simd4(int4 shape, int x,int y,int z){
    
    struct int2x2 ret;
    int szdiv4=shape.z/4;
    ret.m00= z + szdiv4*(x + shape.x* y);
    ret.m10= ret.m00 + szdiv4;
    ret.m01= ret.m00 + szdiv4*shape.x;
    ret.m11 = ret.m01 + szdiv4;
    return ret;  
}


float get3df(__global const float* src, int4 srcshape, int x,int y,int z) {
    return src[x + srcshape.x*(y  + srcshape.y * z)];
}
void set3df(__global float* dst, int4 dstshape, int x, int y, int z, float value) {
    dst[x + dstshape.x*(y + dstshape.y * z)] = value;
}

void set3df_nhwc(__global float* dst, int4 dstshape, int x, int y, int z, float value) {
    dst[z + dstshape.z*(x + dstshape.x * y)] = value;
}

__kernel void concat_z(int4 dstofs,int4 dststride, __global float *dst, int4 dstsize, __global const float *src0,int4 src0size, __global const float *src1,int4 src1size) {
    int i = get_global_id(0),j = get_global_id(1),k = get_global_id(2);
    float val=(k<src0size.z)? get3df(src0,src0size,i,j,k) : get3df(src1,src1size,i,j,k-src0size.z);
    set3df(dst,dstsize, i,j,k, val);
}

__kernel void matmul_on_z(int4 dstofs,int4 dststride,__global float *dst, int4 dstsize, __global const float *src0,int4 src0size, __constant const float* matrix, int4 matrixsize) {
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

__kernel void debug_fill(int4 dstofs,int4 dststride ,__global float* dst,int4 dstsize, float x){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    set3df(dst,dstsize,i,j,k, x+(float)(i%100)+(float)(j%100)*100.0+(float)(k%100)*10000.0);
}

__kernel void avpool2x2(int4 dstofs,int4 dststride, __global float* dst,int4 dstsize, __global const float* src0, int4 src0size){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    int si=lin_index(src0size,i*2,j*2,k);
    float val=0.25f*(src0[si] + src0[si+1] + src0[si+src0size.x]+src0[si+src0size.x+1]);
    //float val  = get3df(src0,src0size, i*2,j*2,k);
    set3df(dst,dstsize,i,j,k, val);
}
__kernel void avpool2x2_nhwc(int4 dstofs,int4 dststride, __global float* dst,int4 dstsize, __global const float* src0, int4 src0size){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    int si00=lin_index_nhwc(src0size,i*2,(j+1)*2,k);
    int si10=lin_index_nhwc(src0size,i*2,(j+1)*2,k);

    float val=0.25f*(src0[si00] + src0[si00+src0size.z] + src0[si10]+src0[si10+src0size.z]);
    
    set3df_nhwc(dst,dstsize,i,j,k, val);
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
__kernel void vector_mul_add_clamp(int4 dstofs,int4 dststride, __global float *dst, __global const float *src0,  float scale, float ofs, float min, float max) {
    int i = get_global_id(0);
    float f=src0[i]*scale + ofs;
    if (f<min) f=min; else if (f>max) f=max;
    dst[i] = f;
}									

// todo .. bias & custom saturation.
// todo - interleaving 'z' probably better (width,height,channels vs ..)

__kernel void conv2d_planar(
        int4  dstofs,
        int4 dststride,
        __global float* dst,                 // 3d array wdith,height, dstchannels
        int4 dst_shape,
        __constant const float* src,  // 3d array width,height,srcchannels
        int4 src_shape,
        __constant const float* filter, // 4D array, width,height,srcchannels,dstchannels
        int4 filter_shape,
        int2 src_stride,
        float negfactor)    // set 0.0 for relu, 1.0 for nothing, 0.1 for modified relu
{
    int ix=get_global_id(0); // dest x
    int iy=get_global_id(1); // dest y
    
    DETILE(ix,iy);
    int dst_channel=get_global_id(2); // dest channel

    int sx = ix*src_stride.x;
    int sy = iy*src_stride.y;

    float sum=0.0;

    int layerofs = dst_channel * filter_shape.z;
    // requires shape0.z == shape1.z
    // TODO - unrolling opt
    int fi = filter_shape.x*filter_shape.y*filter_shape.z * dst_channel;
    int kxmax=filter_shape.x;
    int kymax=filter_shape.y;
    if (sx+kxmax > src_shape.x){kxmax=src_shape.x-sx;}
    if (sy+kymax > src_shape.y){kymax=src_shape.y-sy;}
    for (int kz=0; kz<filter_shape.z; kz++) {
        
        int si =lin_index(src_shape, sx,sy,kz);
        for (int ky=0; ky<kymax; ky++){
            
            for (int kx=0; kx<kxmax; kx++) {

                //float s=get3df(src,src_shape,sx+kx, sy+ky, kz);
                //float f=get3df(filter, filter_shape, kx,ky,kz + layerofs);
                float s = src[si]; si+=1;
                float f = filter[fi]; fi+=1;
                sum+=s*f;
            }
            si+= src_shape.x-kxmax;
            
        }        
    }

    set3df(dst, dst_shape, ix,iy,dst_channel, leaky_relu(sum,negfactor));
}

__kernel void conv2d_nhwc(
        int4  dstofs,int4 dststride,    
        __global val_t* dst,                 // 3d array wdith,height, dstchannels
        int4 dst_shape,
        __global const val_t* src,  // 3d array width,height,srcchannels
        int4 src_shape,
        __constant const val_t* filter, // 4D array, width,height,srcchannels,dstchannels
        int4 filter_shape,
        int2 src_stride,
        float negfactor)    // set 0.0 for relu, 1.0 for nothing, 0.1 for modified relu
{
    int ix=get_global_id(0); // dest x
    int iy=get_global_id(1); // dest y
    DETILE(ix,iy);
    int dst_channel=get_global_id(2); // dest channel

    int sx = ix*src_stride.x;
    int sy = iy*src_stride.y;

    float sum=0.0;

    int layerofs = dst_channel * filter_shape.z;
    // requires shape0.z == shape1.z
    // TODO - unrolling opt
    int fi = filter_shape.x*filter_shape.y*filter_shape.z * dst_channel;
    int kxmax=filter_shape.x;
    int kymax=filter_shape.y;
    if (sx+kxmax > src_shape.x){kxmax=src_shape.x-sx;}
    if (sy+kymax > src_shape.y){kymax=src_shape.y-sy;}
        
    for (int ky=0; ky<kymax; ky++){
        
        for (int kx=0; kx<kxmax; kx++) {

            int si =lin_index_nhwc(src_shape, sx+kx,sy+ky,0);
            int kz=0;
            for (; (kz+4)<=filter_shape.z; kz+=4,fi+=4, si+=4) {
                sum+=src[si]*filter[fi];
                sum+=src[si+1]*filter[fi+1];
                sum+=src[si+2]*filter[fi+2];
                sum+=src[si+3]*filter[fi+3];
            }
            for (; kz<filter_shape.z; kz+=1,fi+=1, si+=1) {
                sum+=src[si]*filter[fi];
            }
        }        
    }

    set3df_nhwc(dst, dst_shape, ix,iy,dst_channel, leaky_relu(sum,negfactor));
}

__kernel void conv2d_nhwc_block2x2x4(
        int4  dstofs,int4 dststride,    
        __global val_t* dst,                 // 3d array wdith,height, dstchannels
        int4 dst_shape,
        __global const val_t* src,  // 3d array width,height,srcchannels
        int4 src_shape,
        __constant const val_t* filter, // 4D array, width,height,srcchannels,dstchannels
        int4 filter_shape,
        int2 src_stride,
        float negfactor)    // set 0.0 for relu, 1.0 for nothing, 0.1 for modified relu
{
    
    int ix=get_global_id(0)*2; // dest x
    int iy=get_global_id(1)*2; // dest y
    DETILE(ix,iy);
    int dst_channel=get_global_id(2)*4; // dest channel

    int sx = ix*src_stride.x;
    int sy = iy*src_stride.y;


    int layerofs = dst_channel * filter_shape.z;
    // requires shape0.z == shape1.z
    // TODO - unrolling opt
    int filter_channelsize=filter_shape.x*filter_shape.y*filter_shape.z;
    int fi0 = filter_channelsize * dst_channel;
    int fi1 = fi0+ filter_channelsize;
    int fi2 = fi1+ filter_channelsize;
    int fi3 = fi2+ filter_channelsize;

    float sum000=0.0;
    float sum100=0.0;
    float sum010=0.0;
    float sum110=0.0;
    float sum001=0.0;
    float sum101=0.0;
    float sum011=0.0;
    float sum111=0.0;
    float sum002=0.0;
    float sum102=0.0;
    float sum012=0.0;
    float sum112=0.0;
    float sum003=0.0;
    float sum103=0.0;
    float sum013=0.0;
    float sum113=0.0;

    for (int ky=0; ky<filter_shape.x; ky++){
        
        for (int kx=0; kx<filter_shape.y; kx++) {

            int si00 =lin_index_nhwc(src_shape, sx+kx,sy+ky,0);
            int si10 =lin_index_nhwc(src_shape, sx+kx+1,sy+ky,0);
            int si01 =lin_index_nhwc(src_shape, sx+kx,sy+ky+1,0);
            int si11 =lin_index_nhwc(src_shape, sx+kx+1,sy+ky+1,0);
            int kz=0;
            for (; kz<filter_shape.z; kz+=1,fi0+=1, fi1+=1,fi2+=1,fi3+=1, si00+=1,si01+=1,si10+=1,si11+=1) {
                // 2x2 block of sources
                float s00 = src[si00];
                float s10 = src[si10];
                float s01 = src[si01];
                float s11 = src[si11];

                // .. into 2x2x2 accumulators
                float f0 = filter[fi0]; // out layer0
                float f1 = filter[fi1]; // out layer1
                float f2 = filter[fi2]; // out layer2
                float f3 = filter[fi3]; // out layer3
                // 8 loads -> 16 MACCs.

                sum000 += s00 * f0;
                sum100 += s10 * f0;
                sum010 += s01 * f0;
                sum110 += s11 * f0;
                
                sum001 += s00 * f1;
                sum101 += s10 * f1;
                sum011 += s01 * f1;
                sum111 += s11 * f1;

                sum002 += s00 * f2;
                sum102 += s10 * f2;
                sum012 += s01 * f2;
                sum112 += s11 * f2;

                sum003 += s00 * f3;
                sum103 += s10 * f3;
                sum013 += s01 * f3;
                sum113 += s11 * f3;

            }
        }        
    }

    set3df_nhwc(dst, dst_shape, ix,iy,dst_channel, leaky_relu(sum000,negfactor));
    set3df_nhwc(dst, dst_shape, ix+1,iy,dst_channel, leaky_relu(sum010,negfactor));
    set3df_nhwc(dst, dst_shape, ix,iy+1,dst_channel, leaky_relu(sum100,negfactor));
    set3df_nhwc(dst, dst_shape, ix+1,iy+1,dst_channel, leaky_relu(sum110,negfactor));

    set3df_nhwc(dst, dst_shape, ix,iy,dst_channel+1, leaky_relu(sum001,negfactor));
    set3df_nhwc(dst, dst_shape, ix+1,iy,dst_channel+1, leaky_relu(sum011,negfactor));
    set3df_nhwc(dst, dst_shape, ix,iy+1,dst_channel+1, leaky_relu(sum101,negfactor));
    set3df_nhwc(dst, dst_shape, ix+1,iy+1,dst_channel+1, leaky_relu(sum111,negfactor));

    set3df_nhwc(dst, dst_shape, ix,iy,dst_channel+2, leaky_relu(sum002,negfactor));
    set3df_nhwc(dst, dst_shape, ix+1,iy,dst_channel+2, leaky_relu(sum012,negfactor));
    set3df_nhwc(dst, dst_shape, ix,iy+1,dst_channel+2, leaky_relu(sum102,negfactor));
    set3df_nhwc(dst, dst_shape, ix+1,iy+1,dst_channel+2, leaky_relu(sum112,negfactor));

    set3df_nhwc(dst, dst_shape, ix,iy,dst_channel+3, leaky_relu(sum003,negfactor));
    set3df_nhwc(dst, dst_shape, ix+1,iy,dst_channel+3, leaky_relu(sum013,negfactor));
    set3df_nhwc(dst, dst_shape, ix,iy+1,dst_channel+3, leaky_relu(sum103,negfactor));
    set3df_nhwc(dst, dst_shape, ix+1,iy+1,dst_channel+3, leaky_relu(sum113,negfactor));
    
}



__kernel void deconv_xy_2x_planar_unopt(
        int4  dstofs,int4 dststride,    
        __global float* dst,                 // 3d array wdith,height, dstchannels
        int4 dst_shape,
        __global const float* src,  // 3d array width,height,srcchannels
        int4 src_shape,
        __constant const float* filter, // 4D array, width,height,srcchannels,dstchannels
        int4 filter_shape,
        float negfactor)    // set 0.0 for relu, 1.0 for nothing, 0.1 for modified relu
{
    int sx=get_global_id(0); // dest x
    int sy=get_global_id(1); // dest y
    DETILE(sx,sy);
    int dst_channel=get_global_id(2); // dest channel

    int dx = sx*2;
    int dy = sy*2;

    float sum00=0.0;
    float sum01=0.0;
    float sum10=0.0;
    float sum11=0.0;

    int layerofs = dst_channel * filter_shape.z;
    // requires shape0.z == shape1.z
    // TODO - unrolling opt
    for (int kz=0; kz<filter_shape.z; kz++) {
        
        for (int ky=0; ky<filter_shape.y; ky+=2){
            
            
            //int si=lin_index(src_shape, sx,   sy+ky/2,   kz);
            for (int kx=0; kx<filter_shape.x; kx+=2) {

                // todo - 2x2 block read

                float s=get3df(src,src_shape,sx+kx/2,   sy+ky/2,   kz);
                //float f00=get3df(filter, filter_shape, kx,  ky,   kz + layerofs);
                //float f01=get3df(filter, filter_shape, kx+1,ky,   kz + layerofs);
                //float f10=get3df(filter, filter_shape, kx,  ky+1, kz + layerofs);
                //float f11=get3df(filter, filter_shape, kx+1,ky+1, kz + layerofs);

                int fi = lin_index(filter_shape,kx,ky,kz+layerofs);
                float f00 =filter[fi];
                float f01 =filter[fi+1];
                float f10 =filter[fi + filter_shape.x];
                float f11 =filter[fi+1 + filter_shape.x];

                sum00+=s*f00;
                sum01+=s*f01;
                sum10+=s*f10;
                sum11+=s*f11;
            }
            
        }        
    }
    //todo 2x2 block write
    set3df(dst, dst_shape, dx,  dy,  dst_channel, leaky_relu(sum00,negfactor));
    set3df(dst, dst_shape, dx+1,dy,  dst_channel, leaky_relu(sum01,negfactor));
    set3df(dst, dst_shape, dx,  dy+1,dst_channel, leaky_relu(sum10,negfactor));
    set3df(dst, dst_shape, dx+1,dy+1,dst_channel, leaky_relu(sum11,negfactor));

}
__kernel void deconv_xy_2x_planar(
        int4  dstofs,int4 dststride,
        __global float* dst,                 // 3d array wdith,height, dstchannels
        int4 dst_shape,
        __global const float* src,  // 3d array width,height,srcchannels
        int4 src_shape,
        __constant const float* filter, // 4D array, width,height,srcchannels,dstchannels
        int4 filter_shape,
        float negfactor)    // set 0.0 for relu, 1.0 for nothing, 0.1 for modified relu
{
    int sx=get_global_id(0); // dest x
    int sy=get_global_id(1); // dest y
    DETILE(sx,sy);
    int dst_channel=get_global_id(2); // dest channel

    //int lx=get_local_id(0),ly=get_local_id(1);
    //int lsizex=get_local_size(0), lsizey=get_local_size(1);



    int dx = sx*2;
    int dy = sy*2;

    float sum00=0.0;
    float sum01=0.0;
    float sum10=0.0;
    float sum11=0.0;

    int layerofs = dst_channel * filter_shape.z;
    
    // requires shape0.z == shape1.z
    // TODO - unrolling opt

    // todo stop over-run at boundaries!!
    
    for (int kz=0; kz<filter_shape.z; kz++) {

        int fi = lin_index(filter_shape,0,0,kz+layerofs);
        int fi_nextrow=fi + filter_shape.x;
        int si=lin_index(src_shape, sx, sy, kz);

        for (int ky=0; ky<filter_shape.y; ky+=2, fi+=filter_shape.x,fi_nextrow+=filter_shape.x, si+=src_shape.x){

            // kx=0
            float s=src[si];

            sum00+=s*filter[fi];
            sum01+=s*filter[fi+1];
            sum10+=s*filter[fi_nextrow ];
            sum11+=s*filter[fi_nextrow+1];

            //kx=2
            s=src[si+1];
            sum00+=s*filter[fi+2];
            sum01+=s*filter[fi+3];
            sum10+=s*filter[fi_nextrow+2];
            sum11+=s*filter[fi_nextrow+3];

            //kz=3
            s=src[si+2];
            sum00+=s*filter[fi+4];
            sum01+=s*filter[fi+5];
            sum10+=s*filter[fi_nextrow+4];
            sum11+=s*filter[fi_nextrow+5];
            
        }        
    }
 

    //todo 2x2 block write
    
    int di=lin_index(dst_shape, dx,dy,dst_channel);
    dst[di]=leaky_relu(sum00,negfactor);
    dst[di+1]=leaky_relu(sum01,negfactor);
    dst[di+dst_shape.x]=leaky_relu(sum10,negfactor);
    dst[di+dst_shape.x+1]=leaky_relu(sum11,negfactor);
}

__kernel void dilated_conv_xy_2x_nhwc(
        int4  dstofs,int4 dststride,
        __global val_t* dst,                 // 3d array wdith,height, dstchannels
        int4 dst_shape,
        __global const val_t* src,  // 3d array width,height,srcchannels
        int4 src_shape,
        __constant const val_t* filter, // 4D array, width,height,srcchannels,dstchannels
        int4 filter_shape,
        int2 src_stride,
        float negfactor)    // set 0.0 for relu, 1.0 for nothing, 0.1 for modified relu
{
    int ix=get_global_id(0); // dest x
    int iy=get_global_id(1); // dest y
    DETILE(ix,iy);
    int dst_channel=get_global_id(2); // dest channel

    int dx = ix*2;
    int dy = iy*2;
    int sx = ix * src_stride.x;
    int sy = iy * src_stride.y;

    float sum00=0.0;
    float sum10=0.0;
    float sum01=0.0;
    float sum11=0.0;


    int layerofs = dst_channel * filter_shape.z;
        
    for (int ky=0; ky<filter_shape.y; ky+=2){
        for (int kx=0; kx<filter_shape.x; kx+=2) {
            int si = lin_index_nhwc(src_shape, sx+kx/2, sy+ky/2, 0);
            struct int2x2 fi = lin_index_nhwc_2x2(filter_shape, kx, ky, 0+layerofs);

            int kz=0;

            for (; (kz+4)<=filter_shape.z; kz+=4, si+=4,fi.m00+=4,fi.m01+=4,fi.m10+=4,fi.m11+=4) {
                float s0=src[si];
                float s1=src[si+1];
                float s2=src[si+2];
                float s3=src[si+3];

                sum00+=s0*filter[fi.m00];
                sum00+=s1*filter[fi.m00+1];
                sum00+=s2*filter[fi.m00+2];
                sum00+=s3*filter[fi.m00+3];

                sum10+=s0*filter[fi.m10];
                sum10+=s1*filter[fi.m10+1];
                sum10+=s2*filter[fi.m10+2];
                sum10+=s3*filter[fi.m10+3];

                sum01+=s0*filter[fi.m01];
                sum01+=s1*filter[fi.m01+1];
                sum01+=s2*filter[fi.m01+2];
                sum01+=s3*filter[fi.m01+3];

                sum11+=s0*filter[fi.m11];
                sum11+=s1*filter[fi.m11+1];
                sum11+=s2*filter[fi.m11+2];
                sum11+=s3*filter[fi.m11+3];

            }
        }        
    }


    //todo 2x2 block write
    struct int2x2 di=lin_index_nhwc_2x2(dst_shape, dx,  dy,  dst_channel);

    dst[di.m00] = leaky_relu(sum00,negfactor);
    dst[di.m10] = leaky_relu(sum10,negfactor);
    dst[di.m01] = leaky_relu(sum01,negfactor);
    dst[di.m11] = leaky_relu(sum11,negfactor);

}

//BROKEN,Sorry.
__kernel void dilated_conv_xy_2x_nhwc_block2x2x1(
        int4  dstofs,int4 dststride,
        __global val_t* dst,                 // 3d array wdith,height, dstchannels
        int4 dst_shape,
        __global const val4_t* src,  // 3d array width,height,srcchannels
        int4 src_shape,
        __constant const val4_t* filter, // 4D array, width,height,srcchannels,dstchannels
        int4 filter_shape,
        int2 src_stride,
        float negfactor)    // set 0.0 for relu, 1.0 for nothing, 0.1 for modified relu
{
    int ix=get_global_id(0); // dest x
    int iy=get_global_id(1); // dest y
    DETILE(ix,iy);
    int dst_channel=get_global_id(2); // dest channel

    int dx = ix*4;
    int dy = iy*4;
    int sx = ix * src_stride.x*2;
    int sy = iy * src_stride.y*2;

    float4 sum[4]={float4(0.0f),float4(0.0f),float4(0.0f),float4(0.0f)};

    int layerofs = dst_channel * filter_shape.z;
        
    for (int ky=0; ky<filter_shape.y; ky+=2){
        for (int kx=0; kx<filter_shape.x; kx+=2) {

            int sz4=src_shape.z/4;
            struct int2x2 si;

            si.m00 = sz4*((sx+kx/2)+src_shape.x*(sy+ky/2));
            si.m10 = si.m00 + sz4;
            si.m01 = si.m00 + sz4+src_shape.x;
            si.m11 = si.m01 + sz4;

            struct int2x2 fi;  //=lin_index_nhwc_2x2(filter_shape, kx, ky, 0+layerofs);
            int fsz4=filter_shape.z/4;
            
            fi.m00=(layerofs/4) + fsz4*(kx + filter_shape.x*ky);
            fi.m01=fi.m00 + fsz4;
            fi.m10=fi.m00 + fsz4 * filter_shape.x;
            fi.m11=fi.m10 + fsz4;

            int kz=0;

            for (; kx<=filter_shape.z; kz+=4, si.m00+=1,si.m01+=1, si.m10+=1, si.m11+=1, fi.m00+=1,fi.m01+=1,fi.m10+=1,fi.m11+=1) {

                float4 s00=src[si.m00];
                float4 s10=src[si.m10];
                float4 s01=src[si.m01];
                float4 s11=src[si.m11];

                float4 f00 = filter[fi.m00];
                float4 f10 = filter[fi.m10];
                float4 f01 = filter[fi.m01];
                float4 f11 = filter[fi.m11];

                // caution!           
                //sum[y][x]..    sxy  fxy
                
                sum[0][0] += dot(s00, f00);
                sum[0][1] += dot(s00, f10); 
                sum[0][2] += dot(s10, f00);
                sum[0][3] += dot(s10, f10); 

                sum[1][0] += dot(s00, f01);
                sum[1][1] += dot(s00, f11); 
                sum[1][2] += dot(s10, f01);
                sum[1][3] += dot(s10, f11); 

                sum[2][0] += dot(s01, f00);
                sum[2][1] += dot(s01, f10); 
                sum[2][2] += dot(s11, f00);
                sum[2][3] += dot(s11, f10); 

                sum[3][0] += dot(s01, f01);
                sum[3][1] += dot(s01, f11); 
                sum[3][2] += dot(s11, f01);
                sum[3][3] += dot(s11, f11); 
          }

        }        
        
    }

    sum[0]=leaky_relu4(sum[0],negfactor);
    sum[1]=leaky_relu4(sum[1],negfactor);
    sum[2]=leaky_relu4(sum[2],negfactor);
    sum[3]=leaky_relu4(sum[3],negfactor);


    //todo 2x2 block write
    int dsz4=dst_shape.z/4;
    //struct int di=//lin_index_nhwc_2x2_simd4(dst_shape, dx,  dy,  dst_channel)
    int di = dst_channel + dsz4*(dx + dy*dst_shape.x);
    int dst_row=dsz4*dst_shape.x;

    dst[di] = sum[0][0]; di+=dsz4;
    dst[di] = sum[0][1];  di+=dsz4;
    dst[di] = sum[0][2]; di+=dsz4;
    dst[di] = sum[0][3]; di+=dsz4 + dst_row-(dsz4*4);
    dst[di] = sum[1][0]; di+=dsz4;
    dst[di] = sum[1][1];  di+=dsz4;
    dst[di] = sum[1][2]; di+=dsz4;
    dst[di] = sum[1][3]; di+=dsz4 + dst_row-(dsz4*4);
    dst[di] = sum[2][0]; di+=dsz4;
    dst[di] = sum[2][1];  di+=dsz4;
    dst[di] = sum[2][2]; di+=dsz4;
    dst[di] = sum[2][3]; di+=dsz4 + dst_row-(dsz4*4);
    dst[di] = sum[3][0]; di+=dsz4;
    dst[di] = sum[3][1];  di+=dsz4;
    dst[di] = sum[3][2]; di+=dsz4;
    dst[di] = sum[3][3]; di+=dsz4 + dst_row-(dsz4*4);
}

