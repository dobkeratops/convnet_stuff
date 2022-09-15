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
    // TODO - unrolling opt
    int fi = filter_shape.x*filter_shape.y*filter_shape.z * dst_channel;
    for (int kz=0; kz<filter_shape.z; kz++) {
        
        int si =lin_index(src_shape, sx,sy,kz);
        for (int ky=0; ky<filter_shape.y; ky++){
            
            for (int kx=0; kx<filter_shape.x; kx++) {

                //float s=get3df(src,src_shape,sx+kx, sy+ky, kz);
                //float f=get3df(filter, filter_shape, kx,ky,kz + layerofs);
                float s = src[si]; si+=1;
                float f = filter[fi]; fi+=1;
                sum+=s*f;
            }
            si+= src_shape.x-filter_shape.x;
            
        }        
    }
    if (sum<0.0) {sum*=negfactor;}

    set3df(dst, dst_shape, ix,iy,dst_channel, sum);
}

__kernel void deconv_xy_2x_planar(
        __global float* dst,                 // 3d array wdith,height, dstchannels
        int4 dst_shape,
        __global const float* src,  // 3d array width,height,srcchannels
        int4 src_shape,
        __global const float* filter, // 4D array, width,height,srcchannels,dstchannels
        int4 filter_shape,
        float negfactor)    // set 0.0 for relu, 1.0 for nothing, 0.1 for modified relu
{
    int sx=get_global_id(0); // dest x
    int sy=get_global_id(1); // dest y
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
    if (sum00<0.0) {sum00*=negfactor;}
    if (sum01<0.0) {sum01*=negfactor;}
    if (sum10<0.0) {sum10*=negfactor;}
    if (sum11<0.0) {sum11*=negfactor;}

    //todo 2x2 block write
    set3df(dst, dst_shape, dx,  dy,  dst_channel, sum00);
    set3df(dst, dst_shape, dx+1,dy,  dst_channel, sum01);
    set3df(dst, dst_shape, dx,  dy+1,dst_channel, sum10);
    set3df(dst, dst_shape, dx+1,dy+1,dst_channel, sum11);

}
