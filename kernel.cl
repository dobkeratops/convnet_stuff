__kernel void vector_add( __global float *dst, __global const float *src0, __global const float *src1) {
    int i = get_global_id(0);					
    dst[i] = src0[i]+ src1[i];
}									
__kernel void vector_add_scaled(__global float *dst, __global const float *src0, __global const float *src1, float f0, float f1) {
    int i = get_global_id(0);					
    dst[i] = src0[i]*f0 + src1[i]*f1;
}
/*
__kernel void vector_clamp(__global float *dst, __global const float *src0,  float fmin,float fmax) {
    int i = get_global_id(0);
    float f=src0[i];
    if (f<fmin) f=fmin; else if (f>fmax) f=fmax;
    dst[i] = f;
}									
*/
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

// todo - calculate indices per line etc.
float get3df(__global const float* src, int4 srcshape, int x,int y,int z) {
    return src[x + srcshape.x*(y + srcshape.y * z)];
}
void set3df(__global float* dst, int4 dstshape, int x, int y, int z, float value) {
    dst[x + dstshape.x*(y + dstshape.y * z)] = value;
}

// todo .. bias & custom saturation.
// todo - interleaving 'z' probably better (width,height,channels vs ..)

__kernel void conv2d_planar(
        __global float* dst,                 // 3d array wdith,height, dstchannels
        __global const float* src,  // 3d array width,height,srcchannels
        __global const float* filter, // 4D array, width,height,srcchannels,dstchannels
        int4 src_shape,
        int2 src_stride,
        int4 filter_shape,
        int4 dst_shape,
        int dst_channel) {
    int ix=get_global_id(0);
    int iy=get_global_id(1);
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

    set3df(dst, dst_shape, ix,iy,dst_channel, sum);
}
