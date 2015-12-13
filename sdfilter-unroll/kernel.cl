#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// m0 m1 m2
// m3 m4 m5
// m6 m7 m8
__kernel void sdfilter_unroll(__global float *src, __global float *dst, long ldc,
  double m0, double m1, double m2, double m3, double m4, double m5, double m6, double m7, double m8)
{
  long x = get_global_id(0);
  long y = get_global_id(1);
  
  double i0 = src[(x-1)+(y-1)*ldc]*m0;
  double i1 = src[(x)  +(y-1)*ldc]*m1;
  double i2 = src[(x+1)+(y-1)*ldc]*m2;
  double i3 = src[(x-1)+(y)  *ldc]*m3;
  double i4 = src[(x)  + y  * ldc]*m4;
  double i5 = src[(x+1)+(y)  *ldc]*m5;
  double i6 = src[(x-1)+(y+1)*ldc]*m6;
  double i7 = src[(x)  +(y+1)*ldc]*m7;
  double i8 = src[(x+1)+(y+1)*ldc]*m8;

  dst[x+y*ldc] = i0 + i1 + i2 + i3 + i4 + i5 + i6 + i7 + i8;

  double i0_1 = src[(x-1)+((y+1)-1)*ldc]*m0;
  double i1_1 = src[(x)  +((y+1)-1)*ldc]*m1;
  double i2_1 = src[(x+1)+((y+1)-1)*ldc]*m2;
  double i3_1 = src[(x-1)+((y+1))  *ldc]*m3;
  double i4_1 = src[(x)  + (y+1)  * ldc]*m4;
  double i5_1 = src[(x+1)+((y+1))  *ldc]*m5;
  double i6_1 = src[(x-1)+((y+1)+1)*ldc]*m6;
  double i7_1 = src[(x)  +((y+1)+1)*ldc]*m7;
  double i8_1 = src[(x+1)+((y+1)+1)*ldc]*m8;

  dst[x+(y+1)*ldc] = i0_1 + i1_1 + i2_1 + i3_1 + i4_1 + i5_1 + i6_1 + i7_1 + i8_1;
}
