// m0 m1 m2
// m3 m4 m5
// m6 m7 m8
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void dfilter_unroll(__global double *src, __global double *dst, long ldc,
  double m0, double m1, double m2, double m3, double m4, double m5, double m6, double m7, double m8)
{
  long x = get_global_id(0);
  long y = 1*get_global_id(1);
  
  double i00 = src[(x-1)+(y-1)*ldc]*m0;
  double i01 = src[(x)  +(y-1)*ldc]*m1;
  double i02 = src[(x+1)+(y-1)*ldc]*m2;
  double i03 = src[(x-1)+(y)  *ldc]*m3;
  double i04 = src[(x)  + y  * ldc]*m4;
  double i05 = src[(x+1)+(y)  *ldc]*m5;
  double i06 = src[(x-1)+(y+1)*ldc]*m6;
  double i07 = src[(x)  +(y+1)*ldc]*m7;
  double i08 = src[(x+1)+(y+1)*ldc]*m8;

  dst[x+y*ldc] = i00 + i01 + i02 + i03 + i04 + i05 + i06 + i07 + i08;

  double i10 = src[(x-1)+((y+1)-1)*ldc]*m0;
  double i11 = src[(x)  +((y+1)-1)*ldc]*m1;
  double i12 = src[(x+1)+((y+1)-1)*ldc]*m2;
  double i13 = src[(x-1)+((y+1))  *ldc]*m3;
  double i14 = src[(x)  + (y+1)  * ldc]*m4;
  double i15 = src[(x+1)+((y+1))  *ldc]*m5;
  double i16 = src[(x-1)+((y+1)+1)*ldc]*m6;
  double i17 = src[(x)  +((y+1)+1)*ldc]*m7;
  double i18 = src[(x+1)+((y+1)+1)*ldc]*m8;

  dst[x+(y+1)*ldc] = i10 + i11 + i12 + i13 + i14 + i15 + i16 + i17 + i18;
}
