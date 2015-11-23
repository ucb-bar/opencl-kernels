

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void sdgemm_single( __global float *A, __global float *B, __global float *C, int ldc)
{
  long i = get_global_id(0);
  long m = get_global_id(1);
  long n = get_global_id(2);
  double a = A[m+n*ldc];
  double b = B[m*ldc+i];
  double c = ((double)C[i+n*ldc]) + a * b;
  C[i+n*ldc] = (float)c;

  //float a = A[(n*ldc)+i];
  //float b = B[m+n*ldc];
  //C[m+n*ldc] = C[m+n*ldc] + a * b;
}
