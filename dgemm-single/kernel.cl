#pragma OPENCL EXTENSION cl_khr_fp64 : enable


__kernel void dgemm_single( __global double *A, __global double *B, __global double *C, int ldc)
{
  long i = get_global_id(0);
  long m = get_global_id(1);
  long n = get_global_id(2);
  double a = A[m+n*ldc];
  double b = B[m*ldc+i];
  C[i+n*ldc] = C[i+n*ldc] + a * b;
}
