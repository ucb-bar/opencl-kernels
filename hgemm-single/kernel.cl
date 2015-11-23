
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void hgemm_single( __global half *A, __global half *B, __global half *C, int ldc)
{
  long i = get_global_id(0);
  long m = get_global_id(1);
  long n = get_global_id(2);
  float a = A[m+n*ldc];
  float b = B[m*ldc+i];
  float c = ((float)C[i+n*ldc]) + a * b;
  C[i+n*ldc] = (half)c;

  //float a = A[(n*ldc)+i];
  //float b = B[m+n*ldc];
  //C[m+n*ldc] = C[m+n*ldc] + a * b;
}
