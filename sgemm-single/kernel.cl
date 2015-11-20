__kernel void sgemm_single( __global float *A, __global float *B, __global float *C, int ldc)
{
  long i = get_global_id(0);
  long m = get_global_id(1);
  long n = get_global_id(2);
  float a = A[m+n*ldc];
  float b = B[m*ldc+i];
  C[i+n*ldc] = C[i+n*ldc] + a * b;

  //float a = A[(n*ldc)+i];
  //float b = B[m+n*ldc];
  //C[m+n*ldc] = C[m+n*ldc] + a * b;
}
