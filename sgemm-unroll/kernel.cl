__kernel void sgemm_unroll( __global float *A, __global float *B, __global float *C, int ldc)
{
  long i = get_global_id(0);
  long m = 4*get_global_id(1);
  long n = 4*get_global_id(2);

//v_4_4_pre
  float C0 = C[i+n*ldc];
  float C1 = C[i+(n+1)*ldc];
  float C2 = C[i+(n+2)*ldc];
  float C3 = C[i+(n+3)*ldc];

//v_4_4
  float a00 = A[m+0+n*ldc];
  float a01 = A[m+1+n*ldc];
  float a02 = A[m+2+n*ldc];
  float a03 = A[m+3+n*ldc];
  float a10 = A[m+0+(n+1)*ldc];
  float a11 = A[m+1+(n+1)*ldc];
  float a12 = A[m+2+(n+1)*ldc];
  float a13 = A[m+3+(n+1)*ldc];
  float a20 = A[m+0+(n+2)*ldc];
  float a21 = A[m+1+(n+2)*ldc];
  float a22 = A[m+2+(n+2)*ldc];
  float a23 = A[m+3+(n+2)*ldc];
  float a30 = A[m+0+(n+3)*ldc];
  float a31 = A[m+1+(n+3)*ldc];
  float a32 = A[m+2+(n+3)*ldc];
  float a33 = A[m+3+(n+3)*ldc];

  float b0 = B[(m+0)*ldc+i];
  float b1 = B[(m+1)*ldc+i];
  float b2 = B[(m+2)*ldc+i];
  float b3 = B[(m+3)*ldc+i];


  C0 += a00 * b0;
  C1 += a10 * b0;
  C2 += a20 * b0;
  C3 += a30 * b0;

  C0 += a01 * b1;
  C1 += a11 * b1;
  C2 += a21 * b1;
  C3 += a31 * b1;

  C0 += a02 * b2;
  C1 += a12 * b2;
  C2 += a22 * b2;
  C3 += a32 * b2;

  C0 += a03 * b3;
  C1 += a13 * b3;
  C2 += a23 * b3;
  C3 += a33 * b3;

//v_4_4_post
  C[i+n*ldc] = C0;
  C[i+(n+1)*ldc] = C1;
  C[i+(n+2)*ldc] = C2;
  C[i+(n+3)*ldc] = C3;
}
