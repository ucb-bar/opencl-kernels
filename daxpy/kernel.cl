__kernel void daxpy(__global double *src, __global double *dst, double factor)
{
  long i = get_global_id(0);
  dst[i] += src[i] * factor;
}
