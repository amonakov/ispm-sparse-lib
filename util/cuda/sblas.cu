#include <cuda.h>
#include <stdio.h>
#include <assert.h>
#include "cuda/reduce.h"
#include "sblas.h"

namespace sblas
{

template<typename T>
__global__ static void
ddivide_kernel(T *num, T *denom, T *frac)
{
  *frac = fabs(*num / *denom);
}

template<typename T>
void
ddivide(T *num, T *denom, T *frac)
{
  ddivide_kernel<<<1, 1>>>(num, denom, frac);
}


template<typename T, int Nthr>
__global__ static void
copy_indexed_kernel(T *dst, const T *src, const int *i, int n)
{
  int idx = threadIdx.x + Nthr * blockIdx.x;
  if (idx >= n)
    return;
  dst[idx] = src[i[idx]];
}

template<typename T>
void
copy_indexed(T *dst, const T *src, const int *i, int n_elts)
{
  const int n_threads = 256;
  int n_blocks = (n_elts + n_threads - 1) / n_threads;
  copy_indexed_kernel<T, n_threads><<<n_blocks, n_threads>>>(dst, src, i, n_elts);
}


template<typename T, int Nthr>
__global__ static void
add_indexed_kernel(T *dst, const T *src, const int *i, int n)
{
  int idx = threadIdx.x + Nthr * blockIdx.x;
  if (idx >= n)
    return;
  dst[i[idx]] += src[idx];
}

template<typename T>
void
add_indexed(T *dst, const T *src, const int *i, int n_elts)
{
  const int n_threads = 256;
  int n_blocks = (n_elts + n_threads - 1) / n_threads;
  add_indexed_kernel<T, n_threads><<<n_blocks, n_threads>>>(dst, src, i, n_elts);
}


template<typename T, int Nthr>
__global__ static void
xpby_kernel(T *x, const T *x0, const T *y, const T *b, int n)
{
  int idx = threadIdx.x + Nthr * blockIdx.x;
  if (idx >= n)
    return;
  x[idx] = *b * x0[idx] + y[idx];
}

template<typename T>
void
xpby(T *x, const T *x0, const T *y, const T *b, int n_elts)
{
  const int n_threads = 256;
  int n_blocks = (n_elts + n_threads - 1) / n_threads;
  xpby_kernel<T, n_threads><<<n_blocks, n_threads>>>(x, x0, y, b, n_elts);
}


template<typename T, int Nthr>
__global__ static void
sum_kernel(const T *x, T *s, int n)
{
  T sum = 0;
  for (int i = threadIdx.x; i < n; i += Nthr)
    sum += x[i];

  __shared__ T reduce_mem[Nthr / 2];
  reduce_cols_cond<T, Nthr, 1> reduce;
  sum = reduce(reduce_mem, sum);
  if (!threadIdx.x)
    *s = sum;
}

template<typename T, int Nthr>
__global__ static void
sum3_kernel(const T *x1, const T *x2, const T *x3, T *s1, T *s2, T *s3, int n)
{
  T sum1 = 0, sum2 = 0, sum3 = 0;
  for (int i = threadIdx.x; i < n; i += Nthr)
    {
      sum1 += x1[i];
      sum2 += x2[i];
      sum3 += x3[i];
    }

  __shared__ T reduce_mem[Nthr / 2];
  reduce_cols_cond<T, Nthr, 1> reduce;
  sum1 = reduce(reduce_mem, sum1);
  sum2 = reduce(reduce_mem, sum2);
  sum3 = reduce(reduce_mem, sum3);
  if (!threadIdx.x)
    {
      *s1 = sum1;
      *s2 = sum2;
      *s3 = sum3;
    }
}

template<typename T, int Nthr>
__global__ static void
negaxpy_asum_kernel(const T *ap, const T *x, T *y, T *s, int n)
{
  T a = *ap, asum = 0;
  for (int i = threadIdx.x + Nthr * blockIdx.x;
       i < n;
       i += Nthr * gridDim.x)
    {
      T v = y[i] - a * x[i];
      y[i] = v;
      asum += fabs(v);
    }
  __shared__ T reduce_mem[Nthr / 2];
  reduce_cols_cond<T, Nthr, 1> reduce;
  asum = reduce(reduce_mem, asum);
  if (!threadIdx.x)
    s[blockIdx.x] = asum;
}

static int
cuda_full_launch_blocks(int n_threads)
{
  int device;
  struct cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  return prop.maxThreadsPerMultiProcessor / n_threads * prop.multiProcessorCount;
}

template<typename T>
void
negaxpy_asum(const T *a, const T *x, T *y, T *s, int n_elts)
{
  const int n_threads = 256;
  static int n_blocks = cuda_full_launch_blocks(n_threads);
  static T *tmp;
  if (!tmp)
    {
      cudaMalloc(&tmp, n_blocks * sizeof(T));
    }
  negaxpy_asum_kernel<T, n_threads><<<n_blocks, n_threads>>>(a, x, y, tmp, n_elts);
  sum_kernel<T, n_threads><<<1, n_threads>>>(tmp, s, n_blocks);
}

template<typename T, int Nthr>
__global__ static void
dot_kernel(const T *x, const T *y, T *s, int n)
{
  T sum = 0;
  if (x == y)
    for (int i = threadIdx.x + Nthr * blockIdx.x; i < n; i += Nthr * gridDim.x)
      sum += x[i] * x[i];
  else
    for (int i = threadIdx.x + Nthr * blockIdx.x; i < n; i += Nthr * gridDim.x)
      sum += x[i] * y[i];
  __shared__ T reduce_mem[Nthr / 2];
  reduce_cols_cond<T, Nthr, 1> reduce;
  sum = reduce(reduce_mem, sum);
  if (!threadIdx.x)
    s[blockIdx.x] = sum;
}

template<typename T>
void
dot(const T *x, const T *y, T *s, int n_elts)
{
  const int n_threads = 256;
  static int n_blocks = cuda_full_launch_blocks(n_threads);
  static T *tmp;
  if (!tmp)
    {
      cudaMalloc(&tmp, n_blocks * sizeof(T));
    }
  dot_kernel<T, n_threads><<<n_blocks, n_threads>>>(x, y, tmp, n_elts);
  sum_kernel<T, n_threads><<<1, n_threads>>>(tmp, s, n_blocks);
}


template<typename T>
__global__ static void
ppcg_update_scalars_kernel(T *alpha, T *beta, const T *gamma, const T *gammaold, const T *delta)
{
  T g = *gamma;
  T b = g / *gammaold;
  *beta = b;
  *alpha = g / (*delta - b * g / *alpha);
}

template<typename T>
void
ppcg_update_scalars(T *alpha, T *beta, const T *gamma, const T *gammaold, const T *delta)
{
  ppcg_update_scalars_kernel<<<1, 1>>>(alpha, beta, gamma, gammaold, delta);
}


template<typename T, int Nthr>
__global__ static void
ppcg_update_vectors_kernel(T *resnorm_part, T *gamma_part, T *delta_part, const T *alpha, const T *beta,
                         const T *n, const T *m, T *p, T *s, T *q, T *z, T *x, T *r, T *u, T *w, int n_elts)
{
  T a = *alpha, b = *beta, resnorm = 0, gamma = 0, delta = 0;
  for (int i = threadIdx.x + Nthr * blockIdx.x;
       i < n_elts;
       i += Nthr * gridDim.x)
    {
      T zi, qi, pi, si, ri, wi = w[i], ui = u[i];

      z[i] = zi = n[i] + b * z[i];
      q[i] = qi = m[i] + b * q[i];
      s[i] = si = wi   + b * s[i];
      p[i] = pi = ui   + b * p[i];

      u[i] = ui = ui - a * qi;
      w[i] = wi = wi - a * zi;
      r[i] = ri = r[i] - a * si;
      x[i] += a * pi;

      resnorm += fabs(ri);
      gamma   += ri * ui;
      delta   += wi * ui;
    }
  __shared__ T reduce_mem[Nthr / 2];
  reduce_cols_cond<T, Nthr, 1> reduce;
  resnorm = reduce(reduce_mem, resnorm);
  gamma = reduce(reduce_mem, gamma);
  delta = reduce(reduce_mem, delta);
  if (!threadIdx.x)
    {
      resnorm_part[blockIdx.x] = resnorm;
      gamma_part[blockIdx.x] = gamma;
      delta_part[blockIdx.x] = delta;
    }
}

template<typename T>
void
ppcg_update_vectors(T *resnorm, T *gamma, T *delta, const T *alpha, const T *beta,
                  const T *n, const T *m, T *p, T *s, T *q, T *z, T *x, T *r, T *u, T *w, int n_elts)
{
  const int n_threads = 256;
  static int n_blocks = cuda_full_launch_blocks(n_threads);
  static T *tmp1, *tmp2, *tmp3;
  if (!tmp1)
    {
      cudaMalloc(&tmp1, n_blocks * sizeof(T));
      cudaMalloc(&tmp2, n_blocks * sizeof(T));
      cudaMalloc(&tmp3, n_blocks * sizeof(T));
    }
  ppcg_update_vectors_kernel<T, n_threads><<<n_blocks, n_threads>>>
    (tmp1, tmp2, tmp3, alpha, beta, n, m, p, s, q, z, x, r, u, w, n_elts);
  sum3_kernel<T, n_threads><<<1, n_threads>>>(tmp1, tmp2, tmp3, resnorm, gamma, delta, n_blocks);
}


#define TEMPLATE_DECLARATION template
#define T float
#include "sblas.h"
#undef T

#define T double
#include "sblas.h"
#undef T
#undef TEMPLATE_DECLARATION
}
