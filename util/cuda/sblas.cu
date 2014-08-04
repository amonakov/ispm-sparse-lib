#include <cuda.h>
#include <stdio.h>
#include <assert.h>
#include "cuda/reduce.h"
#include "devmem.h"
#include "vector.h"
#include "sblas.h"

#define DT devmem<T>

namespace sblas
{

#define P(p) uncast_devmem(p)

static const int n_threads = 256;

template<typename T, int Nthr>
__global__ static void
copy_indexed_kernel(T *dst, const T *src, const int *i, int n)
{
  int idx = threadIdx.x + Nthr * blockIdx.x;
  if (idx < n)
    dst[idx] = src[i[idx]];
}

template<typename T>
void
copy_indexed(DT *dst, const DT *src, const devmem<int> *i, int n_elts)
{
  int n_blocks = (n_elts + n_threads - 1) / n_threads;
  copy_indexed_kernel<T, n_threads><<<n_blocks, n_threads>>>(P(dst), P(src), P(i), n_elts);
}


template<typename T, int Nthr>
__global__ static void
add_indexed_kernel(T *dst, const T *src, const int *i, int n)
{
  int idx = threadIdx.x + Nthr * blockIdx.x;
  if (idx < n)
    dst[i[idx]] += src[idx];
}

template<typename T>
void
add_indexed(DT *dst, const DT *src, const devmem<int> *i, int n_elts)
{
  int n_blocks = (n_elts + n_threads - 1) / n_threads;
  add_indexed_kernel<T, n_threads><<<n_blocks, n_threads>>>(P(dst), P(src), P(i), n_elts);
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

template<typename T, int Nthr = n_threads>
class full_launch {
  int blocks_;
  static const int n_vecs = 3;
  vector<T, device_memory_space_tag> vecs[n_vecs];
  full_launch()
  {
    int device;
    struct cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    blocks_ = prop.maxThreadsPerMultiProcessor / Nthr * prop.multiProcessorCount;

    for (int i = 0; i < n_vecs; i++)
      vecs[i].resize(blocks_);
  }
public:
  static full_launch &the()
  {
    static full_launch instance;
    return instance;
  }
  int blocks()
  {
    return blocks_;
  }
  DT *vec(int i = 0)
  {
    return vecs[i].data();
  }
};

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
dot(const DT *x, const DT *y, DT *s, int n_elts)
{
  full_launch<T> &config = full_launch<T>::the();
  int n_blocks = config.blocks();
  DT *tmp = config.vec();
  dot_kernel<T, n_threads><<<n_blocks, n_threads>>>(P(x), P(y), P(tmp), n_elts);
  sum_kernel<T, n_threads><<<1, n_threads>>>(P(tmp), P(s), n_blocks);
}


template<typename T>
__global__ static void
ppcg_update_scalars_kernel(T *alpha, T *beta,
                           const T *gamma, const T *gammaold, const T *delta)
{
  T g = *gamma;
  T b = g / *gammaold;
  *beta = b;
  *alpha = g / (*delta - b * g / *alpha);
}

template<typename T>
void
ppcg_update_scalars(DT *alpha, DT *beta,
                    const DT *gamma, const DT *gammaold, const DT *delta)
{
  ppcg_update_scalars_kernel<<<1, 1>>>
    (P(alpha), P(beta), P(gamma), P(gammaold), P(delta));
}


template<typename T, int Nthr>
__global__ static void
ppcg_update_vectors_kernel(T *resnorm_part, T *gamma_part, T *delta_part,
                           const T *alpha, const T *beta,
                           const T *n, const T *m,
                           T *p, T *s, T *q, T *z, T *x, T *r, T *u, T *w,
                           int n_elts)
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
ppcg_update_vectors(DT *resnorm, DT *gamma, DT *delta,
                    const DT *alpha, const DT *beta,
                    const DT *n, const DT *m,
                    DT *p, DT *s, DT *q, DT *z, DT *x, DT *r, DT *u, DT *w,
                    int n_elts)
{
  full_launch<T> &config = full_launch<T>::the();
  int n_blocks = config.blocks();
  DT *tmp1 = config.vec(0);
  DT *tmp2 = config.vec(1);
  DT *tmp3 = config.vec(2);
  ppcg_update_vectors_kernel<T, n_threads><<<n_blocks, n_threads>>>
    (P(tmp1), P(tmp2), P(tmp3), P(alpha), P(beta),
     P(n), P(m), P(p), P(s), P(q), P(z), P(x), P(r), P(u), P(w), n_elts);
  sum3_kernel<T, n_threads><<<1, n_threads>>>
    (P(tmp1), P(tmp2), P(tmp3), P(resnorm), P(gamma), P(delta), n_blocks);
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
