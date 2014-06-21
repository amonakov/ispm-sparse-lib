#ifndef CUDA_REDUCE_H
#define CUDA_REDUCE_H

/* Implementation of a specialized parallel reduction.

   Each reduce_cols_[no]cond template implements a parallel reduction of N
   elements of type T arranged in S rows along rows using N threads.  Threads
   are numbered in column-major order.  For N = 32 and S = 4:
   0 4  8 12 16 20 24 28       0 + 4 + ... + 28
   1 5  9 13 17 21 25 29   =>  1 + ...
   2 6 10 14 18 22 26 30       2 + ...
   3 7 11 15 19 23 27 31       3 + 7 + ... + 31
   (threads numbered from 0 to S - 1 hold the result).
   */

/* reduce_cols_nocond variant performs reduction unconditionally (even for
   threads that do not contribute to the result).  Useful for N <= warp size.  */
template<class T, unsigned N, unsigned S, bool last = N == S>
struct reduce_cols_nocond;

template<class T, unsigned N, unsigned S>
struct reduce_cols_nocond<T, N, S, true>
{
  __device__ T operator()(volatile T *cols, T cur)
  {
    return cur;
  }
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300 && CUDA_VERSION < 6050
static inline __device__ double
__shfl_down(double v, unsigned n)
{
  int hi = __double2hiint(v), lo = __double2loint(v);
  hi = __shfl_down(hi, n);
  lo = __shfl_down(lo, n);
  return __hiloint2double(hi, lo);
}
#endif

template<class T, unsigned N, unsigned S>
struct reduce_cols_nocond<T, N, S, false>
{
  __device__ T operator()(volatile T *cols, T cur)
  {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
    if (N <= 32)
      {
	cur += __shfl_down(cur, N / 2);
      }
    else
#endif
      {
	if (N > 32) __syncthreads();

	cols[threadIdx.x] = cur;

	if (N > 32) __syncthreads();

	cur += cols[threadIdx.x + N / 2];
      }

    reduce_cols_nocond<T, N / 2, S> r;
    return r(cols, cur);
  }
};

template<class T, unsigned N, unsigned S, bool last = N == S>
struct reduce_cols_cond;

template<class T, unsigned N, unsigned S>
struct reduce_cols_cond<T, N, S, true>
{
  __device__ T operator()(volatile T *cols, T cur)
  {
    return cur;
  }
};

template<class T, unsigned N, unsigned S>
struct reduce_cols_cond<T, N, S, false>
{
  __device__ T operator()(volatile T *cols, T cur)
  {
    if (N <= 32)
      {
	if (threadIdx.x >= N)
	  return cur;
	reduce_cols_nocond<T, N, S> r;
	return r(cols, cur);
      }

    if (N > 32) __syncthreads();

    if (((unsigned)(threadIdx.x - N / 2)) < N / 2)
      cols[threadIdx.x - N / 2] = cur;

    if (N > 32) __syncthreads();

    if (threadIdx.x < N / 2)
      cur += cols[threadIdx.x];

    reduce_cols_cond<T, N / 2, S> r;
    return r(cols, cur);
  }
};

template<class T>
struct reduce_cols_var
{
  __device__ T operator()(volatile T *cols, T cur, unsigned N, unsigned S)
  {
    for (; N > S; N /= 2)
      {
	__syncthreads();

	if (((unsigned)(threadIdx.x - N / 2)) < N / 2)
	  cols[threadIdx.x - N / 2] = cur;

	 __syncthreads();

	if (threadIdx.x < N / 2)
	  cur += cols[threadIdx.x];
      }
    return cur;
  }
};

// TODO: implement reduction via intrinsics


#endif
