#ifndef ISPM_CUDA_MEMCPY_H
#define ISPM_CUDA_MEMCPY_H

#include <cuda_runtime.h>
#include "devmem.h"
#include "check.h"

struct cudaMemcpyDirection
{
  template<class T>
  cudaMemcpyKind operator()(devmem<T> *, const devmem<T> *)
  {
    return cudaMemcpyDeviceToDevice;
  }
  template<class T>
  cudaMemcpyKind operator()(devmem<T> *, const T *)
  {
    return cudaMemcpyHostToDevice;
  }
  template<class T>
  cudaMemcpyKind operator()(T *, const devmem<T> *)
  {
    return cudaMemcpyDeviceToHost;
  }
};

template<typename T, typename S>
static inline void
copy(T *dst, const S *src, size_t n_elts)

{
  CUDA_CHECK(cudaMemcpy(dst, src, n_elts * sizeof(T),
                        cudaMemcpyDirection()(dst, src)));
}

template<typename T, typename S>
static inline void
copy_async(T *dst, const S *src, size_t n_elts)

{
  CUDA_CHECK(cudaMemcpyAsync(dst, src, n_elts * sizeof(T),
                             cudaMemcpyDirection()(dst, src)));
}

#endif
