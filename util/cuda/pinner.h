#ifndef ISPM_PINNER_H
#define ISPM_PINNER_H

#include <cuda_runtime.h>
#include "check.h"

template<typename T>
class cuda_pinned_region
{
  T *ptr;
  devmem<T> *devptr;

  cuda_pinned_region(const cuda_pinned_region &);
  void operator=(const cuda_pinned_region &);
public:
  cuda_pinned_region() :
    ptr(0), devptr(0)
  {}
  cuda_pinned_region(T *ptr, size_t n_elts) :
    ptr(ptr), devptr(0)
  {
    CUDA_CHECK(cudaHostRegister(ptr, n_elts * sizeof(T), cudaHostRegisterMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&devptr, ptr, 0));
  }
  ~cuda_pinned_region()
  {
    if (ptr)
      CUDA_CHECK(cudaHostUnregister(ptr));
  }
  devmem<T> *operator()()
  {
    return devptr;
  }
};

#endif
