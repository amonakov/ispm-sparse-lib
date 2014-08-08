#ifndef ISPM_PINNER_H
#define ISPM_PINNER_H

#include <cuda_runtime.h>
#include "check.h"

template<typename T>
static devmem<T> *
host2devmem(T *p)
{
  devmem<T> *r;
  CUDA_CHECK(cudaHostGetDevicePointer(&r, p, 0));
  return r;
}

template<typename T>
static const devmem<T> *
host2devmem(const T *p)
{
  const devmem<T> *r;
  CUDA_CHECK(cudaHostGetDevicePointer(&r, p, 0));
  return r;
}

template<typename T>
class pinmemptr
{
  T *ptr;
  devmem<T> *devptr;
public:
  pinmemptr(T *ptr) :
    ptr(ptr), devptr(host2devmem(ptr))
  {}
  T &operator*()
  {
    return *ptr;
  }
  devmem<T> *operator()()
  {
    return devptr;
  }
};

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
    devptr = host2devmem(ptr);
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
