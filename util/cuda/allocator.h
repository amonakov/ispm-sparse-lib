#ifndef ISPM_ALLOCATOR_H
#define ISPM_ALLOCATOR_H

#include <cuda_runtime.h>
#include "check.h"

template<typename T>
class cuda_device_allocator: public std::allocator<T>
{
  typedef std::allocator<T> parent;
public:
  using typename parent::pointer;
  using typename parent::size_type;
  pointer allocate(size_type n, const void * = 0)
  {
    pointer p;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
  }
  void deallocate(pointer p, size_type)
  {
    CUDA_CHECK(cudaFree(p));
  }
  template<typename O>
  struct rebind
  {
    typedef cuda_device_allocator<O> other;
  };
};
template<typename T1, typename T2>
inline bool operator==(const cuda_device_allocator<T1> &,
                       const cuda_device_allocator<T2> &)
{
  return true;
}
template<typename T1, typename T2>
inline bool operator!=(const cuda_device_allocator<T1> &,
                       const cuda_device_allocator<T2> &)
{
  return false;
}

template<typename T>
class cuda_pinned_allocator: public std::allocator<T>
{
  typedef std::allocator<T> parent;
public:
  using typename parent::pointer;
  using typename parent::size_type;
  pointer allocate(size_type n, const void * = 0)
  {
    pointer p;
    CUDA_CHECK(cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocMapped));
    return p;
  }
  void deallocate(pointer p, size_type)
  {
    CUDA_CHECK(cudaFreeHost(p));
  }
  template<typename O>
  struct rebind
  {
    typedef cuda_pinned_allocator<O> other;
  };
};
template<typename T1, typename T2>
inline bool operator==(const cuda_pinned_allocator<T1> &,
                       const cuda_pinned_allocator<T2> &)
{
  return true;
}
template<typename T1, typename T2>
inline bool operator!=(const cuda_pinned_allocator<T1> &,
                       const cuda_pinned_allocator<T2> &)
{
  return false;
}

#endif
