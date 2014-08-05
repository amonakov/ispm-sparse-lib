#ifndef VECTOR_H
#define VECTOR_H

#include <vector>
#include <cstring>
#include <driver_types.h>
#include <cuda_runtime.h>

#include "util/cuda/check.h"

struct host_memory_space_tag {};
struct device_memory_space_tag {};

template<typename T, typename memory_space = host_memory_space_tag> class vector;

template<typename T> class vector<T, host_memory_space_tag>
 : public std::vector<T>
{
  typedef std::vector<T> parent;
public:
  vector()
  { }
  vector(size_t n)
   : parent(n)
  { }
  vector(size_t n, T v)
   : parent(n, v)
  { }
  template<typename SpaceSrc>
  vector(const vector<T, SpaceSrc> &v)
   : parent(v.size())
  {
    copy<T, host_memory_space_tag, SpaceSrc>
        (this->data(), v.data(), v.size());
  }
  template<typename SpaceSrc>
  vector<T, host_memory_space_tag> &
  operator=(const vector<T, SpaceSrc> &v)
  {
    this->resize(v.size());
    copy<T, host_memory_space_tag, SpaceSrc>
        (this->data(), v.data(), v.size());
    return *this;
  }
};

template<typename T> class vector<T, device_memory_space_tag>
{
  T *d_ptr;
  size_t n_elts;
  size_t n_reserved;
  bool pinned;
  static const float reserve_factor = 1.1;

  void allocate()
  {
    if (pinned)
      CUDA_CHECK(cudaHostAlloc(&d_ptr, n_reserved * sizeof(T), cudaHostAllocMapped));
    else
      CUDA_CHECK(cudaMalloc(&d_ptr, n_reserved * sizeof(T)));
  }
  void deallocate()
  {
    if (!d_ptr)
      return;
    if (pinned)
      CUDA_CHECK(cudaFreeHost(d_ptr));
    else
      CUDA_CHECK(cudaFree(d_ptr));
  }
public:
  vector():
    d_ptr(NULL),
    n_elts(0),
    n_reserved(0),
    pinned(false)
  {
  }
  vector(size_t n, bool pinned = false):
    n_elts(n),
    n_reserved(reserve_factor*n),
    pinned(pinned)
  {
    allocate();
    if (pinned)
      memset(d_ptr, 0, n_reserved * sizeof(T));
    else
      CUDA_CHECK(cudaMemset(d_ptr, 0, n_reserved * sizeof(T)));
  }
  template<typename SpaceSrc>
  vector(const vector<T, SpaceSrc> &v):
    n_elts(v.size()),
    n_reserved(reserve_factor*n_elts),
    pinned(false)
  {
    allocate();
    copy<T, device_memory_space_tag, SpaceSrc>
        (data(), v.data(), v.size());
  }
  template<typename SpaceSrc>
  vector<T, device_memory_space_tag> &
  operator=(vector<T, SpaceSrc> &v)
  {
    n_elts = v.size();
    if (n_reserved < v.size())
      {
        deallocate();
        n_reserved = reserve_factor*n_elts;
        allocate();
      }
    copy<T, device_memory_space_tag, SpaceSrc>
        (data(), v.data(), v.size());
    return *this;
  }
  ~vector()
  {
    deallocate();
  }
  size_t size() const
  {
    return n_elts;
  }
  const T *data() const
  {
    return d_ptr;
  }
  T *data()
  {
    return d_ptr;
  }
  void swap(vector<T, device_memory_space_tag> &other)
  {
    std::swap(this->d_ptr, other.d_ptr);
    std::swap(this->n_elts, other.n_elts);
    std::swap(this->n_reserved, other.n_reserved);
    std::swap(this->pinned, other.pinned);
  }
};

#endif
