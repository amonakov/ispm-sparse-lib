#ifndef VECTOR_H
#define VECTOR_H

#include <vector>

#include "cudamemcpy.h"
#include "devmem.h"
#include "allocator.h"

struct host_memory_space_tag {};
struct device_memory_space_tag {};

template<typename T, typename memory_space = host_memory_space_tag>
class vector;

template<typename T>
class vector<T, host_memory_space_tag>
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
  vector(const vector<T, device_memory_space_tag> &v)
   : parent(v.size())
  {
    copy(this->data(), v.data(), v.size());
  }
  vector<T, host_memory_space_tag> &
  operator=(const vector<T, device_memory_space_tag> &v)
  {
    if (this->size() < v.size())
      resize(v.size());
    copy(this->data(), v.data(), v.size());
    return *this;
  }
};

template<typename T>
class vector<T, device_memory_space_tag>
 : private std::vector<devmem<T>, cuda_device_allocator<devmem<T> > >
{
  typedef typename std::vector<devmem<T>, cuda_device_allocator<devmem<T> > >
    parent;
public:
  vector()
  { }
  vector(size_t n)
   : parent(n)
  { }
  vector(const vector<T, host_memory_space_tag> &v)
   : parent(v.size())
  {
    copy(this->data(), v.data(), v.size());
  }
  template<typename TI>
  vector(const TI first, const TI last)
   : parent(last - first)
  {
    assign(first, last);
  }
  template<typename TI>
  void assign(const TI first, const TI last)
  {
    if (first + size() < last)
      resize(last - first);
    copy(this->data(), &*first, last - first);
  }
  template<typename TI>
  void assign_async(const TI first, const TI last)
  {
    if (first + size() < last)
      resize(last - first);
    copy_async(this->data(), &*first, last - first);
  }
  T *udata()
  {
    return uncast_devmem(this->data());
  }
  using parent::data;
  using parent::size;
  using parent::resize;
};

#endif
