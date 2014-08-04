#ifndef DEVMEM_H
#define DEVMEM_H

template<class T>
class devmem
{
  class {} _[sizeof(T)];
};

template<typename T>
static T *
uncast_devmem(devmem<T> *p)
{
  return reinterpret_cast<T *>(p);
}

template<typename T>
static const T *
uncast_devmem(const devmem<T> *p)
{
  return reinterpret_cast<const T *>(p);
}

#endif
