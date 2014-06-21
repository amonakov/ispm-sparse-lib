#ifndef CUDA_TEXTURE_WRAP_H
#define CUDA_TEXTURE_WRAP_H

/* This template is used to determine the type to be passed as template
   argument for the CUDA texture<> type.  With the notable exception of
   'double' type (and its vector variants), this is always the same as the
   type we want to store.  Since CUDA textures does not support 'double', we
   substitute int2 instead (as recommended in the manuals).  */

template<typename T>
struct tex_elm_type
{
  typedef T type;
};
template<>
struct tex_elm_type<double>
{
  typedef int2 type;
};
template<>
struct tex_elm_type<double2>
{
  typedef int4 type;
};
template<>
struct tex_elm_type<double4>
{
  typedef int4 type;
};

/* This uses the above to implement a generic 'linear texture' type.  */
template<typename T>
struct linear_texture_type
{
  typedef typename tex_elm_type<T>::type stored_type;
  typedef texture<stored_type, 1, cudaReadModeElementType> type;
};

/* We provide one texture of each type (just one is enough for our purposes).
   This template is specialized for each type; its only purpose is to provide
   a reference to the corresponding texture object.  */
template<typename T>
struct texture_for_type;

#define DECLARE_TEXTURE_FOR_TYPE(T)          \
                                             \
template<>                                   \
struct texture_for_type<T>                   \
{                                            \
  static linear_texture_type<T>::type tex;   \
}

#define INSTANTIATE_TEXTURE_FOR_TYPE(T)      \
linear_texture_type<T>::type texture_for_type<T>::tex __attribute__((visibility("internal")))

DECLARE_TEXTURE_FOR_TYPE (unsigned);
DECLARE_TEXTURE_FOR_TYPE (float);
DECLARE_TEXTURE_FOR_TYPE (float2);
DECLARE_TEXTURE_FOR_TYPE (float4);
DECLARE_TEXTURE_FOR_TYPE (double);
DECLARE_TEXTURE_FOR_TYPE (double2);
DECLARE_TEXTURE_FOR_TYPE (double4);

INSTANTIATE_TEXTURE_FOR_TYPE (unsigned);
INSTANTIATE_TEXTURE_FOR_TYPE (float);
INSTANTIATE_TEXTURE_FOR_TYPE (float2);
INSTANTIATE_TEXTURE_FOR_TYPE (float4);
INSTANTIATE_TEXTURE_FOR_TYPE (double);
INSTANTIATE_TEXTURE_FOR_TYPE (double2);
INSTANTIATE_TEXTURE_FOR_TYPE (double4);

#undef DECLARE_TEXTURE_FOR_TYPE
#undef INSTANTIATE_TEXTURE_FOR_TYPE

/* Object of this type are used to bind texture objects to regions of global
   memory.  */
template<typename T>
class linear_texture_binder
{
  typedef typename linear_texture_type<T>::type &tex_rtype;
  tex_rtype tex;
  typedef texture_for_type<T> tex_getter;
public:
  linear_texture_binder(const T *p, unsigned n): tex(tex_getter::tex)
  {
    cudaBindTexture(0, tex, p, n * sizeof(T));
  }
  linear_texture_binder(const vector<T, device_memory_space_tag> &v): tex(tex_getter::tex)
  {
    cudaBindTexture(0, tex, v.data(), v.size() * sizeof(T));
  }
  ~linear_texture_binder()
  {
    cudaUnbindTexture(tex);
  }
};

/* A generic fetch from texture or memory.  */
template<class T>
static inline __device__ T
fetch(typename linear_texture_type<T>::type tex, unsigned i)
{
  return tex1Dfetch(tex, i);
}

template<>
inline __device__ double
fetch(linear_texture_type<double>::type tex, unsigned i)
{
  int2 t = tex1Dfetch(tex, i);
  return __hiloint2double(t.y, t.x);
}

template<>
inline __device__ double2
fetch(linear_texture_type<double2>::type tex, unsigned i)
{
  int4 t = tex1Dfetch(tex, i);
  return make_double2(__hiloint2double(t.y, t.x),
                      __hiloint2double(t.w, t.z));
}

template<>
inline __device__ double4
fetch(linear_texture_type<double4>::type tex, unsigned i)
{
  int4 t = tex1Dfetch(tex, i * 2);
  int4 u = tex1Dfetch(tex, i * 2 + 1);
  return make_double4(__hiloint2double(t.y, t.x),
                      __hiloint2double(t.w, t.z),
                      __hiloint2double(u.y, u.x),
                      __hiloint2double(u.w, u.z));
}

template<class T>
static inline __device__ T
fetch(T *a, unsigned i)
{
  return a[i];
}

template<class T>
static inline __device__ T
fetch(const T *a, unsigned i)
{
  return a[i];
}

#endif
