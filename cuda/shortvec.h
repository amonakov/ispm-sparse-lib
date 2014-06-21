/*
   This header file implements C++ wrappers for short vector types recognized
   by CUDA (float2, float4 and the like).

   We need to do it in a quite round-about way since Nvidia CUDA compilers do
   not produce efficient loads/stores for user-defined aligned structs (as of
   version 3.2).  For example,

	struct __align__(8) my_float2
	{
	  float e[2];
	};

	__global__ void foo(my_float2 *a)
	{
	  __shared__ float s[1];
	  my_float2 aa = *a;
	  s[0] += aa.e[0] + aa.e[1];
	}

   Produces two ld.global.f32 instructions instead of a ld.global.v2.f32 insn.
   However, objects of vector types defined in CUDA includes are loaded using
   vector loads thanks to special handling in the compiler (that basically
   recognizes such vector types by name).
*/

#ifndef SHORTVEC_H
#define SHORTVEC_H

/* This template is specialized for each of the supported short vector types.  */
template<typename T, int N> struct nvcc_recog_vec;

/*
   The specialization records the name of the CUDA short vector type (e.g.
   float4) with elements of type T and width W in the `name' field.
*/
#define NVCC_RECOG_VEC_SPEC(T, W, NAME)\
template<> struct nvcc_recog_vec<T, W> \
{                                      \
  typedef NAME name;                   \
}

NVCC_RECOG_VEC_SPEC(unsigned, 1, unsigned);
NVCC_RECOG_VEC_SPEC(unsigned, 2, uint2);
NVCC_RECOG_VEC_SPEC(unsigned, 4, uint4);
NVCC_RECOG_VEC_SPEC(float,    1, float);
NVCC_RECOG_VEC_SPEC(float,    2, float2);
NVCC_RECOG_VEC_SPEC(float,    4, float4);
NVCC_RECOG_VEC_SPEC(double,   1, double);
NVCC_RECOG_VEC_SPEC(double,   2, double2);
NVCC_RECOG_VEC_SPEC(double,   4, double4);

#undef NVCC_RECOG_VEC_SPEC

/* This template is specialized for each of the supported vector lengths (1, 2
   and 4) to provide indexed access to vector elements (the underlying types
   only provide access by .x/.y/.z/.w field designator).  */
template<typename T, int N> struct shortvec_idx_access;

template<typename T> struct shortvec_idx_access<T, 1>
{
  __device__ T &operator()(typename nvcc_recog_vec<T, 1>::name &v, unsigned i)
  {
    return v;
  }
};
template<typename T> struct shortvec_idx_access<T, 2>
{
  __device__ T &operator()(typename nvcc_recog_vec<T, 2>::name &v, unsigned i)
  {
    return i ? v.y : v.x;
  }
};
template<typename T> struct shortvec_idx_access<T, 4>
{
  __device__ T &operator()(typename nvcc_recog_vec<T, 4>::name &v, unsigned i)
  {
    switch (i)
      {
	case 0: return v.x;
	case 1: return v.y;
	case 2: return v.z;
	default: return v.w;
      }
  }
};

/* Implementation of an efficient template vector type using the above
   infrastructure.  */
template<typename T, int N> struct shortvec
{
  /* Type name that is recognized by Nvidia compiler, e.g. float4.  */
  typedef typename nvcc_recog_vec<T, N>::name cuname;
  cuname t;
  __device__ shortvec &operator=(const shortvec &v)
  {
    /* For some reason assigning directly (t = v.t) may produce inefficient
       code.  This workaround seems to work better, though.  */
    cuname tt = v.t;
    t = tt;
    return *this;
  }
  __device__ shortvec &operator=(const cuname &v)
  {
    cuname tt = v;
    t = tt;
    return *this;
  }
  __device__ T &operator[](unsigned i)
  {
    return shortvec_idx_access<T, N>()(t, i);
  }
  __device__ const T &operator[](unsigned i) const
  {
    return shortvec_idx_access<T, N>()(t, i);
  }
};

#endif
