/* This file is includes itself multiple times to expand multiple
   specializations of the SpMV kernel with the C Preprocessor.  */

#define JOIN(A, B) A##B
#define RJOIN(A, B) JOIN(A, B)

#define KERNELNAME_BASE spmv_kernel
#define KERNEL_TMPL "kernel.h"

#if !defined(GENKERNELS_1)
#define GENKERNELS_1

#define USE_TEX_SLICEPTR true
#define SLICEARG (texture_for_type<unsigned>::tex)
#define KERNELNAME_1 RJOIN(KERNELNAME_BASE, _st)
#include __FILE__
#undef USE_TEX_SLICEPTR
#undef SLICEARG
#undef KERNELNAME_1

#define USE_TEX_SLICEPTR false
#define SLICEARG sliceptr
#define KERNELNAME_1 RJOIN(KERNELNAME_BASE, _sg)
#include __FILE__
#undef USE_TEX_SLICEPTR
#undef SLICEARG
#undef KERNELNAME_1

#elif !defined(GENKERNELS_2)
#define GENKERNELS_2

#define USE_TEX_X true
#define CLASS_T
#define T_PASS
#define KERNELNAME_2 RJOIN(KERNELNAME_1, _xt)

#define T float
#define XARG  (texture_for_type<float>::tex)
#define XARGW (texture_for_type<typename nvcc_recog_vec<float, W>::name>::tex)
#include __FILE__
#undef XARGW
#undef XARG
#undef T

#define T double
#define XARG  (texture_for_type<double>::tex)
#define XARGW (texture_for_type<typename nvcc_recog_vec<double, W>::name>::tex)
#include __FILE__
#undef XARGW
#undef XARG
#undef T

#undef KERNELNAME_2
#undef T_PASS
#undef CLASS_T
#undef USE_TEX_X

#define USE_TEX_X false
#define CLASS_T class T,
#define T_PASS T,
#define KERNELNAME_2 RJOIN(KERNELNAME_1, _xg)
#define XARG x
#define XARGW ((const typename nvcc_recog_vec<T, W>::name *)x)
#include __FILE__
#undef XARGW
#undef XARG
#undef KERNELNAME_2
#undef T_PASS
#undef CLASS_T
#undef USE_TEX_X
#undef GENKERNELS_2

#else

#define H 1
#define KERNELNAME RJOIN(KERNELNAME_2, _h1)
#include KERNEL_TMPL
#undef KERNELNAME
#undef H

#define H 2
#define KERNELNAME RJOIN(KERNELNAME_2, _h2)
#include KERNEL_TMPL
#undef KERNELNAME
#undef H

#endif

#undef RJOIN
#undef JOIN

#undef KERNELNAME_BASE
#undef KERNEL_TMPL
