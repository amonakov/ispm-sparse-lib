#include <cuda.h>
#include <stdio.h>

#include "util/cuda/vector.h"
#include "formats/formats.h"
#include "cuda/shortvec.h"
#include "cuda/texture-wrap.h"
#include "cuda/reduce.h"
#include "cuda/spmv/kernel-iface.h"

template<bool use_tex_sliceptr,
	 bool use_tex_x,
	 class T, class E, unsigned N, unsigned S, unsigned H, unsigned W, bool V, bool D>
struct spmv_kernel;

#include "util/pp-for.h"
#include "genkernels.h"

template<typename T, typename E>
struct spmv_params
{
  T *xptr, *yptr;
  slell_matrix<E, device_memory_space_tag> &matr;
  spmv_launch_params &run;

  spmv_params(T *x, T *y,
	      slell_matrix<E, device_memory_space_tag> &m,
	      spmv_launch_params &r)
	      : xptr(x), yptr(y), matr(m), run(r) {}
};

template<class T, class E, unsigned N, unsigned S, unsigned H, unsigned W, bool V, bool D>
static void
launch_spmv_kernel_tex(const spmv_params<T, E> &params)
{
  typedef typename nvcc_recog_vec<T, W>::name TW;
  linear_texture_binder<T> *tex_for_arg;
  linear_texture_binder<TW> *tex_for_argw;
  linear_texture_binder<unsigned> *tex_for_matr;

  if (params.run.use_tex_sliceptr)
    tex_for_matr = new linear_texture_binder<unsigned>(params.matr.slice_ptr);

  if (params.run.use_tex_x)
    {
      tex_for_arg = new linear_texture_binder<T>(params.xptr, params.matr.n_cols);
      if (W > 1)
	tex_for_argw
	 = new linear_texture_binder<TW>((const TW *)params.xptr, params.matr.n_cols / W);
    }

  void (*spmv_kernel_ptr)(const unsigned *, const unsigned *, const E *, const T*, T*);
  if (params.run.use_tex_sliceptr)
    if (params.run.use_tex_x)
      spmv_kernel_ptr = *spmv_kernel<true,  true,  T, E, N, S, H, W, V, D>();
    else
      spmv_kernel_ptr = *spmv_kernel<true,  false, T, E, N, S, H, W, V, D>();
  else
    if (params.run.use_tex_x)
      spmv_kernel_ptr = *spmv_kernel<false, true,  T, E, N, S, H, W, V, D>();
    else
      spmv_kernel_ptr = *spmv_kernel<false, false, T, E, N, S, H, W, V, D>();
  CUDA_CHECK(cudaPeekAtLastError());
  spmv_kernel_ptr<<<params.matr.n_slices, N, N*sizeof(T)>>>(params.matr.slice_ptr.data(),
		                                        params.matr.cols.data(),
					                params.matr.elms.data(),
					                params.xptr,
					                params.yptr);

  CUDA_CHECK(cudaPeekAtLastError());
  if (params.run.use_tex_x)
    {
      delete tex_for_arg;
      if (W > 1)
	delete tex_for_argw;
    }
  if (params.run.use_tex_sliceptr)
    delete tex_for_matr;
}

template<class T, class E, unsigned N, unsigned S, unsigned H, unsigned W, bool V, bool D,
         bool ok = SPMV_PARAMS_VALID_P(N, S, H)>
struct launch_spmv_kernel_p;

template<class T, class E, unsigned N, unsigned S, unsigned H, unsigned W, bool V, bool D>
struct launch_spmv_kernel_p<T, E, N, S, H, W, V, D, false>
{ void operator()(const spmv_params<T, E> &){} };

template<class T, class E, unsigned N, unsigned S, unsigned H, unsigned W, bool V, bool D>
struct launch_spmv_kernel_p<T, E, N, S, H, W, V, D, true>
{
  void operator()(const spmv_params<T, E> &params)
  {
    launch_spmv_kernel_tex<T, E, N, S, H, W, V, D>(params);
  }
};

template<class T, class E, unsigned N, unsigned S, unsigned H, unsigned W, bool V, bool D>
static void
launch_spmv_kernel_c(const spmv_params<T, E> &params)
{
  launch_spmv_kernel_p<T, E, N, S, H, W, V, D> launcher;
  launcher(params);
}

template<class T, class E, unsigned N, unsigned S, unsigned H, unsigned W>
static void
launch_spmv_kernel_vd(const spmv_params<T, E> &params)
{
  if (params.matr.var_height)
    if (params.matr.diags)
      launch_spmv_kernel_c<T, E, N, H, H, W, true, true>(params);
    else
      launch_spmv_kernel_c<T, E, N, H, H, W, true, false>(params);
  else
    if (params.matr.diags)
      launch_spmv_kernel_c<T, E, N, S, H, W, false, true>(params);
    else
      launch_spmv_kernel_c<T, E, N, S, H, W, false, false>(params);
}

template<class T, class E, unsigned N, unsigned S, unsigned H>
static void
launch_spmv_kernel_w(const spmv_params<T, E> &params)
{
  switch (params.matr.hblock)
  {
    case    1: launch_spmv_kernel_vd<T, E, N, S, H, 1>(params); break;
    case    2: launch_spmv_kernel_vd<T, E, N, S, H, 2>(params); break;
    //case    4: launch_spmv_kernel_vd<T, E, N, S, H, 4>(params); break;
    default: return;
  }
}

template<class T, class E, unsigned N, unsigned S>
static void
launch_spmv_kernel_h(const spmv_params<T, E> &params)
{
  switch (params.run.H)
  {
    case    1: launch_spmv_kernel_w<T, E, N, S, 1>(params); break;
    case    2: launch_spmv_kernel_w<T, E, N, S, 2>(params); break;
    default: return;
  }
}

template<class T, class E, unsigned N>
static void
launch_spmv_kernel_s(const spmv_params<T, E> &params)
{
  switch (params.run.S)
  {
#if 0
    case    1: launch_spmv_kernel_h<T, E, N,   1>(params); break;
    case    2: launch_spmv_kernel_h<T, E, N,   2>(params); break;
    case    4: launch_spmv_kernel_h<T, E, N,   4>(params); break;
    case    8: launch_spmv_kernel_h<T, E, N,   8>(params); break;
#endif
    case   16: launch_spmv_kernel_h<T, E, N,  16>(params); break;
    case   32: launch_spmv_kernel_h<T, E, N,  32>(params); break;
    case   64: launch_spmv_kernel_h<T, E, N,  64>(params); break;
    case  128: launch_spmv_kernel_h<T, E, N, 128>(params); break;
    case  256: launch_spmv_kernel_h<T, E, N, 256>(params); break;
    case  512: launch_spmv_kernel_h<T, E, N, 512>(params); break;
    case 1024: launch_spmv_kernel_h<T, E, N,1024>(params); break;
    default: return;
  }
}

template<class T, class E>
static void
launch_spmv_kernel_n(const spmv_params<T, E> &params)
{
  switch (params.run.N)
  {
    case   64: launch_spmv_kernel_s<T, E,  64>(params); break;
    case  128: launch_spmv_kernel_s<T, E, 128>(params); break;
    case  256: launch_spmv_kernel_s<T, E, 256>(params); break;
    case  512: launch_spmv_kernel_s<T, E, 512>(params); break;
    default: return;
  }
}

template<class T, class E>
void
launch_spmv_kernel(spmv_launch_params &params,
		   slell_matrix<E, device_memory_space_tag> &m,
		   vector<T, device_memory_space_tag> &x,
		   vector<T, device_memory_space_tag> &y)
{
  spmv_params<T, E> lparams(x.data(), y.data(), m, params);
  if (!params.valid())
    return;
  launch_spmv_kernel_n<T, E>(lparams);
}

template<class T, class E>
void
launch_spmv_kernel(spmv_launch_params &params,
		   slell_matrix<E, device_memory_space_tag> &m,
		   T *xptr,
		   T *yptr)
{
  spmv_params<T, E> lparams(xptr, yptr, m, params);
  if (!params.valid())
    return;
  launch_spmv_kernel_n<T, E>(lparams);
}
