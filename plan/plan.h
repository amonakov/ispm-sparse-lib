#ifndef PLAN_PLAN_H
#define PLAN_PLAN_H

#include "formats/formats.h"
#include "formats/conversion.h"
#include "cuda/spmv/kernel-iface.h"
#include "util/cuda/timer.h"

enum spmv_plan_kind {
  PLAN_DEFAULT,
  PLAN_ESTIMATE,
  PLAN_EXHAUSTIVE,
};

enum spmv_plan_flags {
  SPMV_NO_TEXTURE_MEMORY = 001,
  SPMV_PLAN_NO_DEVICE    = 002
};

template<class T, class E = T>
struct spmv_plan
{
  slell_matrix<E, host_memory_space_tag>   *host_matrix;
  slell_matrix<E, device_memory_space_tag> *device_matrix;
  slell_params mparams;
  spmv_launch_params params;

  spmv_plan():
    host_matrix(NULL),
    device_matrix(NULL)
  {}
  spmv_plan(const csr_matrix<E> &m, spmv_plan_kind kind = PLAN_DEFAULT, unsigned flags = 0);
  ~spmv_plan()
  {
    delete host_matrix;
    delete device_matrix;
  }

  void update_device_matrix_coefs()
  {
    device_matrix->elms = host_matrix->elms;
  }

  void upload()
  {
    if (device_matrix)
      delete device_matrix;
    device_matrix = new slell_matrix<E, device_memory_space_tag>(*host_matrix);
  }

  void change_host_matrix(const csr_matrix<E> &m)
  {
    delete host_matrix;
    host_matrix = new slell_matrix<E, host_memory_space_tag>
     (csr_to_slell(m, mparams.S, mparams.H, mparams.V, mparams.D));
  }

  void execute_spmv(vector<T, device_memory_space_tag> &x,
		    vector<T, device_memory_space_tag> &y);

  void execute_spmv(T *xptr, T *yptr);
};

template<class T, class E>
spmv_plan<T, E>::spmv_plan(const csr_matrix<E> &m, spmv_plan_kind kind, unsigned flags)
{
  unsigned Nthreads_def = 512;
  bool usetex = !(flags & SPMV_NO_TEXTURE_MEMORY);
  bool upload = !(flags & SPMV_PLAN_NO_DEVICE);
  host_matrix = NULL;
  device_matrix = NULL;
  switch (kind)
    {
      case PLAN_DEFAULT:
      case PLAN_ESTIMATE:
	  {
	    host_matrix
	     = new slell_matrix<E, host_memory_space_tag>
	     (csr_to_slell_guess_format(m));
	    if (upload)
	      this->upload();
	    unsigned slice_height = (host_matrix->var_height
				? Nthreads_def
				: host_matrix->slice_height);
	    mparams = (slell_params) {
	      host_matrix->slice_height,
	      host_matrix->hblock,
	      host_matrix->var_height,
	      host_matrix->diags
	    };
	    params = (spmv_launch_params) {
	      Nthreads_def,
	      slice_height,
	      1, usetex, usetex};
	    break;
	  }
      case PLAN_EXHAUSTIVE:
	  {
	    slell_params s;
	    spmv_launch_params p;
	    p.use_tex_x = p.use_tex_sliceptr = usetex;
	    double best_t = __builtin_inf();
	    for (s.S = 32; s.S <= 512; s.S <<= 1)
	      for (s.V = 0; s.V <= 1; s.V += 1)
		for (s.D = 0; s.D <= 1; s.D += 1)
		  for (s.H = 1; s.H <= (s.D ? 1 : 2); s.H <<= 1)
		    {
		      slell_matrix<E, host_memory_space_tag> testmtx
		       = csr_to_slell(m, s.S, s.H, s.V, s.D);
		      if (testmtx.n_slices > 65535)
			continue;
		      slell_matrix<E, device_memory_space_tag> d_testmtx(testmtx);
		      vector<T, device_memory_space_tag> d_x(d_testmtx.n_rows), d_y(d_testmtx.n_rows);
		      p.S = s.S;
		      for (p.N = 128; p.N <= 512; p.N <<= 1)
			for (p.H = 1; p.H <= 2; p.H <<= 1)
			  {
			    if (!p.valid())
			      continue;
			    launch_spmv_kernel<T, E>(p, d_testmtx, d_x, d_y);
			    cudatimer t;
			    t.start();
			    launch_spmv_kernel<T, E>(p, d_testmtx, d_x, d_y);
			    t.stop();
			    if (best_t > t.elapsed_seconds())
			      {
				best_t = t.elapsed_seconds();
				params = p;
				mparams = s;
			      }
			  }
		    }
	    host_matrix
	     = new slell_matrix<E, host_memory_space_tag>
	     (csr_to_slell(m, mparams.S, mparams.H, mparams.V, mparams.D));
	    this->upload();
            if (getenv("SPMV_TUNE_RESULT"))
              fprintf(stderr, "[spmv tuning] %g ms:  %g (%g eff, %g csr-equiv) GB/s\n",
                      1e3 * best_t,
                      1e-9 * host_matrix->spmv_bytes() / best_t,
                      1e-9 * host_matrix->spmv_bytes(true) / best_t,
                      1e-9 * m.spmv_bytes() / best_t);
	    break;
	  }
    }
}

template<class T, class E>
void
spmv_plan<T, E>::execute_spmv(vector<T, device_memory_space_tag> &x,
			      vector<T, device_memory_space_tag> &y)
{
  launch_spmv_kernel<T, E>(this->params, *this->device_matrix, x, y);
}

template<class T, class E>
void
spmv_plan<T, E>::execute_spmv(T *xptr, T *yptr)
{
  launch_spmv_kernel<T, E>(this->params, *this->device_matrix, xptr, yptr);
}

#endif
