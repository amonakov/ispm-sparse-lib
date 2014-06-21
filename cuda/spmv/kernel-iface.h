#ifndef SPMV_GPU_H
#define SPMV_GPU_H

#define SPMV_PARAMS_VALID_P(N, S, H) ((N) >= (S) / (H) && (S) >= (H))

struct spmv_launch_params
{
  unsigned N;
  unsigned S;
  unsigned H;
  bool use_tex_sliceptr;
  bool use_tex_x;

  bool valid() const
  {
    return SPMV_PARAMS_VALID_P(N, S, H);
  }
};

template<class T, class E>
void
launch_spmv_kernel(spmv_launch_params &params,
		   slell_matrix<E, device_memory_space_tag> &m,
		   vector<T, device_memory_space_tag> &x,
		   vector<T, device_memory_space_tag> &y);

extern template
void
launch_spmv_kernel<float, float>(spmv_launch_params &params,
				 slell_matrix<float, device_memory_space_tag> &m,
				 vector<float, device_memory_space_tag> &x,
				 vector<float, device_memory_space_tag> &y);

extern template
void
launch_spmv_kernel<double, float>(spmv_launch_params &params,
				 slell_matrix<float, device_memory_space_tag> &m,
				 vector<double, device_memory_space_tag> &x,
				 vector<double, device_memory_space_tag> &y);

extern template
void
launch_spmv_kernel<double, double>(spmv_launch_params &params,
				 slell_matrix<double, device_memory_space_tag> &m,
				 vector<double, device_memory_space_tag> &x,
				 vector<double, device_memory_space_tag> &y);

template<class T, class E>
void
launch_spmv_kernel(spmv_launch_params &params,
		   slell_matrix<E, device_memory_space_tag> &m,
		   T *xptr,
		   T *yptr);

extern template
void
launch_spmv_kernel<float, float>(spmv_launch_params &params,
				 slell_matrix<float, device_memory_space_tag> &m,
				 float *xptr,
				 float *yptr);

extern template
void
launch_spmv_kernel<double, float>(spmv_launch_params &params,
				 slell_matrix<float, device_memory_space_tag> &m,
				 double *xptr,
				 double *yptr);

extern template
void
launch_spmv_kernel<double, double>(spmv_launch_params &params,
				 slell_matrix<double, device_memory_space_tag> &m,
				 double *xptr,
				 double *yptr);

#endif
