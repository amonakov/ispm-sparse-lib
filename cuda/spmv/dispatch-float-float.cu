#include "cuda/spmv/dispatch.h"

template
void
launch_spmv_kernel<float, float>(spmv_launch_params &params,
				 slell_matrix<float, device_memory_space_tag> &m,
				 vector<float, device_memory_space_tag> &x,
				 vector<float, device_memory_space_tag> &y);

template
void
launch_spmv_kernel<float, float>(spmv_launch_params &params,
				 slell_matrix<float, device_memory_space_tag> &m,
				 float *xptr,
				 float *yptr);
