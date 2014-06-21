#include "cuda/spmv/dispatch.h"

template
void
launch_spmv_kernel<double, float>(spmv_launch_params &params,
				 slell_matrix<float, device_memory_space_tag> &m,
				 vector<double, device_memory_space_tag> &x,
				 vector<double, device_memory_space_tag> &y);

template
void
launch_spmv_kernel<double, float>(spmv_launch_params &params,
				 slell_matrix<float, device_memory_space_tag> &m,
				 double *xptr,
				 double *yptr);
