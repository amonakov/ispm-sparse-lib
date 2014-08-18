#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

static void
report_cuda_error(cudaError_t err)
{
  fprintf(stderr, "CUDA error %x: %s\n", err, cudaGetErrorString(err));
  abort();
}

#define CUDA_CHECK(x) ({cudaError_t e = x; if (e) report_cuda_error(e);})

#endif
