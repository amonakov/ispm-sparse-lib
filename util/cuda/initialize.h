#ifndef ISPM_INITIALIZE_H
#define ISPM_INITIALIZE_H

#include <cuda_runtime.h>
#include "check.h"

static void
ispm_initialize()
{
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
  CUDA_CHECK(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
}

#endif
