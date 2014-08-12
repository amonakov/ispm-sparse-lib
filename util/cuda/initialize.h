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

struct DeviceIDString {
  char str[32];
};

inline DeviceIDString
get_device_string()
{
  int device;
  CUDA_CHECK(cudaGetDevice(&device));

  char busid[13];
  CUDA_CHECK(cudaDeviceGetPCIBusId(busid, sizeof(busid), device));

  struct cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  DeviceIDString s;
  snprintf(s.str, sizeof(s.str), "%s (%s)", busid, prop.name);

  return s;
}

#endif
