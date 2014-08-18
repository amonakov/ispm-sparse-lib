#include <cuda_runtime.h>
#include "check.h"
#include "initialize.h"

void
ispm_initialize()
{
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
  CUDA_CHECK(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
}

DeviceIDString
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
