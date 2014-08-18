#include <cuda_runtime.h>
#include "check.h"
#include "initialize.h"

static cudaDeviceProp
current_device_prop()
{
  int device;
  CUDA_CHECK(cudaGetDevice(&device));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  return prop;
}

void
ispm_initialize()
{
  cudaDeviceProp p = current_device_prop();
  int arch = p.major * 10 + p.minor;
  if (arch != CUDA_ARCH)
  {
    fprintf(stderr, "CUDA device compute capability mismatch: "
            "compiled for sm_%d, got sm_%d\n", CUDA_ARCH, arch);
    abort();
  }
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
  CUDA_CHECK(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
}

DeviceIDString
get_device_string()
{
  cudaDeviceProp p = current_device_prop();

  DeviceIDString s;
  snprintf(s.str, sizeof(s.str), "%04x:%02x:%02x (%s)",
           p.pciDomainID, p.pciBusID, p.pciDeviceID, p.name);

  return s;
}
