#ifndef CUDA_TIMER_H
#define CUDA_TIMER_H

#include <cuda_runtime_api.h>

class cudaEventScoped
{
  cudaEvent_t event;

public:
  cudaEventScoped()
  {
    cudaEventCreate(&event);
  }
  ~cudaEventScoped()
  {
    cudaEventDestroy(event);
  }
  void record(cudaStream_t stream = 0)
  {
    cudaEventRecord(event, stream);
  }
  void await()
  {
    cudaEventSynchronize(event);
  }
  cudaEvent_t& operator()()
  {
    return event;
  }
};

struct cudatimer
{
  cudaEventScoped evt_start, evt_stop;

  void start()
  {
    evt_start.record();
  }
  void stop()
  {
    evt_stop.record();
    evt_stop.await();
  }
  double elapsed_seconds()
  {
    float time;
    cudaEventElapsedTime(&time, evt_start(), evt_stop());
    return 1e-3 * time;
  }
};

#endif

