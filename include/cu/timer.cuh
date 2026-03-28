#pragma once

#include <cuda_runtime.h>

#include "cu/check.cuh"

namespace cu {
struct timer {
  cudaEvent_t start{}, stop{};

  timer() {
    check_cuda(cudaEventCreate(&start));
    check_cuda(cudaEventCreate(&stop));
  }

  ~timer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  float measure(auto&& f) {
    check_cuda(cudaEventRecord(start));
    f();
    check_cuda(cudaEventRecord(stop));
    check_cuda(cudaEventSynchronize(stop));

    float ms{};
    check_cuda(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};
} // namespace cu
