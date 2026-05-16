#pragma once
#include <cuda_runtime.h>

#include <source_location>
#include <stdexcept>
#include <string>

namespace cu {
struct error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline void check_cuda(
  cudaError_t err, const char* msg = "", std::source_location loc = std::source_location::current()
) {
  if (err != cudaSuccess) {
    throw error(
      std::string(loc.file_name()) + ":" + std::to_string(loc.line()) + " [" + msg + "] "
      + cudaGetErrorString(err)
    );
  }
}

inline void check_last_cuda(
  const char* msg = "", std::source_location loc = std::source_location::current()
) {
  check_cuda(cudaGetLastError(), msg, loc);
}
} // namespace cu
