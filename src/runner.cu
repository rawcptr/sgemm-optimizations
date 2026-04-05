#include <fmt/base.h>
#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "cu/runner.cuh"

void check_ref(
  const char* name, const float* mine, const float* ref, std::size_t M, std::size_t N
) {
  float max_err = 0;
  for (std::size_t i = 0; i < M * N; ++i) {
    max_err = std::max(max_err, std::abs(mine[i] - ref[i]));
  }
  if (max_err > 0.01) {
    fmt::println(
      "{}: max err vs cublas: {:.2e}{}", name, max_err, max_err > 1e-2f ? "  <-- WRONG" : ""
    );
  }
}
