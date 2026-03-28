#pragma once
#include <cuda_runtime.h>
#include <fmt/base.h>
#include <fmt/format.h>

#include <algorithm>
#include <vector>

#include "cu/check.cuh"
#include "cu/timer.cuh"

// Correctness check: compare kernel output against a reference on the host.
void check_ref(const float* mine, const float* ref, std::size_t M, std::size_t N);

// Benchmark: warmup + timed iterations, prints GFLOP/s and GB/s.
template<typename F>
void sgemm_bench(
  const char* name,
  F&& kernel_fn,
  std::size_t M,
  std::size_t N,
  std::size_t K,
  int iters = 20,
  int warmup = 5
) {
  for (int i = 0; i < warmup; ++i) kernel_fn();
  cudaDeviceSynchronize();

  cu::timer timer;
  std::vector<float> times{};
  times.reserve(iters);

  for (int i = 0; i < iters; ++i) times.push_back(timer.measure([&] { kernel_fn(); }));

  float min_ms = *std::min_element(times.begin(), times.end());
  float mean_ms = 0;
  for (float t : times) mean_ms += t;
  mean_ms /= iters;

  double mean_s = mean_ms * 1e-3;
  long long flops = 2LL * static_cast<long long>(M) * N * K;
  long long bytes = static_cast<long long>(sizeof(float)) * (M * K + K * N + M * N);

  fmt::println(
    "{} mean {:.3f} ms | min {:.3f} ms | {:.1f} GFLOP/s | {:.1f} GB/s",
    name,
    mean_ms,
    min_ms,
    flops / mean_s / 1e9,
    bytes / mean_s / 1e9
  );
}
