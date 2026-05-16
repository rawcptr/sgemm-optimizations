#pragma once
#include <cuda_runtime.h>

#include "cu/array.cuh"
#include "cu/check.cuh"

namespace v1 {

__global__ void sgemm_v1(
  const float* const A,
  const float* const B,
  float* const C,
  int M,
  int N,
  int K,
  float alpha = 1,
  float beta = 0.0
) {
  auto col = threadIdx.x + blockDim.x * blockIdx.x;
  auto row = threadIdx.y + blockDim.y * blockIdx.y;
  if (row < M && col < N) {
    float tmp = 0.0f;
    for (std::size_t k{}; k < K; ++k) {
      tmp += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = alpha * tmp + beta * C[row * N + col];
  }
}

struct kernel {
  dim3 grid, block;
  std::size_t M, N, K;

  cu::array<f32> operator()(const float* A, const float* B) const {
    cu::array<f32> C(M * N);
    sgemm_v1<<<grid, block>>>(A, B, C.data(), M, N, K);
    cu::check_last_cuda("v1 launch");
    return C;
  }
};
} // namespace v1