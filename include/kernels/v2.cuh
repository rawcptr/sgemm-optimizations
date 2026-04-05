#pragma once
#include <cuda_runtime.h>

#include "cu/array.cuh"
#include "cu/check.cuh"

template<const std::size_t TILE>
__global__ void sgemm_v2(
  const float* const A,
  const float* const B,
  float* const C,
  int M,
  int N,
  int K,
  float alpha = 1,
  float beta = 0.0
) {
  __shared__ float At[TILE][TILE];
  __shared__ float Bt[TILE][TILE];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const auto c_row = blockDim.y * blockIdx.y + ty;
  const auto c_col = blockDim.x * blockIdx.x + tx;

  float dot_product{};

  for (std::size_t tile{}; tile < K; tile += TILE) {
    int a_col = tile + tx;
    int b_row = tile + ty;
    At[ty][tx] = (c_row < M && a_col < K) ? A[c_row * K + a_col] : 0.0f;
    Bt[ty][tx] = (b_row < K && c_col < N) ? B[b_row * N + c_col] : 0.0f;
    __syncthreads();
    for (std::size_t k{}; k < TILE; ++k) {
      dot_product += At[ty][k] * Bt[k][tx];
    }
    __syncthreads();
  }
  if (c_row < M && c_col < N) {
    C[c_row * N + c_col] = alpha * dot_product + beta * C[c_row * N + c_col];
  }
}

template<int TILE = 32>
struct v2_kernel {
  dim3 grid, block;
  std::size_t M, N, K;

  cu::array<f32> operator()(const float* const A, const float* const B) const {
    cu::array<f32> C(M * N);
    sgemm_v2<TILE><<<grid, block>>>(A, B, C.data(), M, N, K);
    cu::check_last_cuda("v2 launch");
    return C;
  }
};
