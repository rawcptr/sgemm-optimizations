#pragma once
#include <cuda_runtime.h>

#include "cu/array.cuh"
#include "cu/check.cuh"

namespace v4 {
constexpr std::size_t VEC = 4; // float4 width

template<std::size_t SIZE>
using Tile = float[SIZE][SIZE];

// Bounds-safe vectorised load from a row-major matrix.
__device__ inline float4 vec_load(
  const float* mat, std::size_t row, std::size_t col, std::size_t rows, std::size_t cols
) {
  auto within_bounds = row < rows && col + 3 < cols;
  auto aligned = reinterpret_cast<uintptr_t>(&mat[row * cols + col]) % 16 == 0;
  if (within_bounds && aligned) {
    return *reinterpret_cast<const float4*>(&mat[row * cols + col]);
  }
  float4 v{};
  if (row < rows) {
    if (col + 0 < cols) v.x = mat[row * cols + col + 0];
    if (col + 1 < cols) v.y = mat[row * cols + col + 1];
    if (col + 2 < cols) v.z = mat[row * cols + col + 2];
    if (col + 3 < cols) v.w = mat[row * cols + col + 3];
  }
  return v;
}

// Scatter a float4 into four consecutive columns of shared memory.
template<std::size_t TILE>
__device__ inline void smem_store(Tile<TILE>& smem, std::size_t row, std::size_t col, float4 v) {
  smem[row][col + 0] = v.x;
  smem[row][col + 1] = v.y;
  smem[row][col + 2] = v.z;
  smem[row][col + 3] = v.w;
}

// Cooperatively load a TILE×TILE sub-matrix into shared memory.
template<std::size_t TILE>
__device__ inline void load_tile(
  const float* mat,
  Tile<TILE>& smem,
  std::size_t ty,
  std::size_t tx,
  std::size_t row,
  std::size_t col,
  std::size_t n_rows,
  std::size_t n_cols
) {
  for (std::size_t r = ty; r < TILE; r += blockDim.y) {
    for (std::size_t c = tx * VEC; c < TILE; c += blockDim.x * VEC) {
      smem_store<TILE>(smem, r, c, vec_load(mat, row + r, col + c, n_rows, n_cols));
    }
  }
}

// Accumulate one (As, Bs) tile pair into thread-local register accumulators.
template<std::size_t TILE, std::size_t TM, std::size_t TN>
__device__ inline void mma_tile(
  float (&acc)[TM][TN], const Tile<TILE>& As, const Tile<TILE>& Bs, std::size_t ty, std::size_t tx
) {
#pragma unroll
  for (std::size_t k = 0; k < TILE; ++k) {
#pragma unroll
    for (std::size_t i = 0; i < TM; ++i) {
#pragma unroll
      for (std::size_t j = 0; j < TN; ++j) {
        acc[i][j] += As[ty * TM + i][k] * Bs[k][tx * TN + j];
      }
    }
  }
}

// Bounds-safe write of one output row-strip.
template<std::size_t TN>
__device__ inline void store_row(
  float* C,
  const float (&acc_row)[TN],
  std::size_t row,
  std::size_t col0,
  std::size_t M,
  std::size_t N,
  float alpha,
  float beta
) {
  if (row >= M) return;
#pragma unroll
  for (std::size_t j = 0; j < TN; j += VEC) {
    std::size_t col = col0 + j;
    if (col + 3 < N) {
      float4 out = {
        alpha * acc_row[j + 0] + beta * C[row * N + col + 0],
        alpha * acc_row[j + 1] + beta * C[row * N + col + 1],
        alpha * acc_row[j + 2] + beta * C[row * N + col + 2],
        alpha * acc_row[j + 3] + beta * C[row * N + col + 3],
      };
      *reinterpret_cast<float4*>(&C[row * N + col]) = out;
    } else {
#pragma unroll
      for (std::size_t k = 0; k < VEC; ++k) {
        if (col + k < N) {
          C[row * N + col + k] = alpha * acc_row[j + k] + beta * C[row * N + col + k];
        }
      }
    }
  }
}

template<std::size_t TILE, std::size_t TM, std::size_t TN>
__global__ void sgemm_v4(
  const float* const A,
  const float* const B,
  float* C,
  std::size_t M,
  std::size_t K,
  std::size_t N,
  float alpha = 1.0f,
  float beta = 0.0f
) {
  static_assert(TILE % TM == 0);
  static_assert(TILE % TN == 0);
  static_assert(TILE % VEC == 0, "TILE must be divisible by VEC (4)");
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  float acc[TM][TN]{};
  const std::size_t tx = threadIdx.x, ty = threadIdx.y;
  const std::size_t row = blockIdx.y * TILE;
  const std::size_t col = blockIdx.x * TILE;
  for (std::size_t tile{}; tile < K; tile += TILE) {
    load_tile<TILE>(A, As, ty, tx, row, tile, M, K);
    load_tile<TILE>(B, Bs, ty, tx, tile, col, K, N);
    __syncthreads();
    mma_tile<TILE, TM, TN>(acc, As, Bs, ty, tx);
    __syncthreads();
  }
#pragma unroll
  for (std::size_t i{}; i < TM; ++i) {
    // clang-format off
    store_row<TN>(C, acc[i], row + ty * TM + i, col + tx * TN, M, N, alpha, beta);
    // clang-format off
  }
}

template<const std::size_t TILE = 64, const std::size_t TM = 8, const std::size_t TN = 8>
struct kernel {
  dim3 grid, block;
  std::size_t M, N, K;
  float alpha = 1.0, beta = 0.0f;

  cu::array<float> operator()(const float* const A, const float* const B) const {
    auto C = cu::array<float>::zeros(M * N) ;
    sgemm_v4<TILE, TM, TN><<<grid, block>>>(A, B, C.data(), M, K, N);
    cu::check_last_cuda("v4 launch");
    return C;
  }
};
}
