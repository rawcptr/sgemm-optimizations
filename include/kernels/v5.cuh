#pragma once
#include <cuda_runtime.h>

#include "cu/array.cuh"
#include "cu/check.cuh"

namespace v5 {
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

// Accumulate one (As, Bs) tile pair into thread-local register accumulators.
template<std::size_t TILE, std::size_t TM, std::size_t TN>
__device__ inline void mma_tile(
  float (&acc)[TM][TN], const Tile<TILE>& As, const Tile<TILE>& Bs, std::size_t ty, std::size_t tx
) {
  // #pragma unroll
  for (std::size_t k = 0; k < TILE; ++k) {
    // #pragma unroll
    for (std::size_t i = 0; i < TM; ++i) {
      // #pragma unroll
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
  // #pragma unroll
  for (std::size_t j = 0; j < TN; j += VEC) {
    std::size_t col = col0 + j;
    auto aligned = reinterpret_cast<uintptr_t>(&C[row * N + col]) % 16 == 0;
    auto within_bounds = col + 3 < N;
    if (aligned && within_bounds) {
      float4 out = {
        alpha * acc_row[j + 0] + beta * C[row * N + col + 0],
        alpha * acc_row[j + 1] + beta * C[row * N + col + 1],
        alpha * acc_row[j + 2] + beta * C[row * N + col + 2],
        alpha * acc_row[j + 3] + beta * C[row * N + col + 3],
      };
      *reinterpret_cast<float4*>(&C[row * N + col]) = out;
    } else {
      // #pragma unroll
      for (std::size_t k = 0; k < VEC; ++k) {
        if (col + k < N) {
          C[row * N + col + k] = alpha * acc_row[j + k] + beta * C[row * N + col + k];
        }
      }
    }
  }
}

template<std::size_t TILE, std::size_t WM, std::size_t WN, std::size_t TM, std::size_t TN>
__global__ void sgemm_v5(
  const float* const A,
  const float* const B,
  float* C,
  std::size_t M,
  std::size_t K,
  std::size_t N,
  float alpha = 1.0f,
  float beta = 0.0f
) {
  static_assert(TILE % WM == 0, "TILE must be multiple of WM");
  static_assert(TILE % WN == 0, "TILE must be multiple of WN");
  static_assert(WM % TM == 0, "WM must be multiple of TM");
  static_assert(WN % TN == 0, "WN must be multiple of TN");
  static_assert(WM * WN == 32 * TM * TN, "Warp tile must map to 32 threads");
  static_assert(TILE % VEC == 0, "TILE must be divisible by 4");

  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  float acc[TM][TN]{};
  const std::size_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  const std::size_t bx = blockIdx.x, by = blockIdx.y;
  const std::size_t tid = tz * blockDim.y * blockDim.x + ty * blockDim.x + tx;
  const std::size_t nthreads = blockDim.x * blockDim.y * blockDim.z;
  const auto warp_id = tid / 32; // how many wraps
  const auto lane_id = tid % 32; // how far along this wrap
  constexpr auto N_WARPS = TILE / WN;
  const auto warp_row = warp_id / N_WARPS;
  const auto warp_col = warp_id % N_WARPS;
  constexpr auto THREADS_IN_WARP_N = WN / TN;
  const auto thread_row_in_warp = lane_id / THREADS_IN_WARP_N;
  const auto thread_col_in_warp = lane_id % THREADS_IN_WARP_N;
  const auto logical_y = (warp_row * (WM / TM)) + thread_row_in_warp;
  const auto logical_x = (warp_col * (WN / TN)) + thread_col_in_warp;
  const auto block_row = by * TILE;
  const auto block_col = bx * TILE;
  for (std::size_t tile{}; tile < K; tile += TILE) {
    for (auto idx{tid}; idx < (TILE * TILE) / VEC; idx += nthreads) {
      auto r = idx / (TILE / VEC);
      auto c = (idx % (TILE / VEC)) * VEC;
      smem_store<TILE>(As, r, c, vec_load(A, block_row + r, tile + c, M, K));
      smem_store<TILE>(Bs, r, c, vec_load(B, tile + r, block_col + c, K, N));
    }
    __syncthreads();
    mma_tile<TILE, TM, TN>(acc, As, Bs, logical_y, logical_x);
    __syncthreads();
  }
  // #pragma unroll
  for (std::size_t i{}; i < TM; ++i) {
    auto row = block_row + warp_row * WM + thread_row_in_warp * TM + i;
    auto col = block_col + warp_col * WN + thread_col_in_warp * TN;
    store_row<TN>(C, acc[i], row, col, M, N, alpha, beta);
  }
}

template<
  std::size_t TILE = 32,
  std::size_t WM = 8,
  std::size_t WN = 64,
  std::size_t TM = 4,
  std::size_t TN = 4>
struct kernel {
  dim3 grid, block;
  std::size_t M, N, K;
  float alpha = 1.0, beta = 0.0f;

  cu::array<float> operator()(const float* const A, const float* const B) const {
    auto C = cu::array<float>::zeros(M * N);
    sgemm_v5<TILE, WM, WN, TM, TN><<<grid, block>>>(A, B, C.data(), M, K, N, alpha, beta);
    cu::check_last_cuda("v5 launch");
    return C;
  }
};

} // namespace v5
