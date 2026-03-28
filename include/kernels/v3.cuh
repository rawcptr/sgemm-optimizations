#pragma once
#include <cuda_runtime.h>

#include "cu/array.cuh"
#include "cu/check.cuh"

template<std::size_t TILE, std::size_t TM, std::size_t TN>
__global__ void sgemm_v3(
  const float* const A,
  const float* const B,
  float* const C,
  std::size_t M,
  std::size_t K,
  std::size_t N,
  float alpha = 1.0f,
  float beta = 0.0f
) {
  //
  static_assert(TILE % TM == 0);
  static_assert(TILE % TN == 0);
  // set up the shared memroy
  // SMEM is shared per block.
  // the TILE is literally sizeof(block)
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  float acc[TM][TN]{};
  // refers to how far along we we in the current block
  // the Mth thread in Nth tile is responsible for TILE_N[THREADBLOCK_M]
  // so they are the index into the THREADBLOCK (which threadblock am i?)
  auto tx = threadIdx.x; // threadblock's topleft col index
  auto ty = threadIdx.y; // threadblock's topleft row index
  auto block_row = blockIdx.y * TILE; // which TILE->row is this (blockDim.y)
  auto block_col = blockIdx.x * TILE; // which TILE->col is this (blockDim.x)
  // we have tiles and such
  // we want to iterate over tiles for accumulating the dot
  // product into the SMEM
  for (std::size_t tile{}; tile < K; tile += TILE) {
    // load data into SMEM here, each thread loads it's own data into SMEM
    for (std::size_t r{ty}; r < TILE; r += blockDim.y) {
      for (std::size_t c{tx}; c < TILE; c += blockDim.x) {
        auto a_row = block_row + r, a_col = tile + c;
        auto b_row = tile + r, b_col = block_col + c;
        As[r][c] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        Bs[r][c] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
      }
    }
    __syncthreads();
    // compute partial dot product
    for (std::size_t k{}; k < TILE; ++k) {
#pragma unroll
      for (std::size_t i{}; i < TM; ++i) {
#pragma unroll
        for (std::size_t j{}; j < TN; ++j) {
          // the accumulator is obviously laid out in
          // row-major format.
          // indexing into As which is composed of the partial dot
          // products of the current tile, the index for a single
          // accumulator would be the strided by TM because
          // the current thread is responsible for TMxTN memory
          // the stride being TM for the row access and TN for column access
          // we are basically computing a mini tile of TMxTN inside the acc
          // A/Bs stores TILExTILE composed for multiple TM * TN matrices
          // we need to get the matrices out and accumulate them according
          // to the accumulators the thread is responsible for.
          // for each accumulator we go through the entire TMxTN area of
          // the SMEM that contributes to the dot product.
          acc[i][j] += As[ty * TM + i][k] // threads in Y * TM
            * Bs[k][tx * TN + j]; // threads in x * TN
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (std::size_t i{}; i < TM; ++i) {
#pragma unroll
    for (std::size_t j{}; j < TN; ++j) {
      auto row = block_row + ty * TM + i;
      auto col = block_col + tx * TN + j;
      if (row < M && col < N) {
        C[row * N + col] = alpha * acc[i][j] + beta * C[row * N + col];
      }
    }
  }
}

template<const std::size_t TILE = 64, const std::size_t TM = 8, const std::size_t TN = 8>
struct v3_kernel {
  dim3 grid, block;
  std::size_t M, N, K;

  cu::array<float> operator()(const float* const A, const float* const B) const {
    cu::array<float> C(M * N);
    sgemm_v3<TILE, TM, TN><<<grid, block>>>(A, B, C.data(), M, K, N);
    cu::check_last_cuda("v3 launch");
    return C;
  }
};
