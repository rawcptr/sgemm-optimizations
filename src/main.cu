#include <cublas_v2.h>
#include <fmt/base.h>

#include "cu/array.cuh"
#include "cu/runner.cuh"
#include "kernels/v2.cuh"
#include "kernels/v3.cuh"
#include "utils.hpp"

void run_sgemm(cublasHandle_t handle, std::size_t M, std::size_t N, std::size_t K) {
  auto a = cu::array<f32>::uniform(M * K);
  auto b = cu::array<f32>::uniform(K * N);
  auto c_ref = cu::array<f32>(M * N);

  // cuBLAS reference (column-major: swap A/B and M/N)
  const f32 alpha = 1.0f, beta = 0.0f;
  // clang-format off
  cublasSgemm(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    N, M, K,
    &alpha,
    b.data(), N,
    a.data(), K,
    &beta,
    c_ref.data(), N
  );
  // clang-format on
  cudaDeviceSynchronize();

  constexpr int TILE = 64;
  constexpr int TM = 4;
  constexpr int TN = 4;

  dim3 grid(cdiv(N, TILE), cdiv(M, TILE));
  // v1_kernel v1{grid, block, int(M), int(N), int(K)};
  // v2_kernel<TILE> v2{grid, dim3{TILE, TILE}, int(M), int(N), int(K)};
  v3_kernel<TILE, TM, TN> v3{grid, dim3{TILE / TN, TILE / TM}, M, N, K};

  auto cv3 = v3(a.data(), b.data());
  // auto cv2 = v2(a.data(), b.data());
  check_ref(cv3.cpu().data(), c_ref.cpu().data(), M, N);
  // check_ref(cv2.cpu().data(), c_ref.cpu().data(), M, N);
  // sgemm_bench("[v2]", [&] { v2(a.data(), b.data()); }, M, N, K);
  sgemm_bench("[v3]", [&] { v3(a.data(), b.data()); }, M, N, K);
  // clang-format off
  sgemm_bench("[cublas]", [&]{
    cublasSgemm(
      handle, CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K,
      &alpha, b.data(), N, a.data(), K, &beta, c_ref.data(), N);
    },
    M, N, K
  );
  // clang-format on
}

int main() {
  cublasHandle_t handle;
  cublasCreate(&handle);
  run_sgemm(handle, 4096, 4096, 4096);
  cublasDestroy(handle);
}
