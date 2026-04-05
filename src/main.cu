#include <cublas_v2.h>
#include <fmt/base.h>

#include "cu/array.cuh"
#include "cu/runner.cuh"
#include "kernels/v1.cuh"
#include "kernels/v2.cuh"
#include "kernels/v3.cuh"
#include "kernels/v4.cuh"
#include "utils.hpp"

// #define DEBUG_ASSERTIONS

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
  v1_kernel v1(dim3{cdiv(N, 32), cdiv(M, 32)}, dim3{32, 32}, M, N, K);
  v2_kernel<16> v2{dim3{cdiv(N, 16), cdiv(M, 16)}, dim3{16, 16}, M, N, K};
  v3_kernel<64, 4, 4> v3{dim3{cdiv(N, 64), cdiv(M, 64)}, dim3{64 / 4, 64 / 4}, M, N, K};
  v4_kernel<64, 4, 4> v4{dim3{cdiv(N, 64), cdiv(M, 64)}, dim3{64 / 4, 64 / 4}, M, N, K};
#ifdef DEBUG_ASSERTIONS
  auto cv1 = v1(a.data(), b.data());
  check_ref("v2", cv1.cpu().data(), c_ref.cpu().data(), M, N);
  auto cv2 = v2(a.data(), b.data());
  check_ref("v2", cv2.cpu().data(), c_ref.cpu().data(), M, N);
  auto cv3 = v3(a.data(), b.data());
  check_ref("v3", cv3.cpu().data(), c_ref.cpu().data(), M, N);
  auto cv4 = v4(a.data(), b.data());
  check_ref("v4", cv4.cpu().data(), c_ref.cpu().data(), M, N);
#endif // !DEBUG_ASSERTIONS
  // clang-format off
  sgemm_bench("cublas", [&]{
    cublasSgemm(
      handle, CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K,
      &alpha, b.data(), N, a.data(), K, &beta, c_ref.data(), N);
    },
    M, N, K
  );
  // clang-format on
  sgemm_bench("v1", [&] { v1(a.data(), b.data()); }, M, N, K);
  sgemm_bench("v2", [&] { v2(a.data(), b.data()); }, M, N, K);
  sgemm_bench("v3", [&] { v3(a.data(), b.data()); }, M, N, K);
  sgemm_bench("v4", [&] { v4(a.data(), b.data()); }, M, N, K);
}

int main() {
  cublasHandle_t handle;
  cublasCreate(&handle);
  run_sgemm(handle, 4096, 4096, 4096);
  cublasDestroy(handle);
}
