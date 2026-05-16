#include <cublas_v2.h>
#include <fmt/base.h>

#include "cu/array.cuh"
#include "cu/runner.cuh"
#include "kernels/v1.cuh"
#include "kernels/v2.cuh"
#include "kernels/v3.cuh"
#include "kernels/v4.cuh"
#include "kernels/v5.cuh"
#include "utils.hpp"

#define DEBUG_ASSERTIONS
#define BENCHMARKING

void run_sgemm(cublasHandle_t handle, std::size_t M, std::size_t N, std::size_t K) {
  auto a = cu::array<f32>::uniform(M * K);
  auto b = cu::array<f32>::uniform(K * N);
  auto c_ref = cu::array<f32>::zeros(M * N);

  const f32 alpha = 1.0f, beta = 0.0f;
  // clang-format off
  auto cublas_sgemm = [&handle, &a, &b, N, M, K, alpha, beta](auto& c) {
    cublasSgemm(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K,
      &alpha,
      b.data(), N,
      a.data(), K,
      &beta,
      c.data(), N
    );
  };
  // clang-format on 
  v1::kernel v1(dim3{cdiv(N, 32), cdiv(M, 32)}, dim3{32, 32}, M, N, K);
  v2::kernel<16> v2{dim3{cdiv(N, 16), cdiv(M, 16)}, dim3{16, 16}, M, N, K};
  v3::kernel<64, 4, 4> v3{dim3{cdiv(N, 64), cdiv(M, 64)}, dim3{64 / 4, 64 / 4}, M, N, K};
  v4::kernel<64, 4, 4> v4{dim3{cdiv(N, 64), cdiv(M, 64)}, dim3{64 / 4, 64 / 4}, M, N, K};
  v5::kernel<64, 8, 64, 4, 4> v5{dim3{cdiv(N, 64), cdiv(M, 64)}, dim3{256}, M, N, K};

#ifdef DEBUG_ASSERTIONS
  cublas_sgemm(c_ref);
  cudaDeviceSynchronize();
  auto cv1 = v1(a.data(), b.data());
  check_ref("cv1", cv1.cpu().data(), c_ref.cpu().data(), M, N);
  auto cv2 = v2(a.data(), b.data());
  check_ref("cv2", cv2.cpu().data(), c_ref.cpu().data(), M, N);
  auto cv3 = v3(a.data(), b.data());
  check_ref("cv3", cv3.cpu().data(), c_ref.cpu().data(), M, N);
  auto cv4 = v4(a.data(), b.data());
  check_ref("cv4", cv4.cpu().data(), c_ref.cpu().data(), M, N);
  auto cv5 = v5(a.data(), b.data());
  check_ref("cv5", cv5.cpu().data(), c_ref.cpu().data(), M, N);
#endif // !DEBUG_ASSERTIONS

#ifdef BENCHMARKING
  sgemm_bench("cublas", [&] { cublas_sgemm(c_ref);}, M, N, K);
  sgemm_bench("v1", [&] { v1(a.data(), b.data()); }, M, N, K);
  sgemm_bench("v2", [&] { v2(a.data(), b.data()); }, M, N, K);
  sgemm_bench("v3", [&] { v3(a.data(), b.data()); }, M, N, K);
  sgemm_bench("v4", [&] { v4(a.data(), b.data()); }, M, N, K);
  sgemm_bench("v5", [&] { v5(a.data(), b.data()); }, M, N, K);
#endif
}

int main() {
  cublasHandle_t handle;
  cublasCreate(&handle);
  run_sgemm(handle, 4096, 4096, 4096);
  cublasDestroy(handle);
}
