#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <random>
#include <vector>

#include "cu/check.cuh"

template<typename T>
concept floating_point = std::is_floating_point_v<T> || std::is_same_v<T, __half>;

using f32 = float;
using f16 = __half;

namespace cu {
namespace detail {
  template<typename T>
  struct cuda_deleter {
    cudaStream_t _stream;

    void operator()(T* ptr) const {
      if (ptr) cudaFreeAsync(ptr, _stream);
    }
  };

  float random(float min, float max) {
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(min, max);
    return dist(gen);
  }
} // namespace detail

template<floating_point T = f32>
class array {
  std::unique_ptr<T, detail::cuda_deleter<T>> _data;
  std::size_t _size = 0;
  cudaStream_t _stream;

 public:
  explicit array(std::size_t n, cudaStream_t stream = 0) : _size{n}, _stream{stream} {
    T* ptr = nullptr;
    check_cuda(cudaMallocAsync(reinterpret_cast<void**>(&ptr), n * sizeof(T), _stream), "array alloc");
    _data = {ptr, detail::cuda_deleter<T>{_stream}};
  }

  static array zeros(std::size_t n, cudaStream_t stream = 0) {
    array result(n, stream);
    check_cuda(cudaMemsetAsync(result.data(), 0, result.size_bytes(), result._stream), "array memset");
    return result;
  }

  static array uniform(std::size_t n, float min = -1.0f, float max = 1.0f, cudaStream_t stream = 0) {
    std::vector<T> host(n);
    for (auto& x : host) {
      if constexpr (std::is_same_v<T, f16>) {
        x = __float2half(detail::random(min, max));
      } else {
        x = detail::random(min, max);
      }
    }
    return device(host, stream);
  }

  static array device(const std::vector<T>& vec, cudaStream_t stream = 0) {
    array result(vec.size(), stream);
    check_cuda(
      cudaMemcpyAsync(result.data(), vec.data(), result.size_bytes(), cudaMemcpyHostToDevice, result._stream),
      "array H2D copy"
    );
    return result;
  }

  array() = delete;
  array(array&&) noexcept = default;
  array& operator=(array&&) noexcept = default;
  array(const array&) = delete;
  array& operator=(const array&) = delete;

  T* data() const noexcept {
    return _data.get();
  }

  std::size_t size() const noexcept {
    return _size;
  }

  std::size_t size_bytes() const noexcept {
    return _size * sizeof(T);
  }

  bool empty() const noexcept {
    return _size == 0;
  }

  std::vector<T> cpu() const {
    std::vector<T> result(_size);
    if (_size) {
      check_cuda(
        cudaMemcpyAsync(result.data(), data(), size_bytes(), cudaMemcpyDeviceToHost, _stream),
        "array D2H copy"
      );
      check_cuda(cudaStreamSynchronize(_stream), "array cpu sync");
    }
    return result;
  }

  void fill_zero() {
    check_cuda(cudaMemsetAsync(data(), 0, size_bytes(), _stream), "array fill_zero");
  }
};
} // namespace cu
