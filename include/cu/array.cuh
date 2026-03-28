#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <random>
#include <vector>

#include "cu/check.cuh"

using f32 = float;
using f16 = __half;

namespace cu {
namespace detail {
  template<typename T>
  struct cuda_deleter {
    void operator()(T* ptr) const {
      if (ptr) cudaFreeAsync(ptr, 0);
    }
  };

  float random(float min, float max) {
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(min, max);
    return dist(gen);
  }
} // namespace detail

template<typename T = f32>
class array {
  std::unique_ptr<T, detail::cuda_deleter<T>> _data;
  std::size_t _size = 0;

 public:
  explicit array(std::size_t n) : _size(n) {
    T* ptr = nullptr;
    check_cuda(cudaMallocAsync(reinterpret_cast<void**>(&ptr), n * sizeof(T), 0), "array alloc");
    _data.reset(ptr);
  }

  static array zeros(std::size_t n) {
    array result(n);
    check_cuda(cudaMemsetAsync(result.data(), 0, result.size_bytes(), 0), "array memset");
    return result;
  }

  static array uniform(std::size_t n, float min = -1.0f, float max = 1.0f) {
    std::vector<T> host(n);
    for (auto& x : host) x = detail::random(min, max);
    return device(host);
  }

  static array device(const std::vector<T>& vec) {
    array result(vec.size());
    check_cuda(
      cudaMemcpyAsync(result.data(), vec.data(), result.size_bytes(), cudaMemcpyHostToDevice, 0),
      "array H2D copy"
    );
    return result;
  }

  array() = default;
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
        cudaMemcpyAsync(result.data(), data(), size_bytes(), cudaMemcpyDeviceToHost, 0),
        "array D2H copy"
      );
      check_cuda(cudaStreamSynchronize(0), "array sync");
    }
    return result;
  }

  void fill_zero() {
    check_cuda(cudaMemsetAsync(data(), 0, size_bytes(), 0), "array fill_zero");
  }
};
} // namespace cu
