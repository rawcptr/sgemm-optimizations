#pragma once

#define CEIL_DIV(n, d) (((n) + (d) - 1) / (d))

template<typename U, typename V>
constexpr unsigned int cdiv(U n, V d) {
  auto tn = static_cast<unsigned int>(n), td = static_cast<unsigned int>(d);
  return (tn + td - 1) / td;
}

inline double gflops(long long M, long long N, long long K, double sec) {
  return (2.0 * M * N * K) / sec / 1e9;
}

inline double bandwidth_bytes(long long bytes, double sec) {
  return bytes / sec / 1e9;
}
