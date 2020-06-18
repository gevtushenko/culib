//
// Created by egi on 6/16/20.
//

#ifndef CULIB_DEVICE_MEMORY_API_H
#define CULIB_DEVICE_MEMORY_API_H

#include <cstddef>
#include <stdexcept>
#include <cuda_runtime.h>

namespace culib
{
namespace device
{

template <typename data_type>
data_type *allocate (std::size_t size) noexcept (false)
{
  data_type *ptr {};
  const size_t memory_to_allocate = size * sizeof (data_type);
  if (cudaMalloc (&ptr, memory_to_allocate) != cudaSuccess)
    throw std::runtime_error (
      "culib: not enough memory to allocate " + std::to_string (memory_to_allocate) + " bytes");
  return ptr;
}

template <typename data_type>
void copy_n (const data_type *src, size_t n, data_type *dst) noexcept (false)
{
  const size_t memory_to_allocate = n * sizeof (data_type);
  if (cudaMemcpy (dst, src, memory_to_allocate, cudaMemcpyDeviceToDevice) != cudaSuccess)
    throw std::runtime_error (
      "culib: failed to copy " + std::to_string (memory_to_allocate) + " bytes");
}

template <typename data_type>
void send_n (const data_type *cpu_src, size_t n, data_type *dst) noexcept (false)
{
  const size_t memory_to_allocate = n * sizeof (data_type);
  if (cudaMemcpy (dst, cpu_src, memory_to_allocate, cudaMemcpyHostToDevice) != cudaSuccess)
    throw std::runtime_error (
      "culib: failed to send " + std::to_string (memory_to_allocate) + " bytes");
}

template <typename data_type>
void recv_n (const data_type *src, size_t n, data_type *cpu_dst) noexcept (false)
{
  const size_t memory_to_allocate = n * sizeof (data_type);
  if (cudaMemcpy (cpu_dst, src, memory_to_allocate, cudaMemcpyDeviceToHost) != cudaSuccess)
    throw std::runtime_error (
      "culib: failed to receive" + std::to_string (memory_to_allocate) + " bytes");
}

template <typename data_type>
data_type recv (const data_type *src) noexcept (false)
{
  data_type tmp;
  recv_n (src, 1, &tmp);
  return tmp;
}

}
}

#endif //CULIB_API_H
