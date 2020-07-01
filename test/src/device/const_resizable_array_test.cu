#include "gtest/gtest.h"

#include <numeric>

#include <cuda_runtime.h>

#include "culib/device/memory/const_resizable_array.cuh"

using data_type = int;
using array_t = culib::device::const_resizable_array<data_type>;
using result_array_t = culib::device::resizable_array<bool>;

TEST(const_resizable_array, empty_object_destructor)
{
  array_t array;
}

__global__ void simple_const_resizable_array_test (
  unsigned int n,
  culib::device::const_resizeable_array_accessor<data_type> accessor,
  bool *is_data_ok)
{
  if (threadIdx.x < n)
    is_data_ok[threadIdx.x] = accessor[threadIdx.x] == threadIdx.x;
}

TEST(const_resizable_array, test_constant_part_only)
{
  const unsigned int n = 4;
  array_t array (n);
  result_array_t result (n);

  std::unique_ptr<data_type> cpu_data (new data_type[n]);
  std::iota (cpu_data.get (), cpu_data.get () + n, 0);
  array.send_n (cpu_data.get (), n);

  simple_const_resizable_array_test<<<1, n>>> (n, array.get_accessor (), result.get ());
}

TEST(const_resizable_array, test_all_parts_only)
{
  const unsigned int n = 10 * 1024;
  array_t array (n);
  result_array_t result (n);

  std::unique_ptr<data_type> cpu_data (new data_type[n]);
  std::iota (cpu_data.get (), cpu_data.get () + n, 0);
  array.send_n (cpu_data.get (), n);

  simple_const_resizable_array_test<<<1, n>>> (n, array.get_accessor (), result.get ());
}
