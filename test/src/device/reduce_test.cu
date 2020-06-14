#include "gtest/gtest.h"
#include "test_helper.cuh"

#include <cuda_runtime.h>

#include "culib/device/reduce.cuh"

#include <vector>
#include <numeric>

template <typename data_type>
void perform_device_reduce_test (size_t elements_count)
{
  std::vector<data_type> h_in (elements_count, 1);

  data_type *d_in {};
  data_type *d_workspace {};
  cudaMalloc (&d_in, elements_count * sizeof (data_type));
  cudaMalloc (&d_workspace, culib::device::reducer<data_type>::get_gpu_workspace_size_in_bytes (elements_count));
  cudaMemcpy (d_in, h_in.data (), elements_count * sizeof (data_type), cudaMemcpyHostToDevice);

  culib::device::reducer<data_type> reducer (d_workspace);
  data_type result = reducer.reduce_from_host (elements_count, d_in);

  EXPECT_EQ (result, elements_count);

  cudaFree (d_workspace);
  cudaFree (d_in);
}

TEST(device_reduce, multiple_blocks_int) { perform_device_reduce_test<int> (10000); }
