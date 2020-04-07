//
// Created by egi on 4/7/20.
//

#include "gtest/gtest.h"
#include "test_helper.cuh"

#include "culib/warp/reduce.cuh"

#include <vector>
#include <numeric>

template <typename data_type>
void perform_reduce_test ()
{
  constexpr size_t warp_size = 32;
  std::vector<data_type> h_in (warp_size, 0);
  std::vector<data_type> h_out (warp_size);

  std::fill_n (h_in.data (), warp_size, 1);

  launch_kernel (
    1 /* blocks */,
    warp_size /* block size */,
    warp_size /* data size */,
    h_in.data (), h_out.data (),

    [] __device__ (data_type const * const in, data_type * const out)
    {
      culib::warp::reduce<data_type> reduce;
      out[threadIdx.x] = reduce (in[threadIdx.x]);
    });

  for (size_t i = 0; i < warp_size; i++)
    EXPECT_EQ (h_out[i], warp_size);
}

TEST(warp_reduce, int) { perform_reduce_test<int> (); }

