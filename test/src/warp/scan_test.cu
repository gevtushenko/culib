//
// Created by egi on 4/4/20.
//

#include "gtest/gtest.h"
#include "test_helper.cuh"

#include "culib/warp/scan.cuh"

#include <vector>
#include <numeric>

template <typename data_type>
void perform_scan_test ()
{
  constexpr size_t warp_size = 32;
  std::vector<data_type> h_in (warp_size);
  std::vector<data_type> h_out (warp_size);
  std::iota (h_in.begin (), h_in.end (), 0);

  launch_kernel (
    1 /* blocks */,
    warp_size /* block size */,
    warp_size /* data size */,
    h_in.data (), h_out.data (),

    [] __device__ (data_type const * const in, data_type * const out)
    {
      out[threadIdx.x] = in[threadIdx.x];
    });

  EXPECT_EQ (0, 1);
}

TEST(warp_scan, int) { perform_scan_test<int> (); }
TEST(warp_scan, double) { perform_scan_test<double> (); }
