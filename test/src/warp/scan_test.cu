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
  std::vector<data_type> h_in (warp_size, 0);
  std::vector<data_type> h_out (warp_size);

  h_in[0] = 1;

  launch_kernel (
    1 /* blocks */,
    warp_size /* block size */,
    warp_size /* data size */,
    h_in.data (), h_out.data (),

    [] __device__ (data_type const * const in, data_type * const out)
    {
      culib::warp::scan<data_type> scan;
      out[threadIdx.x] = scan (in[threadIdx.x]);
    });

  for (size_t i = 0; i < warp_size; i++)
    EXPECT_EQ (h_out[i], 1);
}

TEST(warp_scan, int) { perform_scan_test<int> (); }
// TEST(warp_scan, double) { perform_scan_test<double> (); }
