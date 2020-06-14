//
// Created by egi on 4/5/20.
//

#include "gtest/gtest.h"
#include "test_helper.cuh"

#include "culib/warp/compact.cuh"

#include <vector>
#include <numeric>

template <typename data_type, bool is_shfl_available>
class compacter_helper;

template <typename data_type>
class compacter_helper<data_type, false>
{
public:
  __device__ void operator ()(data_type const * const in, data_type * const out)
  {
    __shared__ data_type cache[32];
    culib::warp::compact<data_type> compact (cache);
    out[threadIdx.x] = compact (in[threadIdx.x], [] (const data_type &value) -> bool { return value > 2; });
  }
};

template <typename data_type>
class compacter_helper<data_type, true>
{
public:
  __device__ void operator ()(data_type const * const in, data_type * const out)
  {
    culib::warp::compact<data_type> compact;
    out[threadIdx.x] = compact (in[threadIdx.x], [] (const data_type &value) -> bool { return value > 2; });
  }
};

template <typename data_type>
class compacter
{
public:
  __device__ void operator () (data_type const * const in, data_type * const out)
  {
    compacter_helper<data_type, culib::warp::is_shuffle_available<data_type> ()> helper;
    helper (in, out);
  }
};

template <typename data_type>
void perform_compact_test ()
{
  constexpr size_t warp_size = 32;
  std::vector<data_type> h_in (warp_size, 0);
  std::vector<data_type> h_out (warp_size);
  std::iota (h_in.data (), h_in.data () + warp_size, 0);

  launch_kernel (
    1 /* blocks */,
    warp_size /* block size */,
    warp_size /* data size */,
    h_in.data (), h_out.data (),
    compacter<data_type> ());

  for (size_t i = 0; i < warp_size; i++)
    {
      if (i > 2)
        EXPECT_EQ (h_out[i], i - 3);
      else
        EXPECT_EQ (h_out[i], culib::warp::compact<int>::not_found);
    }
}

TEST(warp_compact, int) { perform_compact_test <int> (); }
