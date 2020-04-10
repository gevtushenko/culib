#include "gtest/gtest.h"
#include "test_helper.cuh"

#include "culib/block/reduce.cuh"

#include <vector>
#include <numeric>

template <typename data_type>
void perform_block_reduce_test ()
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
      __shared__ data_type cache[warp_size];
      culib::block::reduce<data_type> reduce (cache);
      out[threadIdx.x] = reduce (in[threadIdx.x]);
    });

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
  ASSERT_TRUE (culib::warp::reduce<data_type>::use_shared_memory == false);
#endif

  for (size_t i = 0; i < warp_size; i++)
    EXPECT_EQ (h_out[i], warp_size);
}

TEST(block_reduce, single_warp_int) { perform_block_reduce_test<int> (); }
