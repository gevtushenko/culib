//
// Created by egi on 4/7/20.
//

#include "gtest/gtest.h"
#include "test_helper.cuh"

#include "culib/warp/reduce.cuh"

#include <vector>
#include <numeric>

template <typename data_type>
class test_reducer
{
public:
  __device__ void operator () (data_type const * const in, data_type * const out)
  {
    culib::warp::reduce<data_type> reduce;
    out[threadIdx.x] = reduce (in[threadIdx.x]);
  }
};

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
    test_reducer<data_type> ());

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
  ASSERT_TRUE (culib::warp::reduce<data_type>::use_shared_memory == false);
#endif

  for (size_t i = 0; i < warp_size; i++)
    EXPECT_EQ (h_out[i], warp_size);
}

TEST(warp_reduce, int) { perform_reduce_test<int> (); }

template <typename data_type>
class user_reducer
{
public:
  __device__ void operator () (data_type const * const in, data_type * const out)
  {
    constexpr size_t warp_size = 32;
    __shared__ char cache[warp_size * sizeof (user_type)];
    culib::warp::reduce<data_type> reduce (reinterpret_cast<user_type *> (cache));
    out[threadIdx.x] = reduce (in[threadIdx.x]);
  }
};

void perform_reduce_test_for_user_type ()
{
  using data_type = user_type;

  constexpr size_t warp_size = 32;
  std::vector<data_type> h_in (warp_size);
  std::vector<data_type> h_out (warp_size);

  std::fill_n (h_in.data (), warp_size, user_type { 1 });

  launch_kernel (
    1 /* blocks */,
    warp_size /* block size */,
    warp_size /* data size */,
    h_in.data (), h_out.data (),
    user_reducer<data_type> ());

  ASSERT_TRUE (culib::warp::reduce<data_type>::use_shared_memory == true);

  for (size_t i = 0; i < warp_size; i++)
    EXPECT_EQ (h_out[i].x, warp_size);
}

TEST(warp_reduce, user_type) { perform_reduce_test_for_user_type (); }
