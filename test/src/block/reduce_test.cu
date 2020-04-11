#include "gtest/gtest.h"
#include "test_helper.cuh"

#include "culib/block/reduce.cuh"

#include <vector>
#include <numeric>

template <typename data_type, size_t threads_in_block>
void perform_block_reduce_test ()
{
  std::vector<data_type> h_in (threads_in_block, 1);
  std::vector<data_type> h_out (threads_in_block);

  launch_kernel (
    1 /* blocks */,
    threads_in_block /* block size */,
    threads_in_block /* data size */,
    h_in.data (), h_out.data (),

    [] __device__ (data_type const * const in, data_type * const out)
    {
      __shared__ data_type cache[threads_in_block];
      culib::block::reduce<data_type> reduce (cache);
      out[threadIdx.x] = reduce.reduce_to_master_warp (in[threadIdx.x]);
    });

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
  ASSERT_TRUE (culib::warp::reduce<data_type>::use_shared_memory == false);
#endif

  for (size_t i = 0; i < 32; i++)
    EXPECT_EQ (h_out[i], threads_in_block);
}

TEST(block_reduce, single_warp_int) { perform_block_reduce_test<int, 32> (); }
TEST(block_reduce, multiple_warps_int) { perform_block_reduce_test<int, 4 * 32> (); }

template <typename data_type, size_t threads_in_block>
void perform_block_all_reduce_test ()
{
  std::vector<data_type> h_in (threads_in_block, 1);
  std::vector<data_type> h_out (threads_in_block);

  launch_kernel (
    1 /* blocks */,
    threads_in_block /* block size */,
    threads_in_block /* data size */,
    h_in.data (), h_out.data (),

    [] __device__ (data_type const * const in, data_type * const out)
    {
      __shared__ data_type cache[threads_in_block];
      culib::block::reduce<data_type> reduce (cache);
      out[threadIdx.x] = reduce.all_reduce (in[threadIdx.x]);
    });

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
  ASSERT_TRUE (culib::warp::reduce<data_type>::use_shared_memory == false);
#endif

  for (size_t i = 0; i < threads_in_block; i++)
    EXPECT_EQ (h_out[i], threads_in_block);
}

TEST(block_all_reduce, single_warp_int) { perform_block_all_reduce_test<int, 32> (); }
TEST(block_all_reduce, multiple_warps_int) { perform_block_all_reduce_test<int, 4 * 32> (); }

template <typename data_type, size_t threads_in_block>
void perform_block_all_reduce_max_test ()
{
  std::vector<data_type> h_in (threads_in_block, 0);
  std::vector<data_type> h_out (threads_in_block);

  h_in[threads_in_block / 2] = 42;

  launch_kernel (
    1 /* blocks */,
    threads_in_block /* block size */,
    threads_in_block /* data size */,
    h_in.data (), h_out.data (),

    [] __device__ (data_type const * const in, data_type * const out)
    {
      __shared__ data_type cache[threads_in_block];
      culib::block::reduce<data_type> reduce (cache);
      out[threadIdx.x] = reduce.all_reduce (in[threadIdx.x], culib::warp::binary_op::max<data_type> {});
    });

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
  ASSERT_TRUE (culib::warp::reduce<data_type>::use_shared_memory == false);
#endif

  for (size_t i = 0; i < threads_in_block; i++)
    EXPECT_EQ (h_out[i], 42);
}

TEST(block_all_reduce_max, single_warp_int) { perform_block_all_reduce_max_test<int, 32> (); }
TEST(block_all_reduce_max, multiple_warps_int) { perform_block_all_reduce_max_test<int, 4 * 32> (); }
