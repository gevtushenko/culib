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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
  ASSERT_TRUE (culib::warp::reduce<data_type>::use_shared_memory == false);
#endif

  for (size_t i = 0; i < warp_size; i++)
    EXPECT_EQ (h_out[i], warp_size);
}

TEST(warp_reduce, int) { perform_reduce_test<int> (); }

class user_type
{
public:
  unsigned long long int x {};
  unsigned long long int y {};

public:
  user_type () = default;
  explicit __device__ __host__ user_type (int i) : x (i), y (0ull) {}
  __device__ __host__ user_type (unsigned long long int x_arg, unsigned long long int y_arg) : x (x_arg), y (y_arg) {}

  friend bool operator !=(const user_type &lhs, const user_type &rhs)
  {
    return lhs.x != rhs.x || lhs.y != rhs.y;
  }

  friend __device__ user_type operator+ (const user_type &lhs, const user_type &rhs)
  {
    return user_type (lhs.x + rhs.x, lhs.y + rhs.y);
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

    [] __device__ (data_type const * const in, data_type * const out)
    {
      __shared__ char cache[warp_size * sizeof (user_type)];
      culib::warp::reduce<data_type> reduce (reinterpret_cast<user_type *> (cache));
      out[threadIdx.x] = reduce (in[threadIdx.x]);
    });

  ASSERT_TRUE (culib::warp::reduce<data_type>::use_shared_memory == true);

  for (size_t i = 0; i < warp_size; i++)
    EXPECT_EQ (h_out[i].x, warp_size);
}

TEST(warp_reduce, user_type) { perform_reduce_test_for_user_type (); }
