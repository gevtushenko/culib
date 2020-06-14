//
// Created by egi on 4/7/20.
//

#include "gtest/gtest.h"
#include "test_helper.cuh"

#include "culib/warp/reduce.cuh"

#include <vector>
#include <numeric>

template <typename data_type, bool is_shfl_available>
class test_reducer_helper;

template <typename data_type>
class test_reducer_helper<data_type, false>
{
public:
  __device__ void operator () (data_type const * const in, data_type * const out)
  {
    __shared__ data_type cache[32];
    culib::warp::reduce<data_type> reduce (cache);
    out[threadIdx.x] = reduce (in[threadIdx.x]);
  }
};

template <typename data_type>
class test_reducer_helper<data_type, true>
{
public:
  __device__ void operator () (data_type const * const in, data_type * const out)
  {
    culib::warp::reduce<data_type> reduce;
    out[threadIdx.x] = reduce (in[threadIdx.x]);
  }
};

template <typename data_type>
class test_reducer
{
public:
  __device__ void operator () (data_type const * const in, data_type * const out)
  {
    test_reducer_helper<data_type, culib::warp::is_shuffle_available<data_type>()> reducer;
    reducer (in, out);
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

#if defined(__CUDACC__) // NVCC
  #define CULIB_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
  #define CULIB_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
  #define CULIB_ALIGN(n) __declspec(align(n))
#else
  #error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

template <typename data_type>
struct CULIB_ALIGN(sizeof(data_type)) placeholder
{
  char place[sizeof (data_type) / sizeof (char)];
};

template <typename data_type>
class user_reducer
{
public:
  __device__ void operator () (data_type const * const in, data_type * const out)
  {
    constexpr size_t warp_size = 32;
    __shared__ placeholder<data_type> cache[warp_size];
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
