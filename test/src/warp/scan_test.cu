//
// Created by egi on 4/4/20.
//

#include "gtest/gtest.h"
#include "test_helper.cuh"

#include "culib/warp/scan.cuh"
#include "culib/utils/placeholder.h"

#include <vector>

template <typename data_type, bool is_shfl_supported>
class inclusive_scanner_helper;

template <typename data_type>
class inclusive_scanner_helper<data_type, false>
{
public:
  __device__ void operator () (data_type const * const in, data_type * const out)
  {
    __shared__ data_type cache[32];
    culib::warp::scan<data_type> scan (cache);
    out[threadIdx.x] = scan.inclusive (in[threadIdx.x]);
  }
};

template <typename data_type>
class inclusive_scanner_helper<data_type, true>
{
public:
  __device__ void operator () (data_type const * const in, data_type * const out)
  {
    culib::warp::scan<data_type> scan;
    out[threadIdx.x] = scan.inclusive (in[threadIdx.x]);
  }
};

template <typename data_type>
class inclusive_scanner
{
public:
  __device__ void operator () (data_type const * const in, data_type * const out)
  {
    inclusive_scanner_helper<data_type, culib::warp::is_shuffle_available<data_type>()> scanner;
    scanner (in, out);
  }
};

template <typename data_type>
void perform_inclusive_scan_test ()
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
    inclusive_scanner<data_type> ());

  for (size_t i = 0; i < warp_size; i++)
    EXPECT_EQ (h_out[i], 1);
}

TEST(warp_scan, inclusive_int) { perform_inclusive_scan_test<int> (); }
// TEST(warp_scan, double) { perform_scan_test<double> (); }

template <typename data_type, bool is_shfl_supported>
class test_exclusive_scanner_helper;

template <typename data_type>
class test_exclusive_scanner_helper<data_type, false>
{
public:
  __device__ void operator () (data_type const * const in, data_type * const out)
  {
    __shared__ data_type cache[32];
    culib::warp::scan<data_type> scan (cache);
    out[threadIdx.x] = scan.exclusive (in[threadIdx.x]);
  }
};

template <typename data_type>
class test_exclusive_scanner_helper<data_type, true>
{
public:
  __device__ void operator () (data_type const * const in, data_type * const out)
  {
    culib::warp::scan<data_type> scan;
    out[threadIdx.x] = scan.exclusive (in[threadIdx.x]);
  }
};

template <typename data_type>
class test_exclusive_scanner
{
public:
  __device__ void operator () (data_type const * const in, data_type * const out)
  {
    test_exclusive_scanner_helper<data_type, culib::warp::is_shuffle_available<data_type>()> scanner;
    scanner (in, out);
  }
};

template <typename data_type>
void perform_exclusive_scan_test ()
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
    test_exclusive_scanner<data_type> ());

  EXPECT_EQ (h_out[0], 0);
  for (size_t i = 1; i < warp_size; i++)
    EXPECT_EQ (h_out[i], 1);
}

TEST(warp_scan, exclusive_int) { perform_exclusive_scan_test<int> (); }

template <typename data_type, bool is_shfl_supported>
class test_exclusive_scanner_max_helper;

template <typename data_type>
class test_exclusive_scanner_max_helper<data_type, false>
{
public:
  __device__ void operator () (data_type const * const in, data_type * const out)
  {
    using namespace culib;
    __shared__ data_type cache[32];
    warp::scan<data_type> scan (cache);
    out[threadIdx.x] = scan.exclusive (in[threadIdx.x], binary_op::max<data_type> {});
  }
};

template <typename data_type>
class test_exclusive_scanner_max_helper<data_type, true>
{
public:
  __device__ void operator () (data_type const * const in, data_type * const out)
  {
    using namespace culib;
    warp::scan<data_type> scan;
    out[threadIdx.x] = scan.exclusive (in[threadIdx.x], binary_op::max<data_type> {});
  }
};

template <typename data_type>
class test_exclusive_scanner_max
{
public:
  __device__ void operator () (data_type const * const in, data_type * const out)
  {
    test_exclusive_scanner_max_helper<data_type, culib::warp::is_shuffle_available<data_type>()> scanner;
    scanner (in, out);
  }
};

template <typename data_type>
void perform_exclusive_scan_max_test ()
{
  constexpr size_t warp_size = 32;
  std::vector<data_type> h_in (warp_size, 0);
  std::vector<data_type> h_out (warp_size);

  h_in[0] = 2;
  h_in[1] = 1;
  h_in[2] = 0;
  h_in[warp_size/2] = 3;

  launch_kernel (
    1 /* blocks */,
    warp_size /* block size */,
    warp_size /* data size */,
    h_in.data (), h_out.data (),
    test_exclusive_scanner_max<data_type> ());

  EXPECT_EQ (h_out[0], 0);
  for (size_t i = 1; i < warp_size / 2 + 1; i++)
    EXPECT_EQ (h_out[i], 2);

  for (size_t i = warp_size / 2 + 1; i < warp_size; i++)
    EXPECT_EQ (h_out[i], 3);
}

TEST(warp_scan, exclusive_max_int) { perform_exclusive_scan_max_test<int> (); }

template <typename data_type>
class exclusive_scanner
{
public:
  __device__ void operator () (data_type const * const in, data_type * const out)
  {
    constexpr size_t warp_size = 32;
    __shared__ culib::utils::placeholder<data_type> cache[warp_size];
    culib::warp::scan<data_type> scan (reinterpret_cast<user_type *> (cache));
    out[threadIdx.x] = scan.exclusive (in[threadIdx.x]);
  }
};

void perform_exclusive_scan_test_for_user_type ()
{
  using data_type = user_type;

  constexpr size_t warp_size = 32;
  std::vector<data_type> h_in (warp_size, user_type {0});
  std::vector<data_type> h_out (warp_size);

  h_in[0].x = 1;

  launch_kernel (
    1 /* blocks */,
    warp_size /* block size */,
    warp_size /* data size */,
    h_in.data (), h_out.data (),
    exclusive_scanner<data_type> ());

  EXPECT_EQ (h_out[0].x, 0);
  for (size_t i = 1; i < warp_size; i++)
    EXPECT_EQ (h_out[i].x, 1);
}

TEST(warp_scan, exclusive_user_type) { perform_exclusive_scan_test_for_user_type (); }
