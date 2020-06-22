#include "gtest/gtest.h"
#include "test_helper.cuh"

#include "culib/block/histogram.cuh"

#include <vector>
#include <numeric>

template <typename data_type, unsigned int bins_count>
class block_test_histogram
{
  unsigned int n {};

public:
  block_test_histogram (unsigned int n_arg) : n (n_arg) {}

  __device__ void operator () (data_type const * const in, unsigned int * const out)
  {
    __shared__ unsigned int cache[bins_count];
    culib::block::histogram hist (cache);
    hist (bins_count, n, in, out);
  }
};

template <typename data_type, unsigned int bins_count>
void perform_block_histogram_test ()
{
  constexpr unsigned int threads_in_block = 256;
  constexpr unsigned int n = 1024;
  std::vector<data_type> h_in (n, 1);
  std::vector<unsigned int> h_out (bins_count);

  launch_kernel (
    1 /* blocks */,
    threads_in_block /* block size */,
    n/* in data size */,
    bins_count /* out data size */,
    h_in.data (), h_out.data (),
    block_test_histogram<data_type, bins_count> (n));

  EXPECT_EQ (h_out[0], 0u);
  EXPECT_EQ (h_out[1], n);
  for (size_t i = 2; i < bins_count; i++)
    EXPECT_EQ (h_out[i], 0);
}

TEST(block_histogram, small_int_bins) { perform_block_histogram_test<int, 32> (); }
