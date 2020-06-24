#ifndef CULIB_DEVICE_HISTOGRAM_CUH_
#define CULIB_DEVICE_HISTOGRAM_CUH_

#include "culib/block/histogram.cuh"

#include <algorithm>

namespace culib
{
namespace device
{

namespace details
{

constexpr unsigned int max_blocks_count = 1024;
constexpr unsigned int threads_per_block = 256;

template <typename data_type, typename compression_operation>
__global__ void device_histogram_per_block_shared (
  unsigned int bins_count,
  unsigned int elements_count,
  const data_type *input,
  unsigned int *workspace,
  compression_operation compressor)
{
  const unsigned int block_input_offset = blockIdx.x * blockDim.x;
  const unsigned int total_threads = blockDim.x * gridDim.x;

  extern __shared__ unsigned int cache[]; ///< Cache of bins count size
  culib::block::histogram hist (cache);
  hist.clear_cache (bins_count);
  hist.count (block_input_offset + threadIdx.x, elements_count, total_threads, input, compressor);
  hist.release (bins_count, workspace + bins_count * blockIdx.x);
}

__global__ void device_histogram_accumulate (
  unsigned int bins_count,
  unsigned int blocks_count,
  const unsigned int *workspace,
  unsigned int *result)
{
  if (threadIdx.x < bins_count)
    {
      unsigned int total {};
      for (unsigned int block = threadIdx.x; block < blocks_count; block += blockDim.x)
        total += workspace[block * bins_count + threadIdx.x];
      result[threadIdx.x] = total;
    }
}

}

class histogram
{
  unsigned int *gpu_workspace {};

public:
  histogram () = delete;
  explicit histogram (unsigned int *gpu_workspace_arg) : gpu_workspace (gpu_workspace_arg) { }

  template <typename data_type, typename compression_operation = culib::block::default_compressor>
  inline void operator ()
  (
    unsigned int bins_count,
    unsigned int elements_count,
    const data_type *input,
    unsigned int *output,
    compression_operation compressor = {})
  {
    const auto blocks_count = get_blocks_count (elements_count);
    const auto cache_size = bins_count * sizeof (unsigned int);
    details::device_histogram_per_block_shared<<<blocks_count, details::threads_per_block, cache_size>>> (
      bins_count, elements_count, input, gpu_workspace, compressor);
    details::device_histogram_accumulate<<<1, details::threads_per_block>>> (
      bins_count, blocks_count, gpu_workspace, output);
  }

  static size_t get_gpu_workspace_size (unsigned int bins_count, std::size_t elements_count)
  {
    return bins_count * get_blocks_count (elements_count);
  }

private:
  static size_t get_blocks_count (size_t elements_count)
  {
    return std::min (
      static_cast<size_t> (details::max_blocks_count),
      (elements_count + details::threads_per_block - 1) / details::threads_per_block);
  }
};

}
}

#endif