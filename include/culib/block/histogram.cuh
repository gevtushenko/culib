#ifndef CULIB_BLOCK_HISTOGRAM_CUH_
#define CULIB_BLOCK_HISTOGRAM_CUH_

namespace culib
{
namespace block
{

class default_compressor
{
public:
  template <typename data_type>
  __device__ unsigned int operator () (const data_type &val) const
  {
    return val;
  }
};

class histogram
{
  unsigned int * const cache {};

public:
  histogram () = delete;
  __device__ histogram (
    unsigned int * cache_arg)
    : cache (cache_arg)
  { }

  template <typename data_type, typename compression_operation = default_compressor>
  __device__ void operator() (
    unsigned int bins_count,
    unsigned int n,
    const data_type *in,
    unsigned int *out,
    compression_operation compressor = {})
  {
    clear_cache (bins_count);
    count (n, in, compressor);
    release (bins_count, out);
  }

  template <typename data_type, typename compression_operation = default_compressor>
  __device__ void operator() (
    unsigned int bins_count,
    unsigned int n,
    const data_type *in,
    compression_operation compressor = {})
  {
    clear_cache (bins_count);
    count (n, in, compressor);
  }

  __device__ void clear_cache (unsigned int bins_count)
  {
    for (unsigned int bin = threadIdx.x; bin < bins_count; bin += blockDim.x)
      cache[bin] = {};
    __syncthreads ();
  }

  template <typename data_type, typename compression_operation>
  __device__ void count (unsigned int n, const data_type *in, const compression_operation &compressor)
  {
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x)
      atomicAdd (cache + compressor (in[i]), 1);
    __syncthreads ();
  }

  template <typename data_type, typename compression_operation>
  __device__ void count (
    unsigned int first_element,
    unsigned int elements_count,
    unsigned int stride,
    const data_type *in,
    const compression_operation &compressor)
  {
    for (unsigned int i = first_element; i < elements_count; i += stride)
      atomicAdd (cache + compressor (in[i]), 1);
    __syncthreads ();
  }

  __device__ void release (unsigned int bins_count, unsigned int *out)
  {
    for (unsigned int i = threadIdx.x; i < bins_count; i += blockDim.x)
      out[i] = cache[i];
    __syncthreads ();
  }
};

}
}

#endif