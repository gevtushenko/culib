#ifndef CULIB_DEVICE_REDUCE_H
#define CULIB_DEVICE_REDUCE_H

#include <algorithm>

#include "culib/utils/meta/binary_ops.cuh"
#include "culib/block/reduce.cuh"
#include "culib/warp/utils.cuh"

namespace culib
{
namespace device
{

namespace details
{

constexpr unsigned int max_blocks_count = 1024;
constexpr unsigned int threads_per_block = 256;

template <typename data_type>
__device__ constexpr unsigned int get_device_reduce_cache_size ()
{
  if (culib::warp::is_shuffle_available <data_type> ())
    return 32;
  else
    return threads_per_block;
}

template <typename data_type, typename binary_operation>
__global__ void device_reduce (
  size_t elements_count,
  const data_type *input,
  data_type *workspace,
  binary_operation binary_op)
{
  const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned int stride = blockDim.x * gridDim.x;

  data_type intermediate_value = binary_op.identity ();
  for (size_t element = tid; element < elements_count; element += stride)
    intermediate_value = binary_op (intermediate_value, input[element]);

  constexpr unsigned int cache_elements = get_device_reduce_cache_size<data_type> ();
  __shared__ data_type cache[cache_elements];
  culib::block::reduce<data_type> block_reducer (cache);
  intermediate_value = block_reducer.reduce_to_master_warp (intermediate_value);

  if (threadIdx.x == 0)
    workspace[blockIdx.x] = intermediate_value;
}

}

template <typename data_type>
class reducer
{
  data_type *gpu_workspace {};

public:
  reducer () = delete;
  explicit reducer (data_type *gpu_workspace_arg) : gpu_workspace (gpu_workspace_arg) { }

  /**
   * @tparam binary_operation Binary combining function object thata will be applied in unspecified order.
   *                          The behaviour is undefined if binary_operation modifies any element.
   */
  template<typename binary_operation = binary_op::sum<data_type>>
  inline data_type
  reduce_from_host (
    size_t elements_count,
    const data_type *input,
    binary_operation binary_op = {})
  {
    const auto blocks_count = get_blocks_count (elements_count);
    details::device_reduce<<<blocks_count, details::threads_per_block>>> (elements_count, input, gpu_workspace, binary_op);
    details::device_reduce<<<1, details::threads_per_block>>> (blocks_count, gpu_workspace, gpu_workspace + blocks_count, binary_op);
    data_type result;
    cudaMemcpy (&result, gpu_workspace + blocks_count, sizeof (data_type), cudaMemcpyDeviceToHost);
    return result;
  }

  static size_t get_gpu_workspace_size (size_t elements_count)
  {
    return get_blocks_count (elements_count) + 1;
  }

  static size_t get_gpu_workspace_size_in_bytes (size_t elements_count)
  {
    return get_gpu_workspace_size (elements_count) * sizeof (data_type);
  }

private:
  static size_t get_blocks_count (size_t elements_count)
  {
    return std::min (
      static_cast<size_t> (details::max_blocks_count),
      (elements_count + details::threads_per_block - 1) / details::threads_per_block);
  }
};

} // device
} // culib

#endif
