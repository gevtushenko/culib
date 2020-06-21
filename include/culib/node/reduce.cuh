#ifndef CULIB_NODE_REDUCE_H
#define CULIB_NODE_REDUCE_H

#include <algorithm>

#include "culib/device/reduce.cuh"
#include "culib/node/communication.h"

namespace culib
{
namespace node
{

template <typename data_type>
class reducer
{
  data_type *gpu_workspace {};
  const culib::node::device_communicator &communicator;

public:
  reducer () = delete;
  reducer (
    data_type *gpu_workspace_arg,
    const culib::node::device_communicator &communicator_arg)
    : gpu_workspace (gpu_workspace_arg)
    , communicator (communicator_arg)
    { }

  /**
   * @tparam binary_operation Binary combining function object thata will be applied in unspecified order.
   *                          The behaviour is undefined if binary_operation modifies any element.
   */
  inline data_type
  reduce_from_host (
    size_t current_gpu_elements_count,
    const data_type *input)
  {
    culib::device::reducer<data_type> device_reducer (gpu_workspace);
    return communicator.reduce_sum (device_reducer.reduce_from_host (current_gpu_elements_count, input));
  }

  static size_t get_gpu_workspace_size (size_t elements_count)
  {
    return culib::device::reducer<data_type>::get_blocks_count (elements_count);
  }

  static size_t get_gpu_workspace_size_in_bytes (size_t elements_count)
  {
    return get_gpu_workspace_size (elements_count) * sizeof (data_type);
  }
};

} // node
} // culib

#endif
