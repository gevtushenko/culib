#ifndef CULIB_NODE_REDUCE_H
#define CULIB_NODE_REDUCE_H

#include <algorithm>

#include "culib/device/reduce.cuh"
#include "culib/node/node_info.cuh"

namespace culib
{
namespace node
{

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
    size_t current_gpu_elements_count,
    const data_type *input,
    binary_operation binary_op = {})
  {
    culib::device::reducer<data_type> device_reducer (gpu_workspace);
    device_reducer.reduce_from_host (current_gpu_elements_count, input, binary_op);
  }

  static size_t get_gpu_workspace_size (size_t elements_count, const node_info &node)
  {
    return  + node.devices_count;
  }

  static size_t get_gpu_workspace_size_in_bytes (size_t elements_count, const node_info &node)
  {
    return culib::device::reducer<data_type>::get_gpu_workspace_size (elements_count) * sizeof (data_type)
         + node.devices_count * (sizeof (data_type) + 1);
  }
};

} // node
} // culib

#endif
