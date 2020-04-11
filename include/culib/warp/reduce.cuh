//
// Created by egi on 4/7/20.
//

#ifndef CULIB_WARP_REDUCE_H
#define CULIB_WARP_REDUCE_H

#include <type_traits>

#include "culib/warp/utils.cuh"
#include "culib/warp/shuffle.cuh"
#include "culib/warp/detail/warp_reduce.cuh"
#include "culib/utils/cuda/version.h"
#include "culib/utils/meta/any.cuh"
#include "culib/utils/meta/binary_ops.cuh"

namespace culib
{
namespace warp
{

template<
  typename data_type,
  typename reduce_policy = typename detail::warp_reduce_selector<data_type>::implementation>
class reduce : public reduce_policy
{
public:
  using reduce_policy::reduce_policy;

  /**
   * @tparam binary_operation Binary combining function object thata will be applied in unspecified order.
   *                          The behaviour is undefined if binary_operation modifies any element.
   */
  template<typename binary_operation = binary_op::sum<data_type>>
  __device__
  inline data_type
  operator ()(data_type val, binary_operation binary_op = {})
  {
    return reduce_value (val, binary_op);
  }
};

} // warp
} // culib

#endif // CULIB_WARP_REDUCE_H
