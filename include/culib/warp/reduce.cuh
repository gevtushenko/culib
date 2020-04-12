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

/*!
 @verbatim embed:rst
 A reduction uses a binary combining operator to compute a single aggregate from a list of input elements:

 .. math::
   y = \oplus \left( \cdots \oplus \left( \oplus \left( x_{1}, x_{2} \right), x_{3} \right), \cdots, x_{n} \right)

 Every thread in the warp uses thread-local object of this class specialization. Implementation details
 are gathered in ``reduce_policy``. If ``data_type`` is in list of :ref:`supported <is_shuffle_available>` types
 shuffle instruction is used, otherwise shared memory is needed.
 @endverbatim

 @tparam data_type The reduction input/output element type
 @tparam reduce_policy Type dependent reduce implementation.
 */
template<
  typename data_type,
  typename reduce_policy
    // @cond IGNORE
      = typename detail::warp_reduce_selector<data_type>::implementation
    // @endcond
  >
class reduce : public reduce_policy
{
public:
  using reduce_policy::reduce_policy;

  /*!
   * Performs a warp-wide all-reduce in the calling warp. The output is valid in each lane of the warp.
   * The number of entrant threads must be equal to warpSize.
   *
   * @tparam binary_operation Binary combining function object type that will be applied in unspecified order.
   *                          The behaviour is undefined if binary_operation modifies any element.
   * @param val Thread-local value
   * @return Warp-wide result of reduction
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
