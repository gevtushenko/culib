//
// Created by egi on 4/4/20.
//

#ifndef CULIB_WARP_SCAN_H
#define CULIB_WARP_SCAN_H

#include <type_traits>

#include "culib/warp/shuffle.cuh"
#include "culib/warp/detail/warp_scan.cuh"
#include "culib/utils/cuda/version.h"
#include "culib/utils/meta/any.cuh"

namespace culib
{
namespace warp
{

/**
* @brief Class for parallel scan within warp
*
* Scan uses a binary combining operator to compute a single aggregate from an array of elements. The
* number of entrant threads must be equal to warpSize.
* The default binary combining operator is the sum.
*
* @tparam data_type Scanned data type
* @tparam scan_policy Policy for warp threads data exchange
*/
template<
  typename data_type,
  typename scan_policy = typename detail::warp_scan_selector<data_type>::implementation>
class scan : public scan_policy
{
public:
  using scan_policy::scan_policy;

  /*!
   @verbatim embed:rst

   :math:`y_{i} = \bigoplus_{j=0}^{i} x_{j}`

   @endverbatim

   @param val Warp local value
   @tparam binary_operation Binary combining function object that will be applied in unspecified order.
                            The behaviour is undefined if binary_operation modifies any element.

   @return Value that would be in ``lane-id`` element of warp array after scan.
   */
  template<typename binary_operation
        //@cond IGNORE
          = binary_op::sum<data_type>
        //@endcond
      >
  __device__
  inline data_type
  inclusive (data_type val, binary_operation binary_op = {})
  {
    return scan_value (val, binary_op);
  }

  /*!
   @verbatim embed:rst

   .. math::
     y_{i} = \bigoplus_{j=0}^{i - 1} x_{j}

   @endverbatim

   @param val Warp local value
   @tparam binary_operation Binary combining function object that will be applied in unspecified order.
                            The behaviour is undefined if binary_operation modifies any element.

    @verbatim embed:rst
      Usage example::

        culib::warp::scan<data_type> scan;
        out[threadIdx.x] = scan.exclusive (in[threadIdx.x]);

     @endverbatim

   @return Value that would be in ``lane-id`` element of warp array after scan.
   */
  template<typename binary_operation = binary_op::sum<data_type>>
  __device__ inline data_type exclusive (data_type val, binary_operation binary_op = {})
  {
    shuffle<data_type> shuffler
      = shared_dependency_injector<data_type, shuffle<data_type>, scan_policy::use_shared_memory>::create (
          scan_policy::get_cache ());
    return shuffler (scan_value (val, binary_op));
  }
};

} // warp
} // culib

#endif // CULIB_WARP_SCAN_H
