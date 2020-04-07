//
// Created by egi on 4/7/20.
//

#ifndef CULIB_WARP_REDUCE_H
#define CULIB_WARP_REDUCE_H

#include <type_traits>

#include "culib/warp/shuffle.cuh"
#include "culib/warp/detail/warp_reduce.cuh"
#include "culib/utils/cuda/version.h"
#include "culib/utils/meta/any.cuh"

namespace culib
{
namespace warp
{

template<
  typename data_type,
  typename reduce_policy = typename std::conditional<
    std::integral_constant<bool, utils::meta::is_any<data_type,
      int,
      long,
      long long,
      unsigned int,
      unsigned long,
      unsigned long long,
      float,
      double>::value
                                 && utils::cuda::check_compute_capability<300> ()>::value,
    detail::warp_shfl_reduce<data_type>,
    detail::warp_shrd_reduce<data_type>>::type>
class reduce : public reduce_policy
{
public:
  using reduce_policy::reduce_policy;

  /**
   * @tparam _BinaryOperation Binary combining function object thata will be applied in unspecified order.
   *                          The behaviour is undefined if _BinaryOperation modifies any element.
   */
  template<typename binary_operation = detail::default_reduce_binary_op<data_type>>
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
