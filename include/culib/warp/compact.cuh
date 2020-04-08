//
// Created by egi on 4/5/20.
//

#ifndef CULIB_COMPACT_H
#define CULIB_COMPACT_H

#include <type_traits>

#include "culib/warp/scan.cuh"
#include "culib/utils/cuda/version.h"
#include "culib/utils/meta/any.cuh"

namespace culib
{
namespace warp
{

template<typename data_type>
class compact : public scan<int>
{
public:
  using scan<int>::scan;
  enum { not_found = -1 };

  /*!

  @verbatim embed:rst
  .. note::
    Return ``-1`` for filtered-out elements
  @endverbatim

  @tparam filter_operation
  @param val
  @param filter_op
  */
  template <typename filter_operation>
  __device__ inline int
  operator ()(data_type val, const filter_operation &filter_op)
  {
    const int filter = filter_op (val);
    const int result_position = exclusive (filter);
    return filter ? result_position : not_found;
  }
};

} // warp
} // culib

#endif //CULIB_COMPACT_H
