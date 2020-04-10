//
// Created by egi on 4/5/20.
//

#ifndef CULIB_UTILS_H
#define CULIB_UTILS_H

#include "culib/utils/cuda/version.h"
#include "culib/utils/meta/any.cuh"

namespace culib
{
namespace warp
{

inline __device__
unsigned int get_full_wark_mask ()
{
  return 0xffffffff;
}

inline __device__
unsigned int lane_id ()
{
  unsigned int ret;
  asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
  return ret;
}

template <typename data_type>
constexpr bool is_shuffle_available ()
{
  constexpr bool type_in_list =
    utils::meta::is_any<data_type,
      int,
      long,
      long long,
      unsigned int,
      unsigned long,
      unsigned long long,
      float,
      double>::value;
  constexpr bool version_is_fine =
    utils::cuda::check_compute_capability<300> ();

  return type_in_list && version_is_fine;
}

namespace binary_op
{

template <typename data_type>
class sum
{
public:
  __device__ data_type operator () (const data_type &lhs, const data_type &rhs)
  {
    return lhs + rhs;
  }
};

}

template <
  typename data_type,
  typename warp_object,
  bool use_shared_memory>
class shared_dependency_injector;

template <
  typename data_type,
  typename warp_object>
class shared_dependency_injector<data_type, warp_object, false>
{
public:
  static __device__ warp_object create (data_type *) { return warp_object (); }
};

template <
  typename data_type,
  typename warp_object>
class shared_dependency_injector<data_type, warp_object, true>
{
public:
  static __device__ warp_object create (data_type *cache) { return warp_object (cache); }
};

} // warp
} // culib

#endif //CULIB_UTILS_H
