//
// Created by egi on 4/5/20.
//

#ifndef CULIB_WARP_UTILS_H
#define CULIB_WARP_UTILS_H

#include "culib/utils/cuda/version.h"
#include "culib/utils/meta/any.cuh"
#include "culib/utils/meta/limits.cuh"

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

inline __device__
void sync ()
{
#if __CUDACC_VER_MAJOR__ >= 9
  __syncwarp ();
#endif
}

/**
 * Check if specified data_type is supported by warp shuffle functions.
 */
template <typename data_type>
__device__ constexpr bool is_shuffle_available ()
{
  return
    meta::is_any<data_type,
      int,
      long,
      long long,
      unsigned int,
      unsigned long,
      unsigned long long,
      float,
      double>::value
 && cuda::check_compute_capability<300> ();
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

#endif // CULIB_WARP_UTILS_H
