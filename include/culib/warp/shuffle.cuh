//
// Created by egi on 4/5/20.
//

#ifndef CULIB_SHUFFLE_CUH
#define CULIB_SHUFFLE_CUH

#include <type_traits>

#include "culib/utils/meta/any.cuh"
#include "culib/utils/cuda/version.h"
#include "culib/warp/detail/warp_shuffle.cuh"

namespace culib
{
namespace warp
{

template<
  typename data_type,
  typename shuffle_policy = typename std::conditional<
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
    detail::shuffle_shfl<data_type>,
    detail::shuffle_shrd<data_type>>::type>
class shuffle : public shuffle_policy {
public:
  using shuffle_policy::shuffle_policy;

  __device__
  inline data_type operator ()(data_type val)
  {
    return shuffle_value (val);
  }
};

template <typename data_type, bool use_shared_memory>
class shuffle_dependency_injector;

template <typename data_type>
class shuffle_dependency_injector<data_type, false>
{
public:
  static __device__ shuffle<data_type> create (data_type *) { return shuffle<data_type> (); }
};

template <typename data_type>
class shuffle_dependency_injector<data_type, true>
{
public:
  static __device__ shuffle<data_type> create (data_type *cache) { return shuffle<data_type> ( cache ); }
};

} // warp
} // culib

#endif //CULIB_SHUFFLE_CUH
