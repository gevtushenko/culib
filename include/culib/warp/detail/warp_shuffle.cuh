//
// Created by egi on 4/5/20.
//

#ifndef CULIB_WARP_SHUFFLE_H
#define CULIB_WARP_SHUFFLE_H

#include "culib/warp/utils.cuh"
#include "culib/utils/meta/math.cuh"

namespace culib
{
namespace warp
{
namespace detail
{

template <typename data_type, int warp_size=32>
class shuffle_shfl
{
  const unsigned lid;

public:
  __device__ shuffle_shfl ()
    : lid (lane_id ())
  { }

  __device__ data_type shuffle_value (data_type val)
  {
    data_type result = __shfl_up_sync (get_full_wark_mask (), val, 1, warp_size);
    return lid == 0 ? 0 : result;
  }

public:
  static constexpr bool use_shared = false;
};

template <typename data_type, int warp_size=32>
class shuffle_shrd
{
  data_type *warp_shared_workspace;

public:
  shuffle_shrd () = delete;
  explicit __device__ shuffle_shrd (data_type *warp_shared_workspace_arg)
    : warp_shared_workspace (warp_shared_workspace_arg)
  { }

  __device__ data_type shuffle_value (data_type val)
  {
    // TODO
    return 42;
  }

public:
  static constexpr bool use_shared = true;
};

} // detail
} // warp
} // culib

#endif //CULIB_WARP_SHUFFLE_H
