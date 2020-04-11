//
// Created by egi on 4/5/20.
//

#ifndef CULIB_DETAIL_WARP_REDUCE_CUH
#define CULIB_DETAIL_WARP_REDUCE_CUH

#include <type_traits>

#include "culib/warp/utils.cuh"
#include "culib/utils/meta/math.cuh"

namespace culib
{
namespace warp
{
namespace detail
{

template <typename data_type, int warp_size=32>
class warp_shfl_reduce
{
public:
  warp_shfl_reduce () = default;

  template <typename binary_operation>
  __device__ data_type reduce_value (data_type val, binary_operation binary_op)
  {
    for (int s = warp_size / 2; s > 0; s >>= 1)
      val = binary_op (val, __shfl_xor_sync (get_full_wark_mask (), val, s, warp_size));
    return val;
  }

public:
  static constexpr bool use_shared_memory = false;
};

template <typename data_type, int warp_size=32>
class warp_shrd_reduce
{
  data_type *warp_shared_workspace;

public:
  warp_shrd_reduce () = delete;
  explicit __device__ warp_shrd_reduce (data_type *warp_shared_workspace_arg)
    : warp_shared_workspace (warp_shared_workspace_arg)
  { }

  template <typename binary_operation>
  __device__ data_type reduce_value (data_type val, binary_operation binary_op)
  {
    const int lid = lane_id();
    warp_shared_workspace[lid] = val;
    __syncwarp();

    for (int s = warp_size / 2; s > 0; s >>= 1)
    {
      if (lid < s)
        val = binary_op (warp_shared_workspace[lid + s], val);
      __syncwarp ();

      if (lid < s)
        warp_shared_workspace[lid] = val;
      __syncwarp ();
    }

    data_type result = warp_shared_workspace[0];
    __syncwarp ();
    return result;
  }

public:
  static constexpr bool use_shared_memory = true;
};

template <typename data_type>
class warp_reduce_selector
{
public:
  using implementation = typename std::conditional<
    is_shuffle_available<data_type> (),
      warp_shfl_reduce<data_type>,
      warp_shrd_reduce<data_type>>::type;
};

} // detail
} // warp
} // culib

#endif // CULIB_DETAIL_WARP_SCAN_CUH
