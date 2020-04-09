//
// Created by egi on 4/5/20.
//

#ifndef CULIB_DETAIL_WARP_SCAN_CUH
#define CULIB_DETAIL_WARP_SCAN_CUH

#include "culib/warp/utils.cuh"
#include "culib/utils/meta/math.cuh"

namespace culib
{
namespace warp
{
namespace detail
{

/**
 * Hillis Steele inclusive scan.
 */
template <typename data_type, int warp_size=32>
class warp_shfl_scan
{
private:
  const unsigned lid;

  template <typename binary_operation>
  __device__ data_type scan_step (data_type val, unsigned offset, binary_operation binary_op)
  {
    data_type result = val;
    data_type tmp = __shfl_up_sync (get_full_wark_mask (), result, offset, warp_size);
    result = binary_op (tmp, result);

    if (lid < offset)
      result = val;

    return result;
  }

protected:
  __device__ data_type *get_cache () { return nullptr; }

public:
  __device__ warp_shfl_scan ()
    : lid (lane_id ())
  { }

  template <typename binary_operation>
  __device__ data_type scan_value (data_type val, binary_operation binary_op)
  {
    data_type result = val;

    for (unsigned step = 0; step < utils::math::log2<warp_size>::value; step++)
      result = scan_step (result, 1 << step, binary_op);

    return result;
  }

public:
  static constexpr bool use_shared_memory = false;
};

template <typename data_type, int warp_size=32>
class warp_shrd_scan
{
private:
  const unsigned lid;
  data_type *warp_shared_workspace;

protected:
  __device__ data_type *get_cache () { return warp_shared_workspace; }

public:
  warp_shrd_scan () = delete;
  explicit __device__ warp_shrd_scan (data_type *warp_shared_workspace_arg)
    : lid (lane_id ())
    , warp_shared_workspace (warp_shared_workspace_arg)
  { }

  template <typename binary_operation>
  __device__ data_type scan_value (data_type val, binary_operation binary_op)
  {
    warp_shared_workspace[lid] = val;
    __syncwarp ();

    for (unsigned step = 0; step < utils::math::log2<warp_size>::value; step++)
      {
        const unsigned offset = 1 << step;

        if (lid >= offset)
          val = binary_op (warp_shared_workspace[lid - offset], val);
        __syncwarp ();

        if (lid >= offset)
          warp_shared_workspace[lid] = val;
        __syncwarp ();
      }

    data_type result = warp_shared_workspace[lid];
    __syncwarp ();
    return result;
  }

public:
  static constexpr bool use_shared_memory = true;
};

template <typename data_type>
class warp_scan_selector
{
public:
  using implementation = typename std::conditional<
    is_shuffle_available<data_type> (),
      warp_shfl_scan<data_type>,
      warp_shrd_scan<data_type>>::type;
};

} // detail
} // warp
} // culib

#endif // CULIB_DETAIL_WARP_SCAN_CUH
