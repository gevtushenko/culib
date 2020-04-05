//
// Created by egi on 4/5/20.
//

#ifndef CULIB___WARP_SCAN_CUH
#define CULIB___WARP_SCAN_CUH

#define __FULL_WARP_MASK 0xffffffff

namespace culib
{
namespace warp
{
namespace detail
{

inline __device__ unsigned int lane_id () {
  unsigned int ret;
  asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
  return ret;
}

template <typename data_type>
class default_scan_binary_op
{
public:
  __device__ data_type operator () (const data_type &lhs, const data_type &rhs)
  {
    return lhs + rhs;
  }
};

template <int x>
struct log2 { enum { value = 1 + log2<x/2>::value }; };

template <> struct log2<1> { enum { value = 1 }; };

template <typename data_type, int warp_size=32>
class warp_shfl_scan
{
  const unsigned lid;

  template <typename binary_operation>
  __device__ data_type scan_step (data_type val, unsigned offset, binary_operation binary_op)
  {
    data_type result = val;
    data_type tmp = __shfl_up_sync (__FULL_WARP_MASK, result, offset, warp_size);
    result = binary_op (tmp, result);

    if (lid < offset)
      result = val;

    return result;
  }

public:
  __device__ warp_shfl_scan ()
    : lid (lane_id ())
  { }

  template <typename binary_operation>
  __device__ data_type scan_value (data_type val, binary_operation binary_op)
  {
    data_type result = val;

    for (unsigned step = 0; step < log2<warp_size>::value; step++)
      {
        const int offset = 1 << step;
        result = scan_step (result, offset, binary_op);
      }

    return result;
  }

public:
  static constexpr bool use_shared = false;
};

template <typename data_type, int warp_size=32>
class warp_shrd_scan
{
  data_type *warp_shared_workspace;

public:
  warp_shrd_scan () = delete;
  explicit __device__ warp_shrd_scan (data_type *warp_shared_workspace_arg)
    : warp_shared_workspace (warp_shared_workspace_arg)
  { }

  template <typename binary_operation>
  __device__ data_type scan_value (data_type val, binary_operation binary_op)
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

#endif //CULIB___WARP_SCAN_CUH
