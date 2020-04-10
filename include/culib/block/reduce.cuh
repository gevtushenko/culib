//
// Created by egi on 4/10/20.
//

#ifndef CULIB_BLOCK_REDUCE_H
#define CULIB_BLOCK_REDUCE_H

#include "culib/warp/utils.cuh"
#include "culib/warp/reduce.cuh"

namespace culib
{
namespace block
{

template <int warp_size=32>
class runtime_indexing_policy_1D {
public:
  static __device__ int get_blck_id () { return blockIdx.x; }
  static __device__ int get_thrd_id () { return threadIdx.x; }
  static __device__ int get_lane_id () { return threadIdx.x % warp_size; }
  static __device__ int get_warp_id () { return threadIdx.x / warp_size; }

  static __device__ int get_n_thrd_in_block () { return blockDim.x; }
  static __device__ int get_n_warp_in_block () { return get_n_thrd_in_block() / warp_size; }
};

template<
  typename data_type,
  typename indexing_policy = runtime_indexing_policy_1D<32>>
class reduce
{
  data_type *block_shared_workspace;

public:
  explicit __device__ reduce (data_type *block_shared_workspace_arg)
    : block_shared_workspace (block_shared_workspace_arg)
  { }

  /**
   * @tparam binary_operation Binary combining function object thata will be applied in unspecified order.
   *                          The behaviour is undefined if binary_operation modifies any element.
   */
  template<typename binary_operation = warp::binary_op::sum<data_type>>
  __device__
  inline data_type
  operator ()(data_type val, binary_operation binary_op = {})
  {
    const int tid = indexing_policy::get_thrd_id ();
    const int lid = indexing_policy::get_lane_id ();
    const int wid = indexing_policy::get_warp_id ();

    using warp_reducer = warp::reduce<data_type>;

    data_type *warp_workspace = warp_reducer::use_shared_memory
                              ? block_shared_workspace + wid * warpSize
                              : nullptr;

    warp_reducer warp_reduce
      = warp::shared_dependency_injector<data_type, warp_reducer, warp_reducer::use_shared_memory>::create (
          warp_workspace);

    auto warp_reduce_result = warp_reduce (val, binary_op);

    if (lid == 0)
      block_shared_workspace[wid] = warp_reduce_result;
    __syncthreads ();

    val = tid < indexing_policy::get_n_warp_in_block ()
        ? block_shared_workspace[lid]
        : data_type {};

    if (wid == 0)
      {
        if (warp_reducer::use_shared_memory)
          __syncwarp (); /// Cache could be modified without sync in code below
        val = warp_reduce (val);
      }

    return val;
  }
};

} // block
} // culib

#endif // CULIB_BLOCKREDUCE_H
