#ifndef CULIB_BINARY_OPS_CUH
#define CULIB_BINARY_OPS_CUH

#include "culib/utils/meta/limits.cuh"

namespace culib
{
namespace binary_op
{

template<typename data_type>
class sum
{
public:
  __device__ data_type
  operator() (const data_type &lhs, const data_type &rhs)
  {
    return lhs + rhs;
  }

  __device__ data_type identity () const { return {}; }
};

template<typename data_type>
class max
{
public:
  __device__ data_type
  operator() (const data_type &lhs, const data_type &rhs)
  {
    return ::max (lhs, rhs);
  }

  __device__ data_type
  identity () const { return meta::numeric_limits<data_type>::min (); }
};

} // binary_op
} // culib

#endif
