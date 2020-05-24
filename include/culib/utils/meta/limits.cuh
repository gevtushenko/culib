#ifndef CULIB_LIMITS_CUH
#define CULIB_LIMITS_CUH

#include <bits/c++config.h>

namespace culib
{
namespace meta
{

template <typename data_type>
class numeric_limits
{
public:
  static constexpr bool is_specialized = false;
};

template <>
class numeric_limits<int>
{
public:
  static constexpr bool is_specialized = true;

  __host__ __device__ static constexpr int min () noexcept { return - __INT_MAX__ - 1; }
  __host__ __device__ static constexpr int max () noexcept { return   __INT_MAX__; }
};

} // meta
} // culib

#endif
