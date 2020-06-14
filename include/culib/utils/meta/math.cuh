//
// Created by egi on 4/5/20.
//

#ifndef CULIB_MATH_H
#define CULIB_MATH_H

namespace culib
{
namespace utils
{
namespace math
{

template <unsigned int n>
constexpr __device__ unsigned int log2 ()
{
  return 1 + log2<n / 2> ();
}

template<> constexpr __device__ unsigned int log2<0> () { return 0; }
template<> constexpr __device__ unsigned int log2<1> () { return 0; }

}
}
} // culib

#endif //CULIB_MATH_H
