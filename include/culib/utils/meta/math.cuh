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

constexpr __device__ unsigned int log2 (unsigned int n)
{
  if (n < 2)
    return 0;

  return 1 + log2 (n / 2);
}

}
}
} // culib

#endif //CULIB_MATH_H
