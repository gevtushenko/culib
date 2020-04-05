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

template <int x>
class log2
{
public:
  enum
  {
    value = 1 + log2<x/2>::value
  };
};

template <>
class log2<1>
{
public:
  enum
  {
    value = 1
  };
};

}
}
} // culib

#endif //CULIB_MATH_H
