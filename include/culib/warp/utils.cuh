//
// Created by egi on 4/5/20.
//

#ifndef CULIB_UTILS_H
#define CULIB_UTILS_H

namespace culib
{
namespace warp
{

inline __device__
unsigned int get_full_wark_mask ()
{
  return 0xffffffff;
}

inline __device__
unsigned int lane_id ()
{
  unsigned int ret;
  asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
  return ret;
}

} // warp
} // culib

#endif //CULIB_UTILS_H
