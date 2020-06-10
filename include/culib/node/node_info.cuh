#ifndef CULIB_NODE_INFO_CUH_
#define CULIB_NODE_INFO_CUH_

namespace culib
{

class node_info
{
public:
  node_info (
    unsigned int device_num_arg,
    unsigned int devices_count_arg)
    : device_num (device_num_arg)
    , devices_count (devices_count_arg)
  { }

  const unsigned int device_num {};
  const unsigned int devices_count {};
};

}

#endif