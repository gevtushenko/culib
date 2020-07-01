//
// Created by egi on 7/1/20.
//

#include "culib/device/memory/resizable_array.h"
#include "culib/device/memory/api.h"

#ifndef CULIB_CONST_RESIZABLE_ARRAY_CUH
#define CULIB_CONST_RESIZABLE_ARRAY_CUH

namespace culib
{
namespace device
{

template <typename data_type, unsigned int size>
__constant__ data_type cache[size];

template<typename data_type, unsigned int const_size>
class const_resizable_array;

template<typename data_type, unsigned int const_size = 1024>
class const_resizeable_array_accessor
{
  data_type *ptr;
public:
  const_resizeable_array_accessor () = delete;

  __device__ data_type operator[] (unsigned int idx)
  {
    if (idx < const_size)
      return cache<data_type, const_size>[idx];
    else
      return ptr[idx - const_size];
  }

private:
  __host__ const_resizeable_array_accessor (data_type *ptr_arg) : ptr (ptr_arg) { }

  friend class const_resizable_array<data_type, const_size>;
};

template<typename data_type, unsigned int const_size = 1024>
class const_resizable_array
{
  resizable_array<data_type> dynamic_memory;

public:
  const_resizable_array () = default;
  explicit const_resizable_array (unsigned int size)
  {
    resize (size);
  }

  bool resize (unsigned int size)
  {
    if (size > const_size)
      return dynamic_memory.resize (size - const_size);
    return true;
  }

  void send_n (const data_type *host_data, unsigned int n)
  {
    if (const_size)
      cudaMemcpyToSymbol (cache<data_type, const_size>, host_data, sizeof (data_type) * std::min (const_size, n));

    if (n > const_size)
      culib::device::send_n (host_data + const_size, n - const_size, dynamic_memory.get ());
  }

  const_resizeable_array_accessor<data_type, const_size> get_accessor ()
  {
    return { dynamic_memory.get () };
  }
};

}
}

#endif //CULIB_CONST_RESIZEABLE_ARRAY_H
