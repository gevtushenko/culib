//
// Created by egi on 7/1/20.
//

#include "culib/device/memory/resizable_array.h"
#include "culib/device/memory/api.h"

#ifndef CULIB_CONST_RESIZABLE_ARRAY_CUH
#define CULIB_CONST_RESIZABLE_ARRAY_CUH

#ifndef CULIB_CONST_STORAGE_SIZE
#define CULIB_CONST_STORAGE_SIZE 1024
#endif

#ifndef CULIB_CONST_STORAGE_TYPE
#define CULIB_CONST_STORAGE_TYPE char
#endif

namespace culib
{
namespace device
{

__constant__ CULIB_CONST_STORAGE_TYPE cache[CULIB_CONST_STORAGE_SIZE];

template<typename data_type>
class const_resizable_array;

template<typename data_type>
class const_resizeable_array_accessor
{
  data_type *ptr;
public:
  const_resizeable_array_accessor () = delete;

  __device__ data_type operator[] (unsigned int idx)
  {
    constexpr unsigned int elements_in_cache = CULIB_CONST_STORAGE_SIZE * sizeof (CULIB_CONST_STORAGE_TYPE)
                                             / sizeof (data_type);
    if (idx < elements_in_cache)
      return cache[idx];
    else
      return ptr[idx - elements_in_cache];
  }

private:
  __host__ const_resizeable_array_accessor (data_type *ptr_arg) : ptr (ptr_arg) { }

  friend class const_resizable_array<data_type>;
};

template<typename data_type>
class const_resizable_array
{
  const unsigned int elements_in_cache = CULIB_CONST_STORAGE_SIZE * sizeof (CULIB_CONST_STORAGE_TYPE)
                                       / sizeof (data_type);
  resizable_array<data_type> dynamic_memory;

public:
  const_resizable_array () = default;
  explicit const_resizable_array (unsigned int size)
  {
    resize (size);
  }

  bool resize (unsigned int size)
  {
    if (size > elements_in_cache)
      return dynamic_memory.resize (size - elements_in_cache);
    return true;
  }

  void send_n (const data_type *host_data, unsigned int n)
  {
    if (elements_in_cache)
      cudaMemcpyToSymbol (cache, host_data, sizeof (data_type) * std::min (elements_in_cache, n));

    if (n > elements_in_cache)
      culib::device::send_n (host_data + elements_in_cache, n - elements_in_cache, dynamic_memory.get ());
  }

  const_resizeable_array_accessor<data_type> get_accessor ()
  {
    return { dynamic_memory.get () };
  }
};

}
}

#endif //CULIB_CONST_RESIZEABLE_ARRAY_H
