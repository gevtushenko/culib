//
// Created by egi on 6/16/20.
//

#ifndef CULIB_POINTER_H
#define CULIB_POINTER_H

#include <cuda_runtime.h>
#include <stdexcept>

namespace culib
{
namespace device
{

template <typename data_type>
class pointer
{
  using ptr_t = data_type *;
  using const_ptr_t = const data_type *;
  ptr_t ptr {};

public:
  pointer () = default;
  explicit pointer (ptr_t raw_ptr) : ptr (raw_ptr) { }

  pointer (const pointer&) = delete;
  pointer& operator=(const pointer&) = delete;

  pointer (pointer&& rhs)
    : ptr (rhs.release ())
  { }

  pointer& operator=(pointer&& rhs)
  {
    reset (rhs.release ());
    return *this;
  }

  ptr_t release () noexcept
  {
    ptr_t result = get ();
    ptr = ptr_t ();
    return result;
  }

  void reset () noexcept (false)
  {
    if (cudaFree (ptr) != cudaSuccess)
      throw std::runtime_error ("culib unique_ptr can't free pointer\n");
    ptr = nullptr;
  }

  void reset (ptr_t raw_ptr) noexcept (false)
  {
    reset ();
    ptr = raw_ptr;
  }

  ptr_t get () noexcept { return ptr; }
  const_ptr_t get () const noexcept { return ptr; }

  ~pointer () noexcept
  {
    reset (); ///< Terminate in case of exception
  }
};

}
}

#endif //CULIB_POINTER_H
