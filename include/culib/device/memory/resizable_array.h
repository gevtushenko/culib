//
// Created by egi on 6/18/20.
//

#ifndef CULIB_RESIZABLE_ARRAY_H
#define CULIB_RESIZABLE_ARRAY_H

#include "culib/device/memory/api.h"
#include "culib/device/memory/pointer.h"

namespace culib
{
namespace device
{

// TODO Use virtual memory if possible

template <typename data_type>
class resizable_array
{
  std::size_t size {};
  std::size_t allocated {};
  pointer<data_type> memory;

public:
  resizable_array () = default;
  resizable_array (std::size_t size)
  {
    if (!resize (size, false))
      throw std::runtime_error ("culib: can't allocate " + std::to_string (size) + " elements");
  }

  data_type * get () { return memory.get (); }
  const data_type * get () const { return memory.get (); }

  std::size_t get_size () const
  {
    return size;
  }

  std::size_t get_allocated () const
  {
    return allocated;
  }

  data_type *release ()
  {
    size = allocated = 0;
    return memory.release ();
  }

  bool resize (std::size_t new_size, bool preserve_old_memory = true) noexcept
  {
    if (new_size > allocated)
      {
        pointer<data_type> new_memory;
        try {
            new_memory.reset (culib::device::allocate<data_type> (new_size));
          }
        catch (...) {
            return false;
          }

        if (preserve_old_memory)
          {
            try {
              const std::size_t memory_size_to_copy = std::min (new_size, size);
              culib::device::copy_n (memory.get (), memory_size_to_copy, new_memory.get ());
            }
            catch (...) {
              return false;
            }
          }

        try {
            memory = std::move (new_memory);
          }
        catch (...) {
            return false;
          }
        allocated = new_size;
      }

    size = new_size;

    return true;
  }
};

}
}

#endif //CULIB_RESIZABLE_ARRAY_H
