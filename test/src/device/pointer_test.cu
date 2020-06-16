#include "gtest/gtest.h"

#include <cuda_runtime.h>

#include "culib/device/memory/pointer.h"

using data_type = int;
using ptr_t = culib::device::pointer<data_type>;

TEST(device_pointer, empty_object_destructor)
{
  ptr_t ptr;
  EXPECT_EQ (ptr.get (), nullptr);
}

TEST(device_pointer, filled_object_destructor)
{
  data_type *d_ptr {};
  cudaMalloc (&d_ptr, sizeof (data_type));
  ptr_t ptr (d_ptr);
}

TEST(device_pointer, reset)
{
  data_type *d_ptr {};
  cudaMalloc (&d_ptr, sizeof (data_type));
  ptr_t ptr (d_ptr);
  ptr.reset ();
  EXPECT_EQ (ptr.get (), nullptr);
}

TEST(device_pointer, move_constructor)
{
  data_type *d_ptr {};
  cudaMalloc (&d_ptr, sizeof (data_type));
  ptr_t ptr_src (d_ptr);
  ptr_t ptr_dst = std::move (ptr_src);

  EXPECT_EQ (ptr_src.get (), nullptr);
  EXPECT_EQ (ptr_dst.get (), d_ptr);
}

TEST(device_pointer, move_to_empty)
{
  data_type *d_ptr {};
  cudaMalloc (&d_ptr, sizeof (data_type));
  ptr_t ptr_src (d_ptr);
  ptr_t ptr_dst;

  ptr_dst = std::move (ptr_src);

  EXPECT_EQ (ptr_src.get (), nullptr);
  EXPECT_EQ (ptr_dst.get (), d_ptr);
}
