#include "gtest/gtest.h"

#include <cuda_runtime.h>

#include "culib/device/memory/resizable_array.h"

using data_type = int;
using array_t = culib::device::resizable_array<data_type>;

TEST(resizable_array, empty_object_destructor)
{
  array_t array;
  EXPECT_EQ (array.get (), nullptr);
}

TEST(resizable_array, allocate)
{
  array_t array;
  const bool success = array.resize (1);
  EXPECT_EQ (success, true); // Should be enough memory for 1 int
  EXPECT_NE (array.get (), nullptr);
  EXPECT_EQ (array.get_size (), 1u);
}

TEST(resizable_array, resize_with_preserving)
{
  const size_t first_size = 1;
  const size_t second_size = 2;

  array_t array;
  array.resize (first_size);
  EXPECT_EQ (array.get_size (), first_size);

  const int magic_value = 42;
  culib::device::send_n (&magic_value, 1, array.get ());
  array.resize (second_size);

  EXPECT_NE (array.get (), nullptr);
  EXPECT_EQ (array.get_size (), second_size);

  EXPECT_EQ (culib::device::recv (array.get ()), magic_value);
}
