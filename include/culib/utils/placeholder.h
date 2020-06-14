//
// Created by egi on 6/14/20.
//

#ifndef CULIB_PLACEHOLDER_H
#define CULIB_PLACEHOLDER_H

#if defined(__CUDACC__) // NVCC
  #define CULIB_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
  #define CULIB_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
  #define CULIB_ALIGN(n) __declspec(align(n))
#else
  #error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

namespace culib
{
namespace utils
{

template<typename data_type>
struct CULIB_ALIGN(sizeof (data_type)) placeholder
{
  char place[sizeof (data_type) / sizeof (char)];
};

}
}

#endif //CULIB_PLACEHOLDER_H
