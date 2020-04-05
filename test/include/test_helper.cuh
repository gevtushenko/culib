#ifndef TEST_HELPER_H
#define TEST_HELPER_H

#include <cuda_runtime.h>

template <typename data_type, typename action_type>
__global__ void helper_kernel (
  const data_type *in,
  data_type *out,
  const action_type &action)
{
  action (in, out);
}

template <typename data_type, typename action_type>
void launch_kernel (
  dim3 grid,
  dim3 block,
  size_t data_size,
  const data_type *in,
  data_type *out,
  const action_type &action)
{
  data_type *d_in {};
  data_type *d_out {};

  cudaMalloc (&d_in, data_size * sizeof (data_type));
  cudaMalloc (&d_out, data_size * sizeof (data_type));

  cudaMemcpy (d_in, in, data_size * sizeof (data_type), cudaMemcpyHostToDevice);

  helper_kernel<<<grid, block>>> (d_in, d_out, action);

  cudaMemcpy (out, d_out, data_size * sizeof (data_type), cudaMemcpyDeviceToHost);

  cudaFree (d_out);
  cudaFree (d_in);
}

#endif