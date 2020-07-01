#include <memory>

#include <cuda_runtime.h>

constexpr int ilp = 16;

#include <cub/cub.cuh>

#include "culib/warp/reduce.cuh"
#include "culib/warp/scan.cuh"

#include "results.h"

template <int ilp, typename data_type, bool expclusive>
__global__ void test_cub_warp_exclusive_scan_sum_kernel (
  data_type *data,
  data_type param,
  unsigned long long int *elapsed
)
{
  static_assert (ilp > 0, "ILP should be greater than zero");

  data_type thread_values[ilp];
  for (int i = 0; i < ilp; i++)
    thread_values[i] = param + i;

  typedef cub::WarpScan<data_type> WarpScan;
  __shared__ typename WarpScan::TempStorage temp_storage[ilp];

  unsigned long long int begin = clock64 ();

  if (expclusive)
    {
      for (int i = 0; i < ilp; i++)
        WarpScan(temp_storage[i]).ExclusiveSum (thread_values[i], thread_values[i]);
    }
  else
    {
      for (int i = 0; i < ilp; i++)
        WarpScan(temp_storage[i]).InclusiveSum (thread_values[i], thread_values[i]);
    }

  unsigned long long int end = clock64 ();

  for (int i = 1; i < ilp; i++)
    thread_values[0] += thread_values[i];

  data[threadIdx.x] = thread_values[0];

  if (threadIdx.x == 0)
    elapsed[0] = (end - begin) / ilp;
}

template <int ilp, typename data_type>
__global__ void test_cub_warp_reduce_sum_kernel (
	data_type *data,
	data_type param,
	unsigned long long int *elapsed
	)
{
  static_assert (ilp > 0, "ILP should be greater than zero");

	data_type thread_values[ilp];
	for (int i = 0; i < ilp; i++)
	  thread_values[i] = param + i;

	typedef cub::WarpReduce<data_type> WarpReduce;
	__shared__ typename WarpReduce::TempStorage temp_storage[ilp];

  unsigned long long int begin = clock64 ();

  for (int i = 0; i < ilp; i++)
    thread_values[i] = WarpReduce(temp_storage[i]).Sum (thread_values[i]);

  unsigned long long int end = clock64 ();

  for (int i = 1; i < ilp; i++)
    thread_values[0] += thread_values[i];

	data[threadIdx.x] = thread_values[0];

  if (threadIdx.x == 0)
		elapsed[0] = (end - begin) / ilp;
}

template <int ilp, typename data_type, bool exclusive>
__global__ void test_culib_warp_exclusive_scan_kernel (
  data_type *data,
  data_type param,
  unsigned long long int *elapsed
)
{
  static_assert (ilp > 0, "ILP should be greater than zero");

  data_type thread_values[ilp];
  for (int i = 0; i < ilp; i++)
    thread_values[i] = param + i;

  culib::warp::scan<data_type> scan;

  unsigned long long int begin = clock64 ();

  if (exclusive)
    {
      for (int i = 0; i < ilp; i++)
        thread_values[i] = scan.exclusive (thread_values[i]);
    }
  else
    {
      for (int i = 0; i < ilp; i++)
        thread_values[i] = scan.inclusive (thread_values[i]);
    }

  unsigned long long int end = clock64 ();

  for (int i = 1; i < ilp; i++)
    thread_values[0] += thread_values[i];

  data[threadIdx.x] = thread_values[0];

  if (threadIdx.x == 0)
    elapsed[0] = (end - begin) / ilp;
}

template <int ilp, typename data_type>
__global__ void test_culib_warp_reduce_sum_kernel (
  data_type *data,
  data_type param,
  unsigned long long int *elapsed
)
{
  static_assert (ilp > 0, "ILP should be greater than zero");

  data_type thread_values[ilp];
  for (int i = 0; i < ilp; i++)
    thread_values[i] = param + i;

  culib::warp::reduce<data_type> reduce;

  unsigned long long int begin = clock64 ();

  for (int i = 0; i < ilp; i++)
    thread_values[i] = reduce (thread_values[i]);

  unsigned long long int end = clock64 ();

  for (int i = 1; i < ilp; i++)
    thread_values[0] += thread_values[i];

  data[threadIdx.x] = thread_values[0];

  if (threadIdx.x == 0)
    elapsed[0] = (end - begin) / ilp;
}

template <typename data_type, int ilp, typename kernel_call_type>
unsigned long long int
test_warp_kernel (const kernel_call_type &kernel_call)
{
	data_type *data {};
  unsigned long long int *elapsed {};
  cudaMalloc (&elapsed, sizeof (unsigned long long int));

	cudaMalloc (&data, sizeof (data_type) * 32 * ilp);
	cudaMemset (data, 0, sizeof (data_type) * 32 * ilp);

	kernel_call (data, elapsed);

	cudaFree (data);

  unsigned long long int result {};
  cudaMemcpy (&result, elapsed, sizeof (unsigned long long int), cudaMemcpyDeviceToHost);

  cudaFree (elapsed);

  return result;
}

template <typename data_type>
size_clk warp_reduce_benchmark_culib ()
{
  return { 32, test_warp_kernel<data_type, ilp> (
    [] (data_type *data, unsigned long long int *elapsed) {
      test_culib_warp_reduce_sum_kernel<ilp, data_type> <<<1, 32>>> (data, 42, elapsed);
    }) };
}

template <typename data_type>
size_clk warp_exclusive_scan_benchmark_culib ()
{
  return { 32, test_warp_kernel<data_type, ilp> (
    [] (data_type *data, unsigned long long int *elapsed) {
      test_culib_warp_exclusive_scan_kernel<ilp, data_type, true> <<<1, 32>>> (data, 42, elapsed);
    }) };
}

template <typename data_type>
size_clk warp_inclusive_scan_benchmark_culib ()
{
  return { 32, test_warp_kernel<data_type, ilp> (
    [] (data_type *data, unsigned long long int *elapsed) {
      test_culib_warp_exclusive_scan_kernel<ilp, data_type, false> <<<1, 32>>> (data, 42, elapsed);
    }) };
}

template <typename data_type>
size_clk warp_reduce_benchmark_cub ()
{
  return { 32, test_warp_kernel<data_type, ilp> (
    [] (data_type *data, unsigned long long int *elapsed) {
      test_cub_warp_reduce_sum_kernel<ilp, data_type> <<<1, 32>>> (data, 42, elapsed);
    }) };
}

template <typename data_type>
size_clk warp_exclusive_scan_benchmark_cub ()
{
  return { 32, test_warp_kernel<data_type, ilp> (
    [] (data_type *data, unsigned long long int *elapsed) {
      test_cub_warp_exclusive_scan_sum_kernel<ilp, data_type, true> <<<1, 32>>> (data, 42, elapsed);
    }) };
}

template <typename data_type>
size_clk warp_inclusive_scan_benchmark_cub ()
{
  return { 32, test_warp_kernel<data_type, ilp> (
    [] (data_type *data, unsigned long long int *elapsed) {
      test_cub_warp_exclusive_scan_sum_kernel<ilp, data_type, false> <<<1, 32>>> (data, 42, elapsed);
    }) };
}

imp_result warp_reduce_benchmark ()
{
  const std::string cub_with_version = std::string ("cub.") + std::to_string (CUB_VERSION);

  return {
    {"culib", {
                         {"int", { warp_reduce_benchmark_culib<int> ()}},
                         {"double", { warp_reduce_benchmark_culib<double> ()}}}},
    {cub_with_version, {
                         {"int", { warp_reduce_benchmark_cub<int> ()}},
                         {"double", { warp_reduce_benchmark_cub<double> ()}}},

    }};
}

imp_result warp_exclusive_scan_benchmark ()
{
  const std::string cub_with_version = std::string ("cub.") + std::to_string (CUB_VERSION);

  return {
    {"culib", {
                {"int", { warp_exclusive_scan_benchmark_culib<int> ()}},
                {"double", { warp_exclusive_scan_benchmark_culib<double> ()}}}},
    {cub_with_version, {
                {"int", { warp_exclusive_scan_benchmark_cub<int> ()}},
                {"double", { warp_exclusive_scan_benchmark_cub<double> ()}}},

    }};
}

imp_result warp_inclusive_scan_benchmark ()
{
  const std::string cub_with_version = std::string ("cub.") + std::to_string (CUB_VERSION);

  return {
    {"culib", {
                {"int", { warp_inclusive_scan_benchmark_culib<int> ()}},
                {"double", { warp_inclusive_scan_benchmark_culib<double> ()}}}},
    {cub_with_version, {
                {"int", { warp_inclusive_scan_benchmark_cub<int> ()}},
                {"double", { warp_inclusive_scan_benchmark_cub<double> ()}}},

    }};
}

alg_result warp_benchmark ()
{
  return {
    { "reduce", warp_reduce_benchmark () },
    { "inclusive scan", warp_inclusive_scan_benchmark() },
    { "exclusive scan", warp_exclusive_scan_benchmark() }
  };
}

int main ()
{
  dev_result results;

  int devices_count {};
  cudaGetDeviceCount(&devices_count);

  for (unsigned int device = 0; device < devices_count; device++)
    {
      cudaDeviceProp device_properties;
      cudaGetDeviceProperties(&device_properties, device);

      if (results.count (device_properties.name))
        continue; ///< Skip devices that has already been tested

      cudaSetDevice (device);

      results[device_properties.name] = { { "warp", warp_benchmark () } };
    }

  print_results (results);
  dump_results (results);
}
