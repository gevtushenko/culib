#include <iostream>
#include <memory>

// There is no such file in cub library, so you should execute something like this to compile benchmark
// version=$(git describe --tags); echo "#define CUB_VERSION \"${version}\"" > cub_version.cuh
#include <cub/cub_version.cuh>
#include <cub/cub.cuh>

#include "culib/warp/reduce.cuh"

template<typename data_type>
std::string type_to_string ();

template<> std::string type_to_string<int> ()    { return "int"; }
template<> std::string type_to_string<double> () { return "double"; }

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
void test_warp_reduce_sum (const std::string &name, const kernel_call_type &kernel_call)
{
	data_type *data {};
  unsigned long long int *elapsed {};
  cudaMalloc (&elapsed, sizeof (unsigned long long int));

	cudaMalloc (&data, sizeof (data_type) * 32 * ilp);
	cudaMemset (data, 0, sizeof (data_type) * 32 * ilp);

	kernel_call (data, elapsed);

	cudaFree (data);

	{
		unsigned long long int result {};
		cudaMemcpy (&result, elapsed, sizeof (unsigned long long int), cudaMemcpyDeviceToHost);
		std::cout << name << "[" << type_to_string<data_type> () << "] " << result << std::endl;
	}

  cudaFree (elapsed);
}

template <typename data_type>
void warp_reduce_benchmark ()
{
  constexpr int ilp = 16;

  const std::string cub_with_version = std::string ("cub.") + CUB_VERSION;
  test_warp_reduce_sum<data_type, ilp> (cub_with_version,
    [] (data_type *data, unsigned long long int *elapsed) {
      test_cub_warp_reduce_sum_kernel<ilp, data_type> <<<1, 32>>> (data, 42, elapsed);
    });

  test_warp_reduce_sum<data_type, ilp> ("culib",
    [] (data_type *data, unsigned long long int *elapsed) {
      test_culib_warp_reduce_sum_kernel<ilp, data_type> <<<1, 32>>> (data, 42, elapsed);
    });
}

int main ()
{
  warp_reduce_benchmark<int>();
  warp_reduce_benchmark<double>();
}
