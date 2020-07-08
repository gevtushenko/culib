#include <png.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <memory>
#include <chrono>

#include <immintrin.h>
#include <smmintrin.h>

#include "png_reader.h"

#include "culib/device/memory/resizable_array.h"
#include "culib/device/memory/api.h"

enum class calculation_mode
{
  cpu, cpu_mt, cpu_mt_simd, gpu, unknown
};

calculation_mode str_to_mode (const char *mode)
{
  if (strcmp (mode, "cpu") == 0)
    return calculation_mode::cpu;
  if (strcmp (mode, "cpu_mt") == 0)
    return calculation_mode::cpu_mt;
  if (strcmp (mode, "cpu_mt_simd") == 0)
    return calculation_mode::cpu_mt_simd;
  else if (strcmp (mode, "gpu") == 0)
    return calculation_mode::gpu;
  return calculation_mode::unknown;
}

class result_class
{
public:
  float elapsed {};
  std::unique_ptr<unsigned char[]> data;
};

result_class cpu_div (const img_class *img, unsigned char div)
{
  result_class cpu_result;
  cpu_result.data.reset (new unsigned char[img->pixels_count]);

  auto begin = std::chrono::high_resolution_clock::now ();
  unsigned char *out = cpu_result.data.get ();

  const unsigned char *in = img->data.get ();
  for (unsigned int i = 0; i < img->pixels_count; i++)
    out[i] = in[i] / div;
  auto end = std::chrono::high_resolution_clock::now ();
  cpu_result.elapsed = std::chrono::duration_cast<std::chrono::duration<double>> (end - begin).count ();

  return cpu_result;
}

template <int div>
__global__ void gpu_div_kernel (
  unsigned int pixels_count,
  const unsigned char * __restrict__ input,
        unsigned char * __restrict__ output)
{
  const unsigned int pixel = threadIdx.x + blockDim.x * blockIdx.x;

  if (pixel < pixels_count)
    output[pixel] = input[pixel] / div;
}

template <int div>
result_class gpu_div (const img_class *img)
{
  result_class gpu_result;

  const unsigned int img_elements = img->row_size * img->height;
  culib::device::resizable_array<unsigned char> device_result (img_elements);
  culib::device::resizable_array<unsigned char> device_img (img_elements);
  culib::device::send_n (img->data.get (), img_elements, device_img.get ());

  unsigned int threads_per_block = 128;
  unsigned int blocks_count = (img_elements + threads_per_block - 1) / threads_per_block;

  cudaEvent_t begin, end;
  cudaEventCreate (&begin);
  cudaEventCreate (&end);

  cudaEventRecord(begin);

  gpu_div_kernel<div> <<<blocks_count, threads_per_block>>>(img_elements, device_img.get (), device_result.get ());

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  gpu_result.data.reset (new unsigned char[img_elements]);
  culib::device::recv_n (device_result.get (), img_elements, gpu_result.data.get ());

  cudaEventElapsedTime (&gpu_result.elapsed, begin, end);
  gpu_result.elapsed /= 1000;
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  return gpu_result;
}

template <int div>
__global__ void gpu_div_kernel_padded (
  const unsigned char * __restrict__ input,
  unsigned char * __restrict__ output)
{
  const unsigned int pixel = threadIdx.x + blockDim.x * blockIdx.x;
  output[pixel] = input[pixel] / div;
}

template <int div>
result_class gpu_div_padded (const img_class *img)
{
  result_class gpu_result;

  const unsigned int img_elements = img->row_size * img->height;
  unsigned int threads_per_block = 128;
  unsigned int blocks_count = (img_elements + threads_per_block - 1) / threads_per_block;

  culib::device::resizable_array<unsigned char> device_result (threads_per_block * blocks_count);
  culib::device::resizable_array<unsigned char> device_img (threads_per_block * blocks_count);
  culib::device::send_n (img->data.get (), img_elements, device_img.get ());

  cudaEvent_t begin, end;
  cudaEventCreate (&begin);
  cudaEventCreate (&end);

  cudaEventRecord(begin);

  gpu_div_kernel_padded<div> <<<blocks_count, threads_per_block>>>(device_img.get (), device_result.get ());

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  gpu_result.data.reset (new unsigned char[img_elements]);
  culib::device::recv_n (device_result.get (), img_elements, gpu_result.data.get ());

  cudaEventElapsedTime (&gpu_result.elapsed, begin, end);
  gpu_result.elapsed /= 1000;
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  return gpu_result;
}

struct char_vec
{
public:
  __device__ char_vec (const unsigned char * __restrict__ input)
  {
    *reinterpret_cast<int2*>(data) = *reinterpret_cast<const int2*> (input);
  }

  __device__ void store (unsigned char * output) const
  {
    *reinterpret_cast<int2*> (output) = *reinterpret_cast<const int2*>(data);
  }

public:
  static constexpr int size = 8;
  unsigned char data[size];
};

template <int div>
__global__ void gpu_div_kernel_vec (
  const unsigned char * __restrict__ input,
  unsigned char * __restrict__ output)
{
  auto tid = threadIdx.x + blockDim.x * blockIdx.x;

  char_vec vec (input + tid * 8);

  unsigned char *vec_data = vec.data;
  for (int i = 0; i < vec.size; i++)
    vec_data[i] /= div;

  vec.store (output + tid * 8);
}

unsigned int round_up (unsigned int val, unsigned int multiple)
{
  if (multiple == 0)
    return val;

  int remainder = val % multiple;
  if (remainder == 0)
    return val;

  return val + multiple - remainder;
}

template <int div>
result_class gpu_div_vec (const img_class *img)
{
  result_class gpu_result;

  const unsigned int img_elements = img->row_size * img->height;
  unsigned int threads_per_block = 128;
  unsigned int blocks_count = (img_elements / 8 + threads_per_block - 1) / threads_per_block;
  unsigned int padded_size = round_up (img_elements, 8);

  culib::device::resizable_array<unsigned char> device_result (padded_size);
  culib::device::resizable_array<unsigned char> device_img (padded_size);
  culib::device::send_n (img->data.get (), img_elements, device_img.get ());

  cudaEvent_t begin, end;
  cudaEventCreate (&begin);
  cudaEventCreate (&end);

  cudaEventRecord(begin);

  gpu_div_kernel_vec<div> <<<blocks_count, threads_per_block>>>(device_img.get (), device_result.get ());

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  gpu_result.data.reset (new unsigned char[img_elements]);
  culib::device::recv_n (device_result.get (), img_elements, gpu_result.data.get ());

  cudaEventElapsedTime (&gpu_result.elapsed, begin, end);
  gpu_result.elapsed /= 1000;
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  return gpu_result;
}

int main (int argc, char *argv[])
{
  if (argc != 2)
    {
      std::cout << "Usage: " << argv[0] << " [path to png file]";
      return 1;
    }

  auto img = read_png_file (argv[1]);
  if (!img)
    {
      std::cerr << "Can't read " << argv[1] << "\n";
      return 1;
    }

  if (!img->is_gray || img->width != img->row_size)
    {
      std::cerr << "Only grayscale png without alpha channel is supported now\n";
      return 1;
    }

  constexpr unsigned char div = 4;

  auto cpu_result = cpu_div (img.get (), div);
  std::cout << "cpu: " << cpu_result.elapsed << "s\n";

  auto gpu_result = gpu_div<div> (img.get ());
  std::cout << "gpu: " << gpu_result.elapsed << "s\n";

  auto gpu_padded_result = gpu_div_padded<div> (img.get ());
  std::cout << "gpu_padded: " << gpu_padded_result.elapsed << "s\n";

  auto gpu_vec_result = gpu_div_vec<div> (img.get ());
  std::cout << "gpu_vec: " << gpu_vec_result.elapsed << "s\n";

  // write_png_file (result.data.get (), img->width, img->height, "result.png");

  return 0;
}