#include <algorithm>
#include <iostream>
#include <vector>
#include <thread>
#include <memory>
#include <chrono>

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
  result_class () = default;
  explicit result_class (unsigned int pixels_count)
    : data (new unsigned char[pixels_count])
  { }

  float elapsed {};
  std::unique_ptr<unsigned char[]> data;
};

template <int div>
void cpu_div_kernel (
  const unsigned int first,
  const unsigned int last,
  const unsigned char *input, unsigned char *output)
{
  for (unsigned int i = first; i < last; i++)
    output[i] = input[i] / div;
}

template <int div>
result_class cpu_div (const img_class *img)
{
  result_class cpu_result;
  cpu_result.data.reset (new unsigned char[img->pixels_count]);

  auto begin = std::chrono::high_resolution_clock::now ();

  cpu_div_kernel<div> (0, img->pixels_count, img->data.get (), cpu_result.data.get ());

  auto end = std::chrono::high_resolution_clock::now ();
  cpu_result.elapsed = std::chrono::duration_cast<std::chrono::duration<double>> (end - begin).count ();

  return cpu_result;
}

template <int div>
result_class cpu_div_mt (const img_class *img)
{
  result_class cpu_result;
  cpu_result.data.reset (new unsigned char[img->pixels_count]);

  auto begin = std::chrono::high_resolution_clock::now ();
  unsigned char *out = cpu_result.data.get ();
  const unsigned char *in = img->data.get ();

  std::vector<std::thread> threads;
  unsigned int threads_count = std::thread::hardware_concurrency ();
  unsigned int chunk_size = img->pixels_count / threads_count;

  for (unsigned int tid = 1; tid < 2; tid++)
    {
      threads.push_back (std::thread ([&, tid] () {
          const size_t first = chunk_size * tid;
          const size_t last = tid == threads_count - 1 ? img->pixels_count : first + chunk_size;

          cpu_div_kernel<div> (first, last, in, out);
        }));
    }

  cpu_div_kernel<div> (0, (threads_count == 1 ? img->pixels_count : chunk_size), in, out);

  for (auto &thread: threads)
    thread.join ();

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

class single_gpu_data
{
public:
  single_gpu_data (
    const img_class *img,
    unsigned int gpu_id,
    unsigned int gpus_count)
  {
    const unsigned int pixels_count = img->pixels_count;
    const unsigned int chunk_size = pixels_count / gpus_count;
    first_pixel = chunk_size * gpu_id;
    last_pixel = gpu_id == gpus_count - 1
               ? pixels_count
               : chunk_size * (gpu_id + 1);

    blocks_count = (get_n_subpixels () / 8 + threads_per_block - 1) / threads_per_block;

    device_result.resize (get_n_padded_subpixels ());
    device_img.resize (get_n_padded_subpixels ());

    culib::device::send_n (img->data.get (), get_n_subpixels (), device_img.get ());

    cudaEventCreate (&begin);
    cudaEventCreate (&end);
  }

  ~single_gpu_data ()
  {
    cudaEventDestroy (begin);
    cudaEventDestroy (end);
  }

  unsigned int get_n_subpixels () const
  {
    return last_pixel - first_pixel;
  }

  unsigned int get_n_padded_subpixels () const
  {
    return round_up (std::max (get_n_subpixels (), blocks_count * threads_per_block), 8);
  }

  void receive (unsigned char *result) const
  {
    culib::device::recv_n (device_result.get (), get_n_subpixels (), result + first_pixel);
  }

  template <int div>
  void launch ()
  {
    cudaEventRecord(begin);

    gpu_div_kernel_vec<div> <<<blocks_count, threads_per_block>>>(device_img.get (), device_result.get ());

    cudaEventRecord(end);
  }

  float get_elapsed ()
  {
    cudaEventSynchronize (end);

    float result {};
    cudaEventElapsedTime (&result, begin, end);
    return result / 1000.0;
  }

private:
  cudaEvent_t begin, end;
  culib::device::resizable_array<unsigned char> device_result;
  culib::device::resizable_array<unsigned char> device_img;

  unsigned int first_pixel {};
  unsigned int last_pixel {};
  unsigned int blocks_count {};
  unsigned int threads_per_block = 256;
};

template <int div>
result_class gpu_div_vec (const img_class *img)
{
  result_class gpu_result (img->pixels_count);
  single_gpu_data gpu_data (img, 0, 1);

  auto begin = std::chrono::high_resolution_clock::now ();

  for (unsigned int i = 0; i < 100; i++)
    gpu_data.launch<div> ();
  gpu_data.get_elapsed ();
  auto end = std::chrono::high_resolution_clock::now ();

  gpu_result.elapsed = std::chrono::duration_cast<std::chrono::duration<double>> (end - begin).count ();
  gpu_data.receive (gpu_result.data.get ());
  return gpu_result;
}

template <int div>
result_class gpu_div_vec_multiple_gpu_per_single_thread (const img_class *img)
{
  int devices_count {};
  cudaGetDeviceCount (&devices_count);

  result_class gpu_result (img->pixels_count);
  std::unique_ptr<std::unique_ptr<single_gpu_data>[]> gpu_data (new std::unique_ptr<single_gpu_data>[devices_count]);

  for (unsigned int device_id = 0; device_id < devices_count; device_id++)
    {
      cudaSetDevice (device_id);
      gpu_data[device_id].reset (new single_gpu_data (img, device_id, devices_count));
      cudaDeviceSynchronize ();
    }

  auto begin = std::chrono::high_resolution_clock::now ();

  for (unsigned int i = 0; i < 100; i++)
    {
      for (unsigned int device_id = 0; device_id < devices_count; device_id++)
        {
          cudaSetDevice (device_id);
          gpu_data[device_id]->launch<div> ();
        }
    }

  for (unsigned int device_id = 0; device_id < devices_count; device_id++)
    {
      cudaSetDevice (device_id);
      gpu_data[device_id]->get_elapsed ();
    }

  auto end = std::chrono::high_resolution_clock::now ();
  gpu_result.elapsed = std::chrono::duration_cast<std::chrono::duration<double>> (end - begin).count ();

  return gpu_result;
}

template <int div>
result_class gpu_div_vec_multiple_gpu_per_multi_thread (const img_class *img)
{
    int devices_count {};
    cudaGetDeviceCount (&devices_count);

    result_class gpu_result (img->pixels_count);
    std::unique_ptr<std::unique_ptr<single_gpu_data>[]> gpu_data (new std::unique_ptr<single_gpu_data>[devices_count]);

    for (unsigned int device_id = 0; device_id < devices_count; device_id++)
    {
        cudaSetDevice (device_id);
        gpu_data[device_id].reset (new single_gpu_data (img, device_id, devices_count));
        cudaDeviceSynchronize ();
    }

    auto begin = std::chrono::high_resolution_clock::now ();

    std::vector<std::thread> threads;

    for (unsigned int device_id = 0; device_id < devices_count; device_id++)
      {
        threads.push_back (std::thread ([&,device_id] () {
            cudaSetDevice (device_id);

            for (unsigned int step = 0; step < 100; step++)
              gpu_data[device_id]->launch<div> ();
            gpu_data[device_id]->get_elapsed ();
        }));
      }

    for (auto &thread: threads)
      thread.join ();

    auto end = std::chrono::high_resolution_clock::now ();
    gpu_result.elapsed = std::chrono::duration_cast<std::chrono::duration<double>> (end - begin).count ();

    return gpu_result;
}

int main (int argc, char *argv[])
{
    char *input_file {};
#if PNG_PRESENT
  if (argc != 2)
    {
      std::cout << "Usage: " << argv[0] << " [path to png file]";
      return 1;
    }
    intput_file = argv[1];
#endif

  auto img = read_png_file (input_file);
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

  if (0)
  {
      auto cpu_result = cpu_div<div> (img.get ());
      std::cout << "cpu: " << cpu_result.elapsed << "s\n";

      auto cpu_mt_result = cpu_div_mt<div> (img.get ());
      std::cout << "cpu mt: " << cpu_mt_result.elapsed << "s\n";

      auto gpu_result = gpu_div<div> (img.get ());
      std::cout << "gpu: " << gpu_result.elapsed << "s\n";

      auto gpu_padded_result = gpu_div_padded<div> (img.get ());
      std::cout << "gpu padded: " << gpu_padded_result.elapsed << "s\n";
  }

  auto gpu_vec_result = gpu_div_vec<div> (img.get ());
  std::cout << "gpu vec: " << gpu_vec_result.elapsed << "s\n";

  auto gpu_vec_single_thread_multiple_gpu_result = gpu_div_vec_multiple_gpu_per_single_thread<div> (img.get ());
  std::cout << "gpu vec (single thread multiple gpus): " << gpu_vec_single_thread_multiple_gpu_result.elapsed << "s\n";

  auto gpu_vec_multi_thread_multiple_gpu_result = gpu_div_vec_multiple_gpu_per_multi_thread<div> (img.get ());
  std::cout << "gpu vec (multi-thread multiple gpus): " << gpu_vec_multi_thread_multiple_gpu_result.elapsed << "s\n";

  // write_png_file (result.data.get (), img->width, img->height, "result.png");

  return 0;
}