#include "png_reader.h"

#include "culib/device/memory/resizable_array.h"
#include "culib/device/memory/const_resizable_array.cuh"

#include <iostream>
#include <vector>
#include <thread>
#include <memory>
#include <chrono>

enum class calculation_mode
{
  cpu, cpu_simd, cpu_simd_mt, gpu_constant, gpu, unknown
};

calculation_mode str_to_mode (const char *mode)
{
  if (strcmp (mode, "cpu") == 0)
    return calculation_mode::cpu;
  if (strcmp (mode, "cpu_simd") == 0)
    return calculation_mode::cpu_simd;
  if (strcmp (mode, "cpu_simd_mt") == 0)
    return calculation_mode::cpu_simd_mt;
  if (strcmp (mode, "gpu_constant") == 0)
    return calculation_mode::gpu_constant;
  else if (strcmp (mode, "gpu") == 0)
    return calculation_mode::gpu;
  return calculation_mode::unknown;
}

class result_class
{
public:
  float elapsed {};
  std::unique_ptr<float[]> data;
};

template <typename weights_type>
__global__ void convolution_constant_kernel (
  int width,
  int height,
  const float *data,
  const weights_type weights,
  float *result
  )
{
  const int row = threadIdx.y + blockDim.y * blockIdx.y;
  const int col = threadIdx.x + blockDim.x * blockIdx.x;

  if (row < 1 || row > height - 2 ||
      col < 1 || col > width - 2)
    {
      if (row < height && col < width)
        result[row * width + col] = {};
      return;
    }

  // First row
  const float res_0 = data[(row - 1) * width + col - 1] * weights[0 * 3 + 0];
  const float res_1 = data[(row - 1) * width + col + 0] * weights[0 * 3 + 1];
  const float res_2 = data[(row - 1) * width + col + 1] * weights[0 * 3 + 2];
  const float first_row_sum = res_0 + res_1 + res_2;

  // Second row
  const float res_3 = data[(row + 0) * width + col - 1] * weights[1 * 3 + 0];
  const float res_4 = data[(row + 0) * width + col + 0] * weights[1 * 3 + 1];
  const float res_5 = data[(row + 0) * width + col + 1] * weights[1 * 3 + 2];
  const float second_row_sum = res_3 + res_4 + res_5;

  // Third row
  const float res_6 = data[(row + 1) * width + col - 1] * weights[2 * 3 + 0];
  const float res_7 = data[(row + 1) * width + col + 0] * weights[2 * 3 + 1];
  const float res_8 = data[(row + 1) * width + col + 1] * weights[2 * 3 + 2];
  const float third_row_sum = res_6 + res_7 + res_8;

  result[row * width + col] = first_row_sum + second_row_sum + third_row_sum;
}

template <typename weights_type>
result_class convolution_constant (
  weights_type weights,
  const img_class *img)
{
  result_class result;

  const unsigned char *cpu_char_data = img->data.get ();
  std::unique_ptr<float[]> host_img (new float[img->pixels_count]);
  for (unsigned int i = 0; i < img->pixels_count; i++)
    host_img[i] = cpu_char_data[i];

  culib::device::resizable_array<float> device_img (img->pixels_count);
  culib::device::resizable_array<float> device_result (img->pixels_count);
  culib::device::send_n (host_img.get (), img->pixels_count, device_img.get ());

  const unsigned int threads_per_dim = 16;
  const unsigned int blocks_per_x = (img->width + threads_per_dim - 1) / threads_per_dim;
  const unsigned int blocks_per_y = (img->height + threads_per_dim - 1) / threads_per_dim;
  dim3 thread_block_size (threads_per_dim, threads_per_dim);
  dim3 grid_size (blocks_per_x, blocks_per_y);

  cudaEvent_t begin, end;
  cudaEventCreate (&begin);
  cudaEventCreate (&end);

  cudaEventRecord (begin);
  convolution_constant_kernel<<<grid_size, thread_block_size>>> (
    img->width, img->height, device_img.get (), weights, device_result.get ());
  cudaEventRecord (end);
  cudaEventSynchronize (end);

  cudaEventElapsedTime (&result.elapsed, begin, end);
  result.elapsed /= 1000.0f;

  cudaEventDestroy (begin);
  cudaEventDestroy (end);

  result.data.reset (new float[img->pixels_count]);
  culib::device::recv_n (device_result.get (), img->pixels_count, result.data.get ());

  return result;
}

result_class convolution_cpu_naive (const img_class *img)
{
  result_class result;
  result.data.reset (new float[img->pixels_count]);

  const unsigned char *cpu_char_data = img->data.get ();
  std::unique_ptr<float[]> host_img (new float[img->pixels_count]);
  for (unsigned int i = 0; i < img->pixels_count; i++)
    host_img[i] = cpu_char_data[i];

  const unsigned int height = img->height;
  const unsigned int width = img->width;

  float *res = result.data.get ();

  auto begin = std::chrono::high_resolution_clock::now ();

  constexpr int weights_size = 9;
  const float weights[weights_size] =
    { 2, 5, 2,
      5, 9, 5,
      2, 5, 2 };

  const float *inp = host_img.get ();

  for (unsigned int j = 0; j < width; j++)
    res[j] = {};
  res += width;

  for (unsigned int i = 1; i < height - 1; i++)
    {
      res[0] = {};

      for (unsigned int j = 1; j < width - 1; j++)
        {
          const float r1 = inp[(i - 1) * width + j - 1] * weights[0 * 3 + 0];
          const float r2 = inp[(i - 1) * width + j + 0] * weights[0 * 3 + 1];
          const float r3 = inp[(i - 1) * width + j + 1] * weights[0 * 3 + 2];
          const float first_row_sum = r1 + r2 + r3;

          const float r4 = inp[(i + 0) * width + j - 1] * weights[1 * 3 + 0];
          const float r5 = inp[(i + 0) * width + j + 0] * weights[1 * 3 + 1];
          const float r6 = inp[(i + 0) * width + j + 1] * weights[1 * 3 + 2];
          const float second_row_sum = r4 + r5 + r6;

          const float r7 = inp[(i + 1) * width + j - 1] * weights[2 * 3 + 0];
          const float r8 = inp[(i + 1) * width + j + 0] * weights[2 * 3 + 1];
          const float r9 = inp[(i + 1) * width + j + 1] * weights[2 * 3 + 2];
          const float third_row_sum = r7 + r8 + r9;

          res[j] = first_row_sum + second_row_sum + third_row_sum;
        }

      res[width - 1] = {};
      res += width;
    }

  for (unsigned int j = 0; j < width; j++)
    res[j] = {};

  auto end = std::chrono::high_resolution_clock::now ();
  result.elapsed = std::chrono::duration_cast<std::chrono::duration<double>> (end - begin).count ();

  return result;
}

#include <immintrin.h>

void convolution_1d_simd (const float *in, const float *weights, float *aligned_out, unsigned int width)
{
  __m256 vector_weights[3] __attribute__((aligned(32)));
  __m256 vector_in __attribute__((aligned(32)));
  __m256 accumulator __attribute__((aligned(32)));
  __m256 product __attribute__((aligned(32)));

  // Broadcast weights
  for(unsigned int i = 0; i < 3; ++i)
    vector_weights[i] = _mm256_set1_ps (weights[i]);

  unsigned int i = 0;
  for (; i < width - 8; i += 8)
    {
      accumulator = _mm256_setzero_ps (); // Return vector of type __m256 with all elements set to zero.

      for (int k = -1; k < 2; k++)
        {
          vector_in = _mm256_loadu_ps (in + i + k);
          product = _mm256_mul_ps (vector_weights[k + 1], vector_in);
          accumulator = _mm256_add_ps (accumulator, product);
        }

      _mm256_store_ps (aligned_out + i, accumulator);
    }

  for (; i < width; i++)
    {
      aligned_out[i] = 0.0f;

      for (int k = -1; k < 2; k++)
        aligned_out[i] += in[i + k] * weights[k + 1];
    }
}

result_class convolution_cpu_simd (const img_class *img)
{
  result_class result;
  result.data.reset (new float[img->pixels_count]);

  float *aligned_rows[3] = {};
  const unsigned int convolution_width = img->width - 2;
  for (unsigned int i = 0; i < 3; i ++)
    posix_memalign ((void**) &aligned_rows[i], 32, convolution_width * sizeof (float));

  const unsigned char *cpu_char_data = img->data.get ();
  std::unique_ptr<float[]> host_img (new float[img->pixels_count]);
  for (unsigned int i = 0; i < img->pixels_count; i++)
    host_img[i] = cpu_char_data[i];

  const unsigned int height = img->height;
  const unsigned int width = img->width;

  float *res = result.data.get ();

  auto begin = std::chrono::high_resolution_clock::now ();

  constexpr int weights_size = 9;
  const float weights[weights_size] =
    { 2, 5, 2,
      5, 9, 5,
      2, 5, 2 };

  const float *inp = host_img.get ();

  for (unsigned int j = 0; j < width; j++)
    res[j] = {};
  res += width;

  __m256 sum_1;
  __m256 sum_2;

  for (unsigned int i = 1; i < height - 1; i++)
    {
      res[0] = {};

      convolution_1d_simd (inp + (i - 1) * width + 1, weights, aligned_rows[0], convolution_width);
      convolution_1d_simd (inp + (i + 0) * width + 1, weights, aligned_rows[1], convolution_width);
      convolution_1d_simd (inp + (i + 1) * width + 1, weights, aligned_rows[2], convolution_width);

      unsigned int j = 0;
      for (; j < convolution_width - 8; j += 8)
        {
          sum_1 = _mm256_add_ps (_mm256_load_ps (aligned_rows[0] + j), _mm256_load_ps (aligned_rows[1] + j));
          sum_2 = _mm256_add_ps (sum_1, _mm256_load_ps (aligned_rows[2] + j));

          _mm256_storeu_ps (res + j + 1, sum_2);
        }

      for (; j < convolution_width; j++)
        res[j + 1] = aligned_rows[0][j] + aligned_rows[1][j] + aligned_rows[2][j];

      res[width - 1] = {};
      res += width;
    }

  for (unsigned int j = 0; j < width; j++)
    res[j] = {};

  auto end = std::chrono::high_resolution_clock::now ();
  result.elapsed = std::chrono::duration_cast<std::chrono::duration<double>> (end - begin).count ();

  for (unsigned int i = 0; i < 3; i ++)
    free (aligned_rows[i]);

  return result;
}

unsigned int get_chunk_begin (unsigned int n, unsigned int chunks_count, unsigned int chunk_id)
{
  return (n / chunks_count) * chunk_id;
}

unsigned int get_chunk_end (unsigned int n, unsigned int chunks_count, unsigned int chunk_id)
{
  if (chunk_id == chunks_count - 1)
    return n;
  return (n / chunks_count) * (chunk_id + 1);
}

void convolution_2d_simd (
  const float *inp,
  float *result,
  const unsigned int width,
  const unsigned int first_row,
  const unsigned int last_row)
{
  constexpr int weights_size = 9;
  const float weights[weights_size] =
    { 2, 5, 2,
      5, 9, 5,
      2, 5, 2 };

  float *aligned_rows[3] = {};
  const unsigned int convolution_width = width - 2;
  for (unsigned int i = 0; i < 3; i ++)
    posix_memalign ((void**) &aligned_rows[i], 32, convolution_width * sizeof (float));

  float *res = result + first_row * width;

  if (first_row == 0)
    {
      for (unsigned int j = 0; j < width; j++)
        res[j] = {};
      res += width;
    }

  __m256 sum_1;
  __m256 sum_2;

  for (unsigned int i = first_row; i < last_row; i++)
    {
      res[0] = {};

      convolution_1d_simd (inp + (i - 1) * width + 1, weights, aligned_rows[0], convolution_width);
      convolution_1d_simd (inp + (i + 0) * width + 1, weights, aligned_rows[1], convolution_width);
      convolution_1d_simd (inp + (i + 1) * width + 1, weights, aligned_rows[2], convolution_width);

      unsigned int j = 0;
      for (; j < convolution_width - 8; j += 8)
        {
          sum_1 = _mm256_add_ps (_mm256_load_ps (aligned_rows[0] + j), _mm256_load_ps (aligned_rows[1] + j));
          sum_2 = _mm256_add_ps (sum_1, _mm256_load_ps (aligned_rows[2] + j));

          _mm256_storeu_ps (res + j + 1, sum_2);
        }

      for (; j < convolution_width; j++)
        res[j + 1] = aligned_rows[0][j] + aligned_rows[1][j] + aligned_rows[2][j];

      res[width - 1] = {};
      res += width;
    }

  for (unsigned int i = 0; i < 3; i ++)
    free (aligned_rows[i]);
}

result_class convolution_cpu_simd_mt (const img_class *img)
{
  result_class result;
  result.data.reset (new float[img->pixels_count]);

  const unsigned char *cpu_char_data = img->data.get ();
  std::unique_ptr<float[]> host_img (new float[img->pixels_count]);
  for (unsigned int i = 0; i < img->pixels_count; i++)
    host_img[i] = cpu_char_data[i];

  const unsigned int height = img->height;
  const unsigned int width = img->width;

  float *res = result.data.get ();

  auto begin = std::chrono::high_resolution_clock::now ();

  const unsigned int threads_count = std::thread::hardware_concurrency ();
  std::vector<std::thread> threads;

  for (unsigned int tid = 0; tid < threads_count - 1; tid++)
    {
      threads.push_back (std::thread ([&,tid] () {
        const unsigned int first_row = get_chunk_begin (height, threads_count, tid);
        const unsigned int last_row = get_chunk_end (height, threads_count, tid);
        convolution_2d_simd (host_img.get (), res, width, (first_row + (tid == 0) /* skip first row */), last_row);
      }));
    }

  const unsigned int first_row = get_chunk_begin (height, threads_count, threads_count - 1);
  convolution_2d_simd (host_img.get (), res, width, first_row, height - 1);

  for (unsigned int j = 0; j < width; j++)
    res[j] = {};

  for (auto &thread: threads)
    thread.join ();

  auto end = std::chrono::high_resolution_clock::now ();
  result.elapsed = std::chrono::duration_cast<std::chrono::duration<double>> (end - begin).count ();

  return result;
}

int main (int argc, char *argv[])
{
  if (argc != 3)
    {
      std::cout << "Usage: " << argv[0] << " [mode: cpu || gpu_constant || gpu] [path to png file]";
      return 1;
    }

  const calculation_mode mode = str_to_mode (argv[1]);
  if (mode == calculation_mode::unknown)
    {
      std::cerr << "Supported modes: cpu, gpu_constant, gpu\n";
      return 1;
    }

  auto img = read_png_file (argv[2]);
  if (!img)
    {
      std::cerr << "Can't read " << argv[2] << "\n";
      return 1;
    }

  if (!img->is_gray || img->width != img->row_size)
    {
      std::cerr << "Only grayscale png without alpha channel is supported now\n";
      return 1;
    }

  result_class result = [&] ()
  {
    constexpr int weights_size = 9;
    const float host_weights[weights_size] =
      { 2, 5, 2,
        5, 9, 5,
        2, 5, 2 };

    if (mode == calculation_mode::cpu)
      {
        return convolution_cpu_naive (img.get ());
      }
    if (mode == calculation_mode::cpu_simd)
      {
        return convolution_cpu_simd (img.get ());
      }
    if (mode == calculation_mode::cpu_simd_mt)
      {
        return convolution_cpu_simd_mt (img.get ());
      }
    if (mode == calculation_mode::gpu)
      {
        culib::device::resizable_array<float> device_weights (weights_size);
        culib::device::send_n (host_weights, weights_size, device_weights.get ());
        return convolution_constant (device_weights.get (), img.get ());
      }
    if (mode == calculation_mode::gpu_constant)
      {
        culib::device::const_resizable_array<float, weights_size> device_weights (weights_size);
        device_weights.send_n (host_weights, weights_size);
        return convolution_constant (device_weights.get_accessor (), img.get ());
      }
    return result_class {};
  } ();

  std::cout << "Complete in " << result.elapsed << "s\n";

  float maximum = 0.0;
  float minimum = std::numeric_limits<float>::max ();
  const float *result_data = result.data.get ();
  for (unsigned int i = 0; i < img->pixels_count; i++)
  {
    maximum = std::max (maximum, result_data[i]);
    minimum = std::min (minimum, result_data[i]);
  }

  std::unique_ptr<unsigned char[]> char_result (new unsigned char[img->pixels_count]);
  unsigned char *char_data = char_result.get ();

  for (unsigned int i = 0; i < img->pixels_count; i++)
  {
    const float normilized_value = (result_data[i] - minimum) / (maximum - minimum) * 255.0f;
    char_data[i] = normilized_value;
  }

  write_png_file (char_data, img->width, img->height, "result.png");
  return 0;
}