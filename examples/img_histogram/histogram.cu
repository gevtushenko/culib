#include <png.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <memory>
#include <chrono>

#include <immintrin.h>
#include <smmintrin.h>

#include "culib/device/histogram.cuh"
#include "culib/device/memory/resizable_array.h"
#include "culib/device/memory/api.h"

#include "cub/cub.cuh"

constexpr unsigned int bins_count = 256;

class img_class
{
public:
  const unsigned int width {};
  const unsigned int height {};
  const unsigned int pixels_count {};
  const bool is_gray {};
  const std::size_t row_size {};
  const std::unique_ptr<const png_byte[]> data;

public:
  img_class (
    int width_arg,
    int height_arg,
    bool is_gray_arg,
    size_t row_size_arg,
    std::unique_ptr<png_byte[]> data_arg)
    : width (width_arg)
    , height (height_arg)
    , pixels_count (width * height)
    , is_gray (is_gray_arg)
    , row_size (row_size_arg)
    , data (std::move (data_arg))
  { }

};

std::unique_ptr<img_class> read_png_file (char *filename)
{
  FILE *fp = fopen (filename, "rb");

  if (!fp)
    return {};

  png_structp png = png_create_read_struct (PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

  if (!png)
    return {};

  png_infop info = png_create_info_struct(png);
  if(!info)
    return {};

  if (setjmp(png_jmpbuf(png)))
    return {};

  png_init_io (png, fp);

  png_read_info (png, info);

  int width = png_get_image_width (png, info);
  int height = png_get_image_height (png, info);
  png_byte color_type = png_get_color_type (png, info);
  png_byte bit_depth = png_get_bit_depth (png, info);

  if(bit_depth == 16)
    png_set_strip_16 (png);

  if(color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_palette_to_rgb (png);

  // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
  if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
    png_set_expand_gray_1_2_4_to_8(png);

  if(png_get_valid(png, info, PNG_INFO_tRNS))
    png_set_tRNS_to_alpha(png);

  png_read_update_info(png, info);

  const std::size_t row_size = png_get_rowbytes (png, info);
  std::unique_ptr<png_byte[]> raw_data (new png_byte[height * row_size]);
  std::unique_ptr<png_bytep[]> row_pointers_helper (new png_bytep[height]);

  {
    png_byte *raw_ptr = raw_data.get ();
    for(int y = 0; y < height; y++) {
        row_pointers_helper[y] = raw_ptr;
        raw_ptr += row_size;
      }
  }

  png_read_image(png, row_pointers_helper.get ());

  fclose(fp);

  png_destroy_read_struct(&png, &info, NULL);

  const bool is_gray = color_type == PNG_COLOR_TYPE_GRAY
                    || color_type == PNG_COLOR_TYPE_GRAY_ALPHA;

  return std::unique_ptr<img_class> {new img_class (width, height, is_gray, row_size, std::move (raw_data))};
}

enum class calculation_mode
{
  cpu, cpu_mt, cpu_mt_simd, culib, cub, unknown
};

calculation_mode str_to_mode (const char *mode)
{
  if (strcmp (mode, "cpu") == 0)
    return calculation_mode::cpu;
  if (strcmp (mode, "cpu_mt") == 0)
    return calculation_mode::cpu_mt;
  if (strcmp (mode, "cpu_mt_simd") == 0)
    return calculation_mode::cpu_mt_simd;
  else if (strcmp (mode, "culib") == 0)
    return calculation_mode::culib;
  else if (strcmp (mode, "cub") == 0)
    return calculation_mode::cub;
  return calculation_mode::unknown;
}

class result_class
{
public:
  float elapsed {};
  std::unique_ptr<unsigned int[]> data;
};

result_class cpu_hist (const img_class *img)
{
  result_class cpu_result;
  cpu_result.data.reset (new unsigned int[bins_count]);

  auto begin = std::chrono::high_resolution_clock::now ();
  unsigned int *hist = cpu_result.data.get ();
  std::fill_n (hist, bins_count, 0);

  const unsigned char *in = img->data.get ();
  for (unsigned int i = 0; i < img->pixels_count; i++)
    hist[in[i]]++;
  auto end = std::chrono::high_resolution_clock::now ();
  cpu_result.elapsed = std::chrono::duration_cast<std::chrono::duration<double>> (end - begin).count ();

  return cpu_result;
}

result_class cpu_mt_hist (const img_class *img)
{
  result_class cpu_result;
  cpu_result.data.reset (new unsigned int[bins_count]);

  auto begin = std::chrono::high_resolution_clock::now ();
  unsigned int *hist = cpu_result.data.get ();
  std::fill_n (hist, bins_count, 0);

  const unsigned int threads_count = std::thread::hardware_concurrency ();
  const unsigned int chunk_size = img->pixels_count / threads_count;
  std::vector<std::thread> threads;
  std::unique_ptr<std::unique_ptr<unsigned int[]>[]> thread_buffers (new std::unique_ptr<unsigned int[]>[threads_count]);

  for (unsigned int tid = 0; tid < threads_count; tid++)
    {
      threads.push_back(std::thread ([&,tid] () {
        const unsigned int first_element = chunk_size * tid;
        const unsigned int last_element = tid == threads_count - 1
                                        ? img->pixels_count
                                        : chunk_size * (tid + 1);

        thread_buffers[tid].reset (new unsigned int[bins_count]);
        const unsigned char *in = img->data.get ();
        unsigned int *thread_local_hist = thread_buffers[tid].get ();

        for (unsigned int i = first_element; i < last_element; i++)
          thread_local_hist[in[i]]++;
      }));
    }

  for (auto &thread: threads)
    thread.join ();

  for (unsigned int tid = 0; tid < threads_count; tid++)
    for (unsigned int bin = 0; bin < bins_count; bin++)
      hist[bin] += thread_buffers[tid][bin];

  auto end = std::chrono::high_resolution_clock::now ();
  cpu_result.elapsed = std::chrono::duration_cast<std::chrono::duration<double>> (end - begin).count ();

  return cpu_result;
}

result_class cpu_mt_hist_simd (const img_class *img)
{
  result_class cpu_result;
  cpu_result.data.reset (new unsigned int[bins_count]);

  auto begin = std::chrono::high_resolution_clock::now ();
  unsigned int *hist = cpu_result.data.get ();
  std::fill_n (hist, bins_count, 0);

  const unsigned int threads_count = std::thread::hardware_concurrency ();
  const unsigned int chunk_size = img->pixels_count / threads_count;
  std::vector<std::thread> threads;
  std::unique_ptr<std::unique_ptr<unsigned int[]>[]> thread_buffers (new std::unique_ptr<unsigned int[]>[threads_count]);

  for (unsigned int tid = 0; tid < threads_count; tid++)
    {
      threads.push_back(std::thread ([&,tid] () {
        const unsigned int first_element = chunk_size * tid;
        const unsigned int last_element = tid == threads_count - 1
                                          ? img->pixels_count
                                          : chunk_size * (tid + 1);
        const unsigned int last_element_rounded_down = (last_element / 32) * 32;

        unsigned int thread_buffer[32][bins_count] = {};

        const unsigned char *in = img->data.get () + first_element;

        __m256i next_packed_vec = _mm256_loadu_si256((__m256i*)in);
        for (unsigned int i = first_element; i < last_element_rounded_down; i += 32)
          {
            __m256i current_vec = next_packed_vec;
            next_packed_vec = _mm256_loadu_si256((__m256i*)(in+=32));

            thread_buffer[0][_mm256_extract_epi8(current_vec,  0)]++;
            thread_buffer[1][_mm256_extract_epi8(current_vec,  1)]++;
            thread_buffer[2][_mm256_extract_epi8(current_vec,  2)]++;
            thread_buffer[3][_mm256_extract_epi8(current_vec,  3)]++;
            thread_buffer[4][_mm256_extract_epi8(current_vec,  4)]++;
            thread_buffer[5][_mm256_extract_epi8(current_vec,  5)]++;
            thread_buffer[6][_mm256_extract_epi8(current_vec,  6)]++;
            thread_buffer[7][_mm256_extract_epi8(current_vec,  7)]++;
            thread_buffer[0][_mm256_extract_epi8(current_vec,  8)]++;
            thread_buffer[1][_mm256_extract_epi8(current_vec,  9)]++;
            thread_buffer[2][_mm256_extract_epi8(current_vec, 10)]++;
            thread_buffer[3][_mm256_extract_epi8(current_vec, 11)]++;
            thread_buffer[4][_mm256_extract_epi8(current_vec, 12)]++;
            thread_buffer[5][_mm256_extract_epi8(current_vec, 13)]++;
            thread_buffer[6][_mm256_extract_epi8(current_vec, 14)]++;
            thread_buffer[7][_mm256_extract_epi8(current_vec, 15)]++;
            thread_buffer[0][_mm256_extract_epi8(current_vec, 16)]++;
            thread_buffer[1][_mm256_extract_epi8(current_vec, 17)]++;
            thread_buffer[2][_mm256_extract_epi8(current_vec, 18)]++;
            thread_buffer[3][_mm256_extract_epi8(current_vec, 19)]++;
            thread_buffer[4][_mm256_extract_epi8(current_vec, 20)]++;
            thread_buffer[5][_mm256_extract_epi8(current_vec, 21)]++;
            thread_buffer[6][_mm256_extract_epi8(current_vec, 22)]++;
            thread_buffer[7][_mm256_extract_epi8(current_vec, 23)]++;
            thread_buffer[0][_mm256_extract_epi8(current_vec, 24)]++;
            thread_buffer[1][_mm256_extract_epi8(current_vec, 25)]++;
            thread_buffer[2][_mm256_extract_epi8(current_vec, 26)]++;
            thread_buffer[3][_mm256_extract_epi8(current_vec, 27)]++;
            thread_buffer[4][_mm256_extract_epi8(current_vec, 28)]++;
            thread_buffer[5][_mm256_extract_epi8(current_vec, 29)]++;
            thread_buffer[6][_mm256_extract_epi8(current_vec, 30)]++;
            thread_buffer[7][_mm256_extract_epi8(current_vec, 31)]++;
          }

        for (unsigned int i = last_element_rounded_down; i < last_element; i++)
          thread_buffer[0][img->data[i]]++;

        thread_buffers[tid].reset (new unsigned int[bins_count]);
        for (unsigned int bin = 0; bin < bins_count; bin++)
          thread_buffers[tid][bin] = thread_buffer[0][bin]
                                   + thread_buffer[1][bin]
                                   + thread_buffer[2][bin]
                                   + thread_buffer[3][bin]
                                   + thread_buffer[4][bin]
                                   + thread_buffer[5][bin]
                                   + thread_buffer[6][bin]
                                   + thread_buffer[7][bin];
      }));
    }

  for (auto &thread: threads)
    thread.join ();

  for (unsigned int tid = 0; tid < threads_count; tid++)
    for (unsigned int bin = 0; bin < bins_count; bin++)
      hist[bin] += thread_buffers[tid][bin];

  auto end = std::chrono::high_resolution_clock::now ();
  cpu_result.elapsed = std::chrono::duration_cast<std::chrono::duration<double>> (end - begin).count ();

  return cpu_result;
}

result_class culib_hist (const img_class *img)
{
  result_class culib_result;

  const unsigned int img_elements = img->row_size * img->height;
  culib::device::resizable_array<unsigned int> result (bins_count);
  culib::device::resizable_array<unsigned int> workspace (culib::device::histogram::get_gpu_workspace_size (bins_count, img->pixels_count));
  culib::device::resizable_array<png_byte> device_img (img_elements);
  culib::device::send_n (img->data.get (), img_elements, device_img.get ());

  cudaEvent_t begin, end;
  cudaEventCreate (&begin);
  cudaEventCreate (&end);

  cudaEventRecord(begin);
  culib::device::histogram hist (workspace.get ());
  hist (bins_count, img_elements, device_img.get (), result.get ());
  cudaEventRecord(end);
  cudaEventSynchronize(end);

  culib_result.data.reset (new unsigned int[bins_count]);
  culib::device::recv_n (result.get (), bins_count, culib_result.data.get ());

  cudaEventElapsedTime (&culib_result.elapsed, begin, end);
  culib_result.elapsed /= 1000;
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  return culib_result;
}

result_class cub_hist (const img_class *img)
{
  result_class cub_result;

  int num_samples = img->pixels_count;
  int num_levels = 257;
  int num_bins = num_levels - 1;
  int lower_level = 0;
  int upper_level = 256;

  culib::device::resizable_array<unsigned int> device_histogram (num_bins);
  culib::device::resizable_array<png_byte> device_img (num_samples);
  culib::device::send_n (img->data.get (), num_samples, device_img.get ());

  size_t temp_storage_bytes {};

  cub::DeviceHistogram::HistogramEven (
    nullptr, temp_storage_bytes,
    device_img.get (), device_histogram.get (),
    num_levels, lower_level, upper_level,
    num_samples);

  culib::device::resizable_array<char> device_temp_storage (temp_storage_bytes);

  cudaEvent_t begin, end;
  cudaEventCreate (&begin);
  cudaEventCreate (&end);

  cudaEventRecord (begin);

  cub::DeviceHistogram::HistogramEven (
    device_temp_storage.get (), temp_storage_bytes,
    device_img.get (), device_histogram.get (),
    num_levels, lower_level, upper_level,
    num_samples);

  cudaEventRecord (end);

  cub_result.data.reset (new unsigned int[bins_count]);
  culib::device::recv_n (device_histogram.get (), bins_count, cub_result.data.get ());

  cudaEventElapsedTime (&cub_result.elapsed, begin, end);
  cub_result.elapsed /= 1000;
  cudaEventDestroy (begin);
  cudaEventDestroy (end);
  return cub_result;
}

int main (int argc, char *argv[])
{
  if (argc != 3)
    {
      std::cout << "Usage: " << argv[0] << " [mode: cpu || cpu_mt || culib || cub] [path to png file]";
      return 1;
    }

  const calculation_mode mode = str_to_mode (argv[1]);
  if (mode == calculation_mode::unknown)
    {
      std::cerr << "Supported modes: cpu, culib, cub\n";
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
    if (mode == calculation_mode::culib)
      return culib_hist (img.get ());
    if (mode == calculation_mode::cub)
      return cub_hist (img.get ());
    else if (mode == calculation_mode::cpu)
      return cpu_hist (img.get ());
    else if (mode == calculation_mode::cpu_mt)
      return cpu_mt_hist (img.get ());
    else if (mode == calculation_mode::cpu_mt_simd)
      return cpu_mt_hist_simd (img.get ());
    return result_class {};
  } ();

  std::cout << "Complete in " << result.elapsed << "s\n";

  std::ofstream os ("result.csv");
  for (unsigned int bin = 1; bin < bins_count; bin++)
    os << result.data[bin] << "\n";

  return 0;
}