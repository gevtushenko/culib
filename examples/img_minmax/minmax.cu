#include <algorithm>
#include <iostream>
#include <numeric>
#include <chrono>
#include <nccl.h>

#include "png_reader.h"

enum class calculation_mode
{
  cpu, cpu_mt, culib, nccl, unknown
};

calculation_mode str_to_mode (const char *mode)
{
  if (strcmp (mode, "cpu") == 0)
    return calculation_mode::cpu;
  if (strcmp (mode, "cpu_mt") == 0)
    return calculation_mode::cpu_mt;
  else if (strcmp (mode, "culib") == 0)
    return calculation_mode::culib;
  else if (strcmp (mode, "nccl") == 0)
    return calculation_mode::nccl;
  return calculation_mode::unknown;
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

class result_class
{
public:
  result_class () = default;
  result_class (float elapsed_arg, int minimal_value_arg)
    : elapsed (elapsed_arg)
    , minimal_value (minimal_value_arg)
  { }

  float elapsed {};
  int minimal_value {};
};

result_class nccl_test (const img_class *img)
{
  int devices_count {};
  cudaGetDeviceCount (&devices_count);

  // Initialize nccl
  std::unique_ptr<int[]> devs (new int[devices_count]);
  std::unique_ptr<ncclComm_t[]> comms (new ncclComm_t[devices_count]);

  std::unique_ptr<cudaStream_t[]> streams (new cudaStream_t[devices_count]);
  std::unique_ptr<cudaEvent_t[]> begins (new cudaEvent_t[devices_count]);
  std::unique_ptr<cudaEvent_t[]> ends (new cudaEvent_t[devices_count]);
  std::unique_ptr<unsigned char *[]> inputs (new unsigned char*[devices_count]);

  std::iota (devs.get (), devs.get () + devices_count, 0);
  ncclCommInitAll (comms.get (), devices_count, devs.get ());

  for (int i = 0; i < devices_count; i++)
    {
      const unsigned int chunk_begin = get_chunk_begin (img->pixels_count, devices_count, i);
      const unsigned int chunk_end = get_chunk_end (img->pixels_count, devices_count, i);
      const unsigned int chunk_size = chunk_end - chunk_begin;

      cudaSetDevice (devs[i]);
      cudaMalloc (&inputs[i], sizeof (unsigned char) * (chunk_size + 1));
      cudaMemcpy (inputs[i], img->data.get () + chunk_begin, sizeof (unsigned char) * chunk_size, cudaMemcpyHostToDevice);
      cudaStreamCreate (&streams[i]);
      cudaEventCreate (&begins[i]);
      cudaEventCreate (&ends[i]);
    }

  for (int i = 0; i < devices_count; i++)
    {
      cudaSetDevice (devs[i]);
      cudaEventRecord (begins[i]);
    }

  ncclGroupStart ();
  for (int i = 0; i < devices_count; i++)
    {
      const unsigned int chunk_begin = get_chunk_begin (img->pixels_count, devices_count, i);
      const unsigned int chunk_end = get_chunk_end (img->pixels_count, devices_count, i);
      const unsigned int chunk_size = chunk_end - chunk_begin;

      ncclAllReduce (inputs[i], inputs[i], chunk_size + 1, ncclChar, ncclMin, comms[i], streams[i]);
    }
  ncclGroupEnd ();

  // Synchronize devices
  for (int i = 0; i < devices_count; i++)
    {
      cudaSetDevice (i);
      cudaEventRecord (ends[i]);
    }

  std::unique_ptr<float[]> elapsed_times (new float[devices_count]);

  for (int i = 0; i < devices_count; i++)
    {
      cudaSetDevice (i);
      cudaEventSynchronize (ends[i]);
      cudaEventElapsedTime (elapsed_times.get () + i, begins[i], ends[i]);
      cudaStreamSynchronize (streams[i]);
    }

  std::unique_ptr<unsigned char[]> results (new unsigned char[devices_count]);

  for (int i = 0; i < devices_count; i++)
    {
      const unsigned int chunk_begin = get_chunk_begin (img->pixels_count, devices_count, i);
      const unsigned int chunk_end = get_chunk_end (img->pixels_count, devices_count, i);
      const unsigned int chunk_size = chunk_end - chunk_begin;

      cudaSetDevice (devs[i]);
      cudaMemcpy (&results[i], inputs[i] + chunk_size, sizeof (unsigned char), cudaMemcpyDeviceToHost);
      cudaFree (inputs[i]);
      cudaStreamDestroy (streams[i]);
      cudaEventDestroy (begins[i]);
      cudaEventDestroy (ends[i]);
    }

  // Finalize nccl
  for (int i = 0; i < devices_count; i++)
    ncclCommDestroy (comms[i]);

  return { *std::max_element (elapsed_times.get (), elapsed_times.get () + devices_count) / 1000, results[0]};
}

result_class cpu_test (const img_class *img)
{
  unsigned char min {};
  const unsigned char *input = img->data.get ();

  auto begin = std::chrono::high_resolution_clock::now ();
  for (unsigned int i = 0; i < img->pixels_count; i++)
    if (input[i] < min)
      min = input[i];

  auto end = std::chrono::high_resolution_clock::now ();
  const double elapsed = std::chrono::duration_cast<std::chrono::duration<double>> (end - begin).count ();

  return { static_cast<float> (elapsed), min };
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

  result_class result {};

  switch (mode)
    {
      case calculation_mode::cpu:
        result = cpu_test (img.get ());
        break;
      case calculation_mode::cpu_mt:
      case calculation_mode::culib:
        std::cerr << "Unsupported mode\n";
        return -1;
      case calculation_mode::nccl:
        result = nccl_test (img.get ());
        break;
    };

  std::cout << "Complete in " << result.elapsed << "s (result = " << result.minimal_value << ")\n";

  return 0;
}