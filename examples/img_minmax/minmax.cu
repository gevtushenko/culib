#include <iostream>
#include <numeric>
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

void nccl_test (const img_class *img)
{
  int devices_count {};
  cudaGetDeviceCount (&devices_count);

  // Initialize nccl
  std::unique_ptr<int[]> devs (new int[devices_count]);
  std::unique_ptr<ncclComm_t[]> comms (new ncclComm_t[devices_count]);

  std::unique_ptr<cudaStream_t[]> streams (new cudaStream_t[devices_count]);
  std::unique_ptr<unsigned char *[]> inputs (new unsigned char*[devices_count]);
  std::unique_ptr<unsigned char *[]> outputs (new unsigned char*[devices_count]);

  std::iota (devs.get (), devs.get () + devices_count, 0);
  ncclCommInitAll (comms.get (), devices_count, devs.get ());

  for (int i = 0; i < devices_count; i++)
    {
      const unsigned int chunk_begin = get_chunk_begin (img->pixels_count, devices_count, i);
      const unsigned int chunk_end = get_chunk_end (img->pixels_count, devices_count, i);
      const unsigned int chunk_size = chunk_end - chunk_begin;

      cudaSetDevice (devs[i]);
      cudaMalloc (&inputs[i], sizeof (unsigned char) * chunk_size);
      cudaMalloc (&outputs[i], sizeof (unsigned char) * chunk_size);
      cudaMemcpy (inputs[i], img->data.get () + chunk_begin, sizeof (unsigned char) * chunk_size, cudaMemcpyHostToDevice);
      cudaMemset (outputs[i], 0, sizeof (unsigned char) * chunk_size);
      cudaStreamCreate (&streams[i]);
    }

  ncclGroupStart ();
  for (int i = 0; i < devices_count; i++)
    {
      const unsigned int chunk_begin = get_chunk_begin (img->pixels_count, devices_count, i);
      const unsigned int chunk_end = get_chunk_end (img->pixels_count, devices_count, i);
      const unsigned int chunk_size = chunk_end - chunk_begin;

      ncclAllReduce (inputs[i], outputs[i], chunk_size, ncclChar, ncclMin, comms[i], streams[i]);
    }
  ncclGroupEnd ();

  // Synchronize devices
  for (int i = 0; i < devices_count; i++)
    {
      cudaSetDevice (i);
      cudaStreamSynchronize (streams[i]);
    }

  for (int i = 0; i < devices_count; i++)
    {
      cudaSetDevice (devs[i]);
      cudaFree (inputs[i]);
      cudaFree (outputs[i]);
      cudaStreamDestroy (streams[i]);
    }

  // Finalize nccl
  for (int i = 0; i < devices_count; i++)
    ncclCommDestroy (comms[i]);
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

  switch (mode)
    {
      case calculation_mode::cpu:
      case calculation_mode::cpu_mt:
      case calculation_mode::culib:
        std::cerr << "Unsupported mode\n";
        return -1;
      case calculation_mode::nccl:
        nccl_test (img.get ());
    };

  return 0;
}