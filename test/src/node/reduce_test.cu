#include "gtest/gtest.h"
#include "test_helper.cuh"

#include <cuda_runtime.h>

#include "culib/node/reduce.cuh"

#include <vector>
#include <thread>

template <typename data_type>
void perform_node_reduce_test (size_t elements_count)
{
  int devices_count {};
  cudaGetDeviceCount (&devices_count);

  if (devices_count > 1)
    {
      std::vector<int> devices;
      for (int gpu = 0; gpu < devices_count; gpu++)
        devices.push_back (gpu);

      culib::node::node_communicator<> node_comm (devices);

      std::vector<std::thread> threads;

      for (int thread = 0; thread < devices_count; thread++)
        {
          threads.push_back (std::thread ([&, thread] () {
            std::vector<data_type> h_in (elements_count, 1);

            data_type *d_in {};
            data_type *d_workspace {};
            cudaMalloc (&d_in, elements_count * sizeof (data_type));
            cudaMalloc (&d_workspace, culib::node::reducer<data_type>::get_gpu_workspace_size_in_bytes (elements_count));
            cudaMemcpy (d_in, h_in.data (), elements_count * sizeof (data_type), cudaMemcpyHostToDevice);

            auto comm = node_comm.get_comm (thread);
            culib::node::reducer<data_type> reducer (d_workspace, comm);
            data_type result = reducer.reduce_from_host (elements_count, d_in);

            EXPECT_EQ (result, devices_count * elements_count);

            cudaFree (d_workspace);
            cudaFree (d_in);
          }));
        }

      for (auto &thread: threads)
        thread.join ();
    }
}

TEST(node_reduce, multiple_blocks_int) { perform_node_reduce_test<int> (10000); }
