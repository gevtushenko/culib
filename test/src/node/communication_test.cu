#include "gtest/gtest.h"

#include <cuda_runtime.h>

#include "culib/node/communication.h"
#include "culib/device/memory/resizable_array.h"

#include <vector>
#include <thread>
#include <atomic>

using data_type = int;

TEST(node_communication, atomic_barrier)
{
  std::atomic<unsigned int> counter (0);
  std::vector<std::thread> threads;

  const unsigned int steps_count = 100;
  const unsigned int threads_count = std::thread::hardware_concurrency ();
  culib::node::atomic_threads_synchronizer sync (threads_count);

  for (unsigned int thread_id = 0; thread_id < threads_count; thread_id++)
    {
      threads.push_back (std::thread ([&] () {
        for (unsigned int step = 0; step < steps_count; step++)
          {
            counter.fetch_add (1);
            sync.barrier ();
            EXPECT_EQ (counter.load (), threads_count);
            sync.barrier ();
            counter.fetch_sub (1);
            sync.barrier ();
            EXPECT_EQ (counter.load (), 0u);
            sync.barrier ();
          }
      }));
    }

  for (auto &thread: threads)
    thread.join ();
}

TEST(node_communication, put)
{
  int n = 42;
  int devices_count {};
  cudaGetDeviceCount (&devices_count);

  if (devices_count > 1)
    {
      std::vector<int> devices;
      for (int gpu = 0; gpu < devices_count; gpu++)
        devices.push_back (gpu);

      culib::node::node_communicator<> node_comm (devices);
      std::vector<data_type*> src (devices_count, nullptr);
      std::vector<data_type*> dst (devices_count, nullptr);

      // allocate
      {
        std::vector<std::thread> threads;
        for (int tid = 0; tid < devices_count; tid++)
          {
            threads.push_back (std::thread ([tid, n, &node_comm, &src, &dst] () {
              culib::node::device_communicator comm = node_comm.get_comm (tid);
              cudaSetDevice (comm.get_gpu_id ());

              culib::device::resizable_array<data_type> buffer_to_send; buffer_to_send.resize (n);
              culib::device::resizable_array<data_type> buffer_to_resv; buffer_to_resv.resize (n);
              src[comm.get_gpu_id ()] = buffer_to_send.release ();
              dst[comm.get_gpu_id ()] = buffer_to_resv.release ();
            }));
          }

        for (auto &thread: threads)
          thread.join ();
      }

      // send
      {
        std::vector<std::thread> threads;
        for (int tid = 0; tid < devices_count; tid++)
          {
            threads.push_back (std::thread ([tid, n, devices_count, &node_comm, &src, &dst] () {
              culib::node::device_communicator comm = node_comm.get_comm (tid);
              cudaSetDevice (comm.get_gpu_id ());

              // send to the right
              data_type *src_ptr = src[comm.get_gpu_id ()];
              data_type *dst_ptr = dst[(comm.get_gpu_id () + 1) % devices_count];

              for (int i = 0; i < n; i++)
                culib::device::send (i + comm.get_gpu_id (), src_ptr + i);

              comm.put (src_ptr, n, dst_ptr);
            }));
          }

        for (auto &thread: threads)
          thread.join ();
      }

      // check
      {
        std::vector<std::thread> threads;
        for (int tid = 0; tid < devices_count; tid++)
          {
            threads.push_back (std::thread ([tid, n, devices_count, &node_comm, &src, &dst] () {
              culib::node::device_communicator comm = node_comm.get_comm (tid);
              cudaSetDevice (comm.get_gpu_id ());

              data_type *result_ptr = dst[comm.get_gpu_id ()];
              std::unique_ptr<data_type[]> cpu_data (new data_type[n]);
              culib::device::recv_n (result_ptr, n, cpu_data.get ());

              const int source_device_id = comm.get_gpu_id () > 0
                                         ? comm.get_gpu_id () - 1
                                         : devices_count - 1;

              for (int i = 0; i < n; i++)
                EXPECT_EQ (i + source_device_id, cpu_data[i]);
            }));
          }

        for (auto &thread: threads)
          thread.join ();
      }

      // free
      {
        std::vector<std::thread> threads;
        for (int tid = 0; tid < devices_count; tid++)
          {
            threads.push_back (std::thread ([tid, n, &node_comm, &src, &dst] () {
              culib::node::device_communicator comm = node_comm.get_comm (tid);
              cudaSetDevice (comm.get_gpu_id ());

              culib::device::pointer<data_type> dst_ptr (dst[comm.get_gpu_id ()]);
              culib::device::pointer<data_type> src_ptr (src[comm.get_gpu_id ()]);
            }));
          }

        for (auto &thread: threads)
          thread.join ();
      }
    }
}

TEST(node_communication, get)
{
  int n = 42;
  int devices_count {};
  cudaGetDeviceCount (&devices_count);

  if (devices_count > 1)
    {
      std::vector<int> devices;
      for (int gpu = 0; gpu < devices_count; gpu++)
        devices.push_back (gpu);

      culib::node::node_communicator<> node_comm (devices);
      std::vector<data_type*> src (devices_count, nullptr);
      std::vector<data_type*> dst (devices_count, nullptr);

      // allocate
      {
        std::vector<std::thread> threads;
        for (int tid = 0; tid < devices_count; tid++)
          {
            threads.push_back (std::thread ([tid, n, &node_comm, &src, &dst] () {
              culib::node::device_communicator comm = node_comm.get_comm (tid);
              cudaSetDevice (comm.get_gpu_id ());

              culib::device::resizable_array<data_type> buffer_to_send; buffer_to_send.resize (n);
              culib::device::resizable_array<data_type> buffer_to_resv; buffer_to_resv.resize (n);
              src[comm.get_gpu_id ()] = buffer_to_send.release ();
              dst[comm.get_gpu_id ()] = buffer_to_resv.release ();
            }));
          }

        for (auto &thread: threads)
          thread.join ();
      }

      // fill
      {
        std::vector<std::thread> threads;
        for (int tid = 0; tid < devices_count; tid++)
          {
            threads.push_back (std::thread ([tid, n, devices_count, &node_comm, &src, &dst] () {
              culib::node::device_communicator comm = node_comm.get_comm (tid);
              cudaSetDevice (comm.get_gpu_id ());

              // send to the right
              data_type *src_ptr = src[comm.get_gpu_id ()];

              for (int i = 0; i < n; i++)
                culib::device::send (i + comm.get_gpu_id (), src_ptr + i);
            }));
          }

        for (auto &thread: threads)
          thread.join ();
      }

      // get and check
      {
        std::vector<std::thread> threads;
        for (int tid = 0; tid < devices_count; tid++)
          {
            threads.push_back (std::thread ([tid, n, devices_count, &node_comm, &src, &dst] () {
              culib::node::device_communicator comm = node_comm.get_comm (tid);
              cudaSetDevice (comm.get_gpu_id ());

              data_type *src_ptr = src[(comm.get_gpu_id () + 1) % devices_count];
              data_type *dst_ptr = dst[comm.get_gpu_id ()];
              comm.get (src_ptr, n, dst_ptr);

              data_type *result_ptr = dst[comm.get_gpu_id ()];
              std::unique_ptr<data_type[]> cpu_data (new data_type[n]);
              culib::device::recv_n (result_ptr, n, cpu_data.get ());

              const int source_device_id = comm.get_gpu_id () > 0
                                           ? comm.get_gpu_id () - 1
                                           : devices_count - 1;

              for (int i = 0; i < n; i++)
                EXPECT_EQ (i + source_device_id, cpu_data[i]);
            }));
          }

        for (auto &thread: threads)
          thread.join ();
      }

      // free
      {
        std::vector<std::thread> threads;
        for (int tid = 0; tid < devices_count; tid++)
          {
            threads.push_back (std::thread ([tid, n, &node_comm, &src, &dst] () {
              culib::node::device_communicator comm = node_comm.get_comm (tid);
              cudaSetDevice (comm.get_gpu_id ());

              culib::device::pointer<data_type> dst_ptr (dst[comm.get_gpu_id ()]);
              culib::device::pointer<data_type> src_ptr (src[comm.get_gpu_id ()]);
            }));
          }

        for (auto &thread: threads)
          thread.join ();
      }
    }
}
