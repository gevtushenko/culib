//
// Created by egi on 6/18/20.
//

#ifndef CULIB_COMMUNICATION_H
#define CULIB_COMMUNICATION_H

#include "culib/device/memory/api.h"

#include <vector>
#include <atomic>

namespace culib
{
namespace node
{

template <class sync_policy>
class node_communicator;

class device_communicator
{
  const int gpu_id {};
public:
  device_communicator () = delete;

  int get_gpu_id () { return gpu_id; }

  template <typename data_type>
  void put (const data_type *src, size_t n, data_type *dst) const
  {
    culib::device::copy_n (src, n, dst);
  }

  template <typename data_type>
  void get (const data_type *src, size_t n, data_type *dst) const
  {
    culib::device::copy_n (src, n, dst);
  }

protected:
  explicit device_communicator (int gpu_id_arg)
    : gpu_id (gpu_id_arg)
  { }

  template <class sync_policy>
  friend class node_communicator;
};

class atomic_threads_synchronizer
{
  const unsigned int total_threads {};
  std::atomic<unsigned int> barrier_epoch;
  std::atomic<unsigned int> threads_in_barrier;

public:
  atomic_threads_synchronizer () = delete;
  explicit atomic_threads_synchronizer (unsigned int threads_count);

  void barrier ();
};

template <class sync_policy = atomic_threads_synchronizer>
class node_communicator : public sync_policy
{
  const std::vector<int> gpu_ids;

public:
  node_communicator () = delete;
  node_communicator (const std::vector<int> &gpu_ids_arg)
    : sync_policy (gpu_ids_arg.size ())
    , gpu_ids (gpu_ids_arg)
  {
    const std::string error_msg = "culib: p2p isn't supported";

    for (auto &first_gpu_id: gpu_ids)
      {
        cudaSetDevice (first_gpu_id);

        for (auto &second_gpu_id: gpu_ids)
          {
            if (first_gpu_id != second_gpu_id)
              {
                int can_access_peer {};
                cudaDeviceCanAccessPeer (&can_access_peer, first_gpu_id, second_gpu_id);

                if (!can_access_peer || cudaDeviceEnablePeerAccess (second_gpu_id, 0) != cudaSuccess)
                  throw std::runtime_error (error_msg);
              }
          }
      }
  }

  ~node_communicator () noexcept(false)
  {
    const std::string error_msg = "culib: can't disable p2p access";

    for (auto &first_gpu_id: gpu_ids)
      {
        cudaSetDevice (first_gpu_id);

        for (auto &second_gpu_id: gpu_ids)
          {
            if (first_gpu_id != second_gpu_id)
              {
                int can_access_peer {};
                cudaDeviceCanAccessPeer (&can_access_peer, first_gpu_id, second_gpu_id);

                if (can_access_peer && cudaDeviceDisablePeerAccess (second_gpu_id) != cudaSuccess)
                  throw std::runtime_error (error_msg);
              }
          }
      }
  }

  device_communicator get_comm (int thread_id) const noexcept(false)
  {
    if (thread_id >= static_cast<int> (gpu_ids.size ()))
      throw std::runtime_error ("culib: communicator was created for " +
                                std::to_string (gpu_ids.size ()) + " devices, trying to access " +
                                std::to_string (thread_id));
    const int device_id = gpu_ids[thread_id];
    if (cudaSetDevice (device_id) != cudaSuccess)
      throw std::runtime_error ("culib: can't set device " + std::to_string (device_id));

    return device_communicator (device_id);
  }
};

}
}

#endif //CULIB_COMMUNICATION_H
