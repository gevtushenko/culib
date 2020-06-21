//
// Created by egi on 6/18/20.
//

#ifndef CULIB_COMMUNICATION_H
#define CULIB_COMMUNICATION_H

#include "culib/device/memory/api.h"

#include <vector>
#include <atomic>
#include <memory>

namespace culib
{
namespace node
{

template <class sync_policy>
class node_communicator;

template <class sync_policy>
class device_communicator
{
  const int gpu_id {};
  const node_communicator<sync_policy> &node_comm;

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

  template<typename data_type>
  data_type host_reduce_sum (data_type value) const
  {
    return node_comm.host_reduce_sum (gpu_id, value);
  }

protected:
  device_communicator (
    int gpu_id_arg,
    const node_communicator<sync_policy> &node_comm_arg)
    : gpu_id (gpu_id_arg)
    , node_comm (node_comm_arg)
  { }

  friend class node_communicator<sync_policy>;
};

class atomic_threads_synchronizer
{
  const unsigned int total_threads {};
  mutable std::atomic<unsigned int> barrier_epoch;
  mutable std::atomic<unsigned int> threads_in_barrier;
  mutable std::unique_ptr<void*[]> buffer;

  template<typename data_type>
  data_type &get_buffer (unsigned int thread_id) const
  {
    return *reinterpret_cast<data_type*> (buffer[thread_id]);
  }

public:
  atomic_threads_synchronizer () = delete;
  explicit atomic_threads_synchronizer (unsigned int threads_count);

  template<typename data_type>
  data_type reduce_sum (unsigned int thread_id, data_type value) const
  {
    const unsigned int main_thread = 0;
    buffer[thread_id] = &value;
    barrier ();

    if (thread_id == main_thread)
      for (unsigned int thread = main_thread + 1; thread < total_threads; thread++)
        value += get_buffer<data_type> (thread);

    barrier ();
    value = get_buffer<data_type> (main_thread);
    barrier ();

    return value;
  }

  void barrier () const;
};

template <class sync_policy = atomic_threads_synchronizer>
class node_communicator : protected sync_policy
{
  const std::vector<int> gpu_ids;

protected:
  template<typename data_type>
  data_type host_reduce_sum (unsigned int gpu_id, data_type value) const
  {
    return sync_policy::reduce_sum (gpu_id, value);
  }

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

  device_communicator<sync_policy> get_comm (int thread_id) const noexcept(false)
  {
    if (thread_id >= static_cast<int> (gpu_ids.size ()))
      throw std::runtime_error ("culib: communicator was created for " +
                                std::to_string (gpu_ids.size ()) + " devices, trying to access " +
                                std::to_string (thread_id));
    const int device_id = gpu_ids[thread_id];
    if (cudaSetDevice (device_id) != cudaSuccess)
      throw std::runtime_error ("culib: can't set device " + std::to_string (device_id));

    return { device_id, *this };
  }

  friend class device_communicator<sync_policy>;
};

}
}

#endif //CULIB_COMMUNICATION_H
