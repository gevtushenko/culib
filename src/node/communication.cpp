//
// Created by egi on 6/21/20.
//

#include "culib/node/communication.h"

#include <xmmintrin.h> // For _mm_pause

using namespace culib;
using namespace node;

atomic_threads_synchronizer::atomic_threads_synchronizer (unsigned int threads_count)
  : total_threads (threads_count)
  , barrier_epoch (0u)
  , threads_in_barrier (0u)
  , buffer (new void *[threads_count])
{ }

void atomic_threads_synchronizer::barrier()
{
  const unsigned int thread_epoch = barrier_epoch.load ();

  if (threads_in_barrier.fetch_add (1u) == total_threads - 1)
    {
      threads_in_barrier.store (0);
      barrier_epoch.fetch_add (1u);
    }
  else
    {
      while (thread_epoch == barrier_epoch.load ())
        _mm_pause ();
    }
}
