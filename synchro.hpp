//----------------------------------------------------------------------------

#ifndef DIM_SYNCHRO_HPP
#define DIM_SYNCHRO_HPP

#include <atomic>

namespace dim {

class Synchro
{
public:

  using sync_t = unsigned int; // overflow is correct

  Synchro()
  : sync_{}
  , ack_count_{}
  {
    // nothing more to be done
  }

  void
  sync(int thread_count)
  {
    ack_count_.store(thread_count-1, std::memory_order_release);
    sync_.fetch_add(1, std::memory_order_release);
  }

  void
  wait_for_sync(sync_t &last_sync)
  {
    for(;;)
    {
      if(const auto sync=sync_.load(std::memory_order_acquire);
         last_sync!=sync)
      {
        last_sync=sync;
        break;
      }
    }
  }

  void
  ack()
  {
    ack_count_.fetch_sub(1, std::memory_order_release);
  }

  void
  wait_for_ack()
  {
    while(ack_count_.load(std::memory_order_acquire)!=0)
    {
      // busy wait
    }
  }

private:
  std::atomic<sync_t> sync_;
  std::atomic<int> ack_count_;
};

} // namespace dim

#endif // DIM_SYNCHRO_HPP

//----------------------------------------------------------------------------
