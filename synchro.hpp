//----------------------------------------------------------------------------

#ifndef DIM_SYNCHRO_HPP
#define DIM_SYNCHRO_HPP

#include <cstdint>
#include <atomic>

namespace dim {

namespace impl_ {

void
cpu_pause_()
{
#if defined _MSC_VER
# if defined _M_IX86 || defined __INTEL_COMPILER
  __asm { pause }
# else
  __nop();
# endif
#elif defined __POWERPC__
   __asm__ __volatile__ ("ori r0,r0,0"); // fake nop
#elif defined __i386__ || defined __x86_64__
  __asm__ __volatile__ ("pause");
#elif 0 // may not be available everywhere
  __builtin_ia32_pause();
#else
  // just do nothing...
#endif
}

} // namespace impl_

class SpinLock
{
public:

  SpinLock()
  : flag_{free_flag_}
  {
    // nothing more to be done
  }

  bool // success
  try_lock_w()
  {
    auto expected=free_flag_;
    return flag_.compare_exchange_weak(expected, 0,
                                       std::memory_order_acquire,
                                       std::memory_order_relaxed);
  }

  void
  lock_w()
  {
    while(!try_lock_w())
    {
      while(flag_.load(std::memory_order_relaxed)!=free_flag_)
      {
        impl_::cpu_pause_();
      }
    }
  }

  void
  unlock_w()
  {
    flag_.fetch_add(free_flag_, std::memory_order_release);
  }

  bool // success
  try_lock_r()
  {
    if(flag_.fetch_add(-1, std::memory_order_acquire)<1)
    {
      flag_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    else
    {
      return true;
    }
  }

  void
  lock_r()
  {
    while(!try_lock_r())
    {
      while(flag_.load(std::memory_order_relaxed)<=0)
      {
        impl_::cpu_pause_();
      }
    }
  }

  void
  unlock_r()
  {
    flag_.fetch_add(1, std::memory_order_release);
  }

  bool // success
  try_upgrade()
  {
    auto expected=free_flag_-1;
    return flag_.compare_exchange_weak(expected, 0,
                                       std::memory_order_acquire,
                                       std::memory_order_relaxed);
  }

  void
  upgrade()
  {
    while(!try_upgrade())
    {
      while(flag_.load(std::memory_order_relaxed)!=free_flag_-1)
      {
        impl_::cpu_pause_();
      }
    }
  }

  void
  downgrade()
  {
    flag_.fetch_add(free_flag_-1, std::memory_order_release);
  }

private:

  using flag_t = std::int32_t;

  static constexpr auto free_flag_=flag_t{0x01000000};

  std::atomic<flag_t> flag_;
};

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
