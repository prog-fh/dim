//----------------------------------------------------------------------------

#ifndef DIM_CPU_DETECT_HPP
#define DIM_CPU_DETECT_HPP 1

#if defined _WIN32
# if !defined _WIN32_WINNT
#   define _WIN32_WINNT _WIN32_WINNT_WIN7
# endif
# if !defined WINVER
#   define WINVER _WIN32_WINNT
# endif
# include <windows.h>
#else
# include <sys/sysctl.h>
# include <unistd.h>
#endif

#if defined __APPLE__
# include <mach/mach.h>
# include <mach/thread_policy.h>
#endif

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <thread>

namespace dim::cpu {

namespace impl_ {

template<typename Tag>
struct SysId
{
  using tag_t = Tag;
  int id{-1};
};

template<typename Tag>
inline
bool
valid(SysId<Tag> sys)
{
  return sys.id!=-1;
}

template<typename Tag>
inline
bool
operator==(SysId<Tag> lhs,
           SysId<Tag> rhs)
{
  return lhs.id==rhs.id;
}

template<typename Tag>
inline
bool
operator!=(SysId<Tag> lhs,
           SysId<Tag> rhs)
{
  return !(lhs==rhs);
}

} // namespace impl_

using NumaId = impl_::SysId<struct NumaIdTag>;
using CpuId = impl_::SysId<struct CpuIdTag>;

struct TopologyGroup
{
  NumaId numa{};
  int cache_level{};
  int cache_size{};
  int cache_line{};
  std::vector<CpuId> cpus{};
  std::vector<TopologyGroup> children{};
  using Path = std::vector<const TopologyGroup *>;
};

namespace impl_ {

template<typename Fnct>
inline
bool
visit_(TopologyGroup::Path &path,
       Fnct fnct)
{
  const auto &const_path=path;
  if(!fnct(*const_path.back(), const_path))
  {
    return false;
  }
  for(const auto &child: path.back()->children)
  {
    path.emplace_back(&child);
    if(!visit_(path, fnct))
    {
      return false;
    }
    path.pop_back();
  }
  return true;
}

} // namespace impl_

inline
bool
is_cpu(const TopologyGroup &grp)
{
  return empty(grp.children)&&(size(grp.cpus)==1);
}

inline
bool
contains(const TopologyGroup &grp,
         CpuId cpu)
{
  return find(cbegin(grp.cpus), end(grp.cpus), cpu)!=cend(grp.cpus);
}

template<typename Fnct>
inline
bool
visit(const TopologyGroup &root,
      Fnct fnct)
{
  auto path=TopologyGroup::Path{&root};
  return impl_::visit_(path, fnct);
}

template<typename Cond>
inline
const TopologyGroup *
find(const TopologyGroup &root,
     Cond cond)
{
  const TopologyGroup *result=nullptr;
  visit(root,
    [&](const auto &grp, const auto &path)
    {
      if(cond(grp, path))
      {
        result=&grp;
        return false;
      }
      return true;
    });
  return result;
}

inline
const TopologyGroup *
find_cache(const TopologyGroup &root,
           CpuId cpu,
           int level)
{
  return find(root,
    [&](const auto &grp, const auto &)
    {
      return (grp.cache_level==level)&&contains(grp, cpu);
    });
}

inline
std::vector<CpuId>
collect_indexth_cpu_of_cache_level(const TopologyGroup &root,
                                   int index,
                                   int level)
{
  auto result=std::vector<CpuId>{};
  for(const auto &cpu: root.cpus)
  {
    if(const auto *cache=find_cache(root, cpu, level))
    {
      auto count=(index<0) ? int(size(cache->cpus))+index : index;
      visit(*cache,
        [&](const auto &grp, const auto &)
        {
          if(is_cpu(grp)&&(count--==0))
          {
            result.emplace_back(grp.cpus.front());
            return false;
          }
          return true;
        });
    }
  }
  return result;
}

inline
std::string
to_string(const TopologyGroup &grp)
{
  auto txt=std::string{};
  visit(grp,
    [&](const auto &grp, const auto &path)
    {
      txt+=std::string(2*(size(path)-1), ' ')+'*';
      if(size(path)==1)
      {
        txt+=" HOST";
      }
      if(grp.cache_level>0)
      {
        txt+=" L"+std::to_string(grp.cache_level)+
             '('+std::to_string(grp.cache_size)+
             '/'+std::to_string(grp.cache_line)+')';
      }
      if(valid(grp.numa))
      {
        txt+=" numa_id("+std::to_string(grp.numa.id)+')';
      }
      if((size(grp.cpus)==1)&&empty(grp.children))
      {
        txt+=" cpu_id("+std::to_string(grp.cpus.front().id)+')';
      }
      txt+='\n';
      return true;
    });
  return txt;
}

inline
std::ostream &
operator<<(std::ostream &output,
           const TopologyGroup &grp)
{
  return output << to_string(grp);
}

inline
bool // success
bind_current_thread(CpuId cpu)
{
  if(cpu.id>=0)
  {
#if defined __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu.id, &cpuset);
    if(::pthread_setaffinity_np(::pthread_self(), sizeof(cpuset), &cpuset)==0)
    {
      return true;
    }
#elif defined _WIN32
    GROUP_AFFINITY aff;
    std::memset(&aff, 0, sizeof(aff));
    aff.Mask=1ULL<<(cpu.id&63);
    aff.Group=WORD(cpu.id>>6);
    if(::SetThreadGroupAffinity(::GetCurrentThread(), &aff, nullptr))
    {
      return true;
    }
#elif defined __APPLE__
    // FIXME: under MacOsX, affinity does not mean binding to a specific CPU!
    thread_affinity_policy_data_t policy;
    policy.affinity_tag=cpu.id;
    if(::thread_policy_set(::pthread_mach_thread_np(::pthread_self()),
                           THREAD_AFFINITY_POLICY,
                           (thread_policy_t)&policy,
                           THREAD_AFFINITY_POLICY_COUNT)==KERN_SUCCESS)
    {
      return true;
    }
#endif
  }
  return false;
}

namespace impl_ {

#if defined CTL_HW
  template<typename T>
  inline
  bool // success
  hw_sysctl_(int hw_name,
             T &result)
  {
    int name[2]={CTL_HW, hw_name};
    auto len=sizeof(result);
    for(;;)
    {
      const auto r=::sysctl(name, 2, &result, &len, nullptr, 0);
      if(r>=0)
      {
        return true;
      }
      if(errno!=EINTR)
      {
        return false;
      }
    }
  }

  template<typename T>
  inline
  bool // success
  sysctl_by_name_(const char *name,
                  T &result)
  {
    auto len=sizeof(result);
    for(;;)
    {
      const auto r=::sysctlbyname(name, &result, &len, nullptr, 0);
      if(r>=0)
      {
        return true;
      }
      if(errno!=EINTR)
      {
        return false;
      }
    }
  }
#endif

#if !defined _WIN32
  template<typename T>
  inline
  bool // success
  sysconf_(int name,
           T &result)
  {
    for(;;)
    {
      const auto r=::sysconf(name);
      if(r>=0)
      {
        result=static_cast<T>(r);
        return true;
      }
      if(errno!=EINTR)
      {
        return false;
      }
    }
  }
#endif

inline
int // detected cpu count or 0 if unknown
detect_cpu_count_()
{
  auto cpu_count=int(std::thread::hardware_concurrency());
#if defined _SC_NPROCESSORS_ONLN
  if(cpu_count==0)
  {
    sysconf_(_SC_NPROCESSORS_ONLN, cpu_count);
  }
#endif
#if defined HW_NCPU
  if(cpu_count==0)
  {
    hw_sysctl_(HW_NCPU, cpu_count);
  }
#endif
#if defined _WIN32
  if(cpu_count==0)
  {
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    cpu_count=info.dwNumberOfProcessors;
  }
#endif
  return cpu_count;
}

inline
int // detected cache size for level or 0 if unknown
detect_cache_size_(int level)
{
  auto cache_size=0;
#if defined _SC_LEVEL1_DCACHE_SIZE
  if((cache_size==0)&&(level==1))
  {
    sysconf_(_SC_LEVEL1_DCACHE_SIZE, cache_size);
  }
#endif
#if defined _SC_LEVEL2_CACHE_SIZE
  if((cache_size==0)&&(level==2))
  {
    sysconf_(_SC_LEVEL2_CACHE_SIZE, cache_size);
  }
#endif
#if defined _SC_LEVEL3_CACHE_SIZE
  if((cache_size==0)&&(level==3))
  {
    sysconf_(_SC_LEVEL3_CACHE_SIZE, cache_size);
  }
#endif
#if defined HW_L1DCACHESIZE
  if((cache_size==0)&&(level==1))
  {
    hw_sysctl_(HW_L1DCACHESIZE, cache_size);
  }
#endif
#if defined HW_L2CACHESIZE
  if((cache_size==0)&&(level==2))
  {
    hw_sysctl_(HW_L2CACHESIZE, cache_size);
  }
#endif
#if defined HW_L3CACHESIZE
  if((cache_size==0)&&(level==3))
  {
    hw_sysctl_(HW_L3CACHESIZE, cache_size);
  }
#endif
  (void)level;
  return cache_size;
}

inline
int // detected cache line for level or 0 if unknown
detect_cache_line_(int level)
{
  auto cache_line=0;
#if defined _SC_LEVEL1_DCACHE_LINESIZE
  if((cache_line==0)&&(level==1))
  {
    sysconf_(_SC_LEVEL1_DCACHE_LINESIZE, cache_line);
  }
#endif
#if defined _SC_LEVEL2_CACHE_LINESIZE
  if((cache_line==0)&&(level==2))
  {
    sysconf_(_SC_LEVEL2_CACHE_LINESIZE, cache_line);
  }
#endif
#if defined _SC_LEVEL3_CACHE_LINESIZE
  if((cache_line==0)&&(level==3))
  {
    sysconf_(_SC_LEVEL3_CACHE_LINESIZE, cache_line);
  }
#endif
#if defined HW_CACHELINE
  if(cache_line==0) // any level
  {
    hw_sysctl_(HW_CACHELINE, cache_line);
  }
#endif
  (void)level;
  return cache_line;
}

} // namespace impl_

} // namespace dim::cpu

#if defined __linux__
# include "cpu_detect_linux.hpp"
#elif defined _WIN32
# include "cpu_detect_windows.hpp"
#else // default empty topology
namespace dim::cpu::impl_ {
  inline TopologyGroup detect_() { return TopologyGroup{}; }
}
#endif

namespace dim::cpu {

inline
TopologyGroup
detect()
{
  auto root=impl_::detect_();
  if(empty(root.cpus)) // fallback to hardcoded topology
  {
    const auto cpu_count=std::max(1, impl_::detect_cpu_count_());
#if defined __APPLE__
    auto value=0;
    impl_::sysctl_by_name_("hw.packages", value);
    const auto pkg_count=std::max(1, value);
    value=0;
    impl_::sysctl_by_name_("machdep.cpu.core_count", value);
    const auto core_count=std::max(1, value);
    value=0;
    impl_::sysctl_by_name_("machdep.cpu.thread_count", value);
    const auto thread_count=std::max(1, value);
    const auto smt_count=(thread_count+core_count-1)/core_count;
#else
    const auto pkg_count=1;
    const auto smt_count=1;
#endif
    for(auto pkg=0; pkg< pkg_count; ++pkg)
    {
      auto &l3_grp=root.children.emplace_back(TopologyGroup{});
      l3_grp.numa=NumaId{pkg};
      l3_grp.cache_level=3; // hardcoded
      const auto cpu_begin=cpu_count*pkg/pkg_count;
      const auto cpu_end=std::min(cpu_count, cpu_count*(pkg+1)/pkg_count);
      for(auto cpu=cpu_begin; cpu<cpu_end; ++cpu)
      {
        l3_grp.cpus.emplace_back(CpuId{cpu});
        root.cpus.emplace_back(CpuId{cpu});
      }
      const auto pkg_core_count=(cpu_end-cpu_begin+smt_count-1)/smt_count;
      for(auto core=0; core<pkg_core_count; ++core)
      {
        auto &l2_grp=l3_grp.children.emplace_back(TopologyGroup{});
        l2_grp.cache_level=2; // hardcoded
        auto &l1_grp=l2_grp.children.emplace_back(TopologyGroup{});
        l1_grp.cache_level=1; // hardcoded
        for(auto smt=0; smt<smt_count; ++smt)
        {
          const auto idx=core*smt_count+smt;
          if(idx<int(size(l3_grp.cpus)))
          {
            l2_grp.cpus.emplace_back(l3_grp.cpus[idx]);
            l1_grp.cpus.emplace_back(l3_grp.cpus[idx]);
            auto &cpu_grp=l1_grp.children.emplace_back(TopologyGroup{});
            cpu_grp.cpus.emplace_back(l3_grp.cpus[idx]);
          }
        }
      }
    }
  }
  int cache_size[3]={0}, cache_line[3]={0};
  for(auto level=1; level<=3; ++level)
  {
    cache_size[level-1]=impl_::detect_cache_size_(level);
    cache_line[level-1]=impl_::detect_cache_line_(level);
  }
  visit(root,
    [&](const auto &grp, const auto &)
    {
      for(auto level=1; level<=3; ++level)
      {
        if(grp.cache_level==level)
        {
          if(grp.cache_size<=0)
          {
            const_cast<int &>(grp.cache_size)=cache_size[level-1];
          }
          if(grp.cache_line<=0)
          {
            const_cast<int &>(grp.cache_line)=cache_line[level-1];
          }
        }
      }
      return true;
    });
  return root;
}

} // namespace dim::cpu

#endif // DIM_CPU_DETECT_HPP

//----------------------------------------------------------------------------
