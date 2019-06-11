//----------------------------------------------------------------------------

#ifndef DIM_CPU_DETECT_HPP
#define DIM_CPU_DETECT_HPP 1

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
  int cache_level{-1};
  int cache_size{-1};
  int cache_line{-1};
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
      if(empty(path))
      {
        txt+=" HOST";
      }
      if(grp.cache_level!=-1)
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

} // namespace dim::cpu

#if defined __linux__
# include "cpu_detect_linux.hpp"
#elif defined XX_WIN32 // FIXME: not implemented
# include "cpu_detect_windows.hpp"
#else
namespace dim::cpu::impl_ {

inline
TopologyGroup
detect_()
{
  return TopologyGroup{};
}

} // namespace dim::cpu::impl_
#endif

namespace dim::cpu {

inline
TopologyGroup
detect()
{
  auto root=impl_::detect_();
  if(empty(root.cpus)) // fallback to flat topology
  {
    const auto cpu_count=std::max(1, int(std::thread::hardware_concurrency()));
    for(auto cpu=0; cpu<cpu_count; ++cpu)
    {
      auto &child=root.children.emplace_back(TopologyGroup{});
      child.cpus.emplace_back(CpuId{cpu});
      root.cpus.emplace_back(CpuId{cpu});
    }
    // FIXME: detect other properties?
  }
  return root;
}

} // namespace dim::cpu


#endif // DIM_CPU_DETECT_HPP

//----------------------------------------------------------------------------
