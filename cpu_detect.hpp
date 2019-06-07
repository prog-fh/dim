//----------------------------------------------------------------------------

#ifndef DIM_CPU_DETECT_HPP
#define DIM_CPU_DETECT_HPP 1

#include <vector>
#include <string>
#include <iostream>
#include <thread>
#include <tuple>

namespace dim::cpu_detect {

struct CpuGroup
{
  int numa_id{-1};
  int cache_level{-1};
  int cache_size{-1};
  int cache_line{-1};
  std::vector<int> cpu_id{};
  std::vector<CpuGroup> children{};

  template<typename Fnct>
  bool
  visit(Fnct fnct) const
  {
    auto path=std::vector<const CpuGroup *>{};
    return visit_(path, fnct);
  }

  const CpuGroup &
  first_leaf() const
  {
    return empty(children) ? *this : children.front().first_leaf();
  }

  const CpuGroup &
  last_leaf() const
  {
    return empty(children) ? *this : children.back().last_leaf();
  }

private:

  template<typename Fnct>
  bool
  visit_(std::vector<const CpuGroup *> &path,
         Fnct fnct) const
  {
    const auto &const_path=path;
    if(!fnct(*this, const_path))
    {
      return false;
    }
    path.emplace_back(this);
    for(const auto &child: children)
    {
      if(!child.visit_(path, fnct))
      {
        return false;
      }
    }
    path.pop_back();
    return true;
  }
};

inline
std::string
to_string(const CpuGroup &grp)
{
  auto txt=std::string{};
  grp.visit([&](const auto &grp, const auto &path)
    {
      txt+=std::string(2*size(path), ' ')+'*';
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
      if(grp.numa_id!=-1)
      {
        txt+=" numa_id("+std::to_string(grp.numa_id)+')';
      }
      if((size(grp.cpu_id)==1)&&empty(grp.children))
      {
        txt+=" cpu_id("+std::to_string(grp.cpu_id[0])+')';
      }
      txt+='\n';
      return true;
    });
  return txt;
}

inline
std::ostream &
operator<<(std::ostream &output,
           const CpuGroup &grp)
{
  return output << to_string(grp);
}

} // namespace dim::cpu_detect

#if defined __linux__
# include "cpu_detect_linux.hpp"
#elif defined XX_WIN32 // FIXME: not implemented
# include "cpu_detect_windows.hpp"
#else
namespace dim::cpu_detect::impl_ {

inline
std::tuple<CpuGroup,
           std::vector<int>>
detect_()
{
  return {CpuGroup{}, std::vector<int>{}};
}

} // namespace dim::cpu_detect::impl_
#endif

namespace dim::cpu_detect {

inline
std::tuple<CpuGroup,
           std::vector<int>>
detect()
{
  auto [root, online]=impl_::detect_();
  if(empty(root.cpu_id)) // fallback to flat topology
  {
    const auto cpu_count=std::max(1, int(std::thread::hardware_concurrency()));
    for(auto cpu=0; cpu<cpu_count; ++cpu)
    {
      auto &child=root.children.emplace_back(CpuGroup{});
      child.cpu_id.emplace_back(cpu);
      root.cpu_id.emplace_back(cpu);
    }
    // FIXME: detect other properties?
  }
  return {std::move(root), std::move(online)};
}

} // namespace dim::cpu_detect


#endif // DIM_CPU_DETECT_HPP

//----------------------------------------------------------------------------
