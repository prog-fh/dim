//----------------------------------------------------------------------------

#ifndef DIM_CPU_DETECT_LINUX_HPP
#define DIM_CPU_DETECT_LINUX_HPP 1

// included from "cpu_detect.hpp"

#include <fstream>
#include <cctype>
#include <algorithm>

namespace dim::cpu_detect::impl_ {

inline
std::string
first_line_(const std::string &path)
{
  auto result=std::string{};
  auto input=std::ifstream{path};
  if(input)
  {
    auto line=std::string{};
    if(getline(input, line))
    {
      const auto not_space=
        [&](const auto &c)
        {
          return !std::isspace(c);
        };
      const auto b=std::find_if(cbegin(line), cend(line), not_space);
      const auto e=std::find_if(crbegin(line), crend(line), not_space);
      std::for_each(b, e.base(),
        [&](const auto & c)
        {
          result+=char(std::tolower(c));
        });
    }
  }
  return result;
}

inline
std::vector<int>
read_list_(const std::string &path)
{
  auto result=std::vector<int>{};
  const auto line=first_line_(path);
  for(auto b=cbegin(line), e=b;
      (e=std::find(b, cend(line), ','))!=b;
      b=(e!=cend(line)) ? e+1 : e)
  {
    const auto dash=std::find(b, e, '-');
    const auto first=std::stoi(std::string{b, dash});
    const auto last=(dash==e) ? first : std::stoi(std::string{dash+1, e});
    for(auto id=first; id<=last; ++id)
    {
      result.emplace_back(id);
    }
  }
  return result;
}

inline
int
read_int_(const std::string &path)
{
  auto line=first_line_(path);
  auto factor=1;
  if(!empty(line))
  {
    if(line.back()=='k')
    {
      factor=1024;
      line.pop_back();
    }
    else if(line.back()=='m')
    {
      factor=1024*1024;
      line.pop_back();
    }
  }
  try
  {
    return factor*std::stoi(line);
  }
  catch(...)
  {
    return -1;
  }
}

template<typename CacheFnct>
inline
void
iterate_cache_(int cpu_id,
               CacheFnct fnct)
{
  const auto cpu_path=
    "/sys/devices/system/cpu/cpu"+std::to_string(cpu_id);
  for(auto idx=0; ; ++idx)
  {
    const auto cache_path=cpu_path+"/cache/index"+std::to_string(idx);
    const auto type=first_line_(cache_path+"/type");
    if(empty(type))
    {
      break;
    }
    if((type=="data")||(type=="unified"))
    {
      fnct(cache_path, read_int_(cache_path+"/level"));
    }
  }
}

inline
void
collect_next_level_(CpuGroup &grp)
{
  for(const auto &cpu_id: grp.cpu_id)
  {
    if(grp.cache_level<=1)
    {
      auto &child=grp.children.emplace_back(CpuGroup{});
      child.cpu_id.emplace_back(cpu_id);
    }
    else
    {
      iterate_cache_(cpu_id,
        [&](const auto &cache_path, const auto &cache_level)
        {
          auto cache_size=read_int_(cache_path+"/size");
          auto cache_line=read_int_(cache_path+"/coherency_line_size");
          if(cache_level==grp.cache_level)
          {
            grp.cache_size=cache_size;
            grp.cache_line=cache_line;
          }
          if(cache_level==grp.cache_level-1)
          {
            auto cpu_list=read_list_(cache_path+"/shared_cpu_list");
            auto found=std::find_if(
              cbegin(grp.children), cend(grp.children),
              [&](const auto &child)
              {
                return child.cpu_id==cpu_list;
              });
            if(found==cend(grp.children))
            {
              auto &child=grp.children.emplace_back(CpuGroup{});
              child.cache_level=cache_level;
              child.cache_size=cache_size;
              child.cache_line=cache_line;
              child.cpu_id=cpu_list;
              collect_next_level_(child);
            }
          }
        });
    }
  }
}

inline
std::tuple<CpuGroup,
           std::vector<int>>
detect_()
{
  auto grp=CpuGroup{};
  grp.cpu_id=read_list_("/sys/devices/system/cpu/possible");
  auto max_cache_level=-1;
  for(const auto &cpu_id: grp.cpu_id)
  {
    iterate_cache_(cpu_id,
      [&](const auto &cache_path, const auto &cache_level)
      {
        (void)cache_path;
        max_cache_level=std::max(max_cache_level, cache_level);
      });
  }
  for(const auto &numa_id: read_list_("/sys/devices/system/node/possible"))
  {
    const auto numa_path=
      "/sys/devices/system/node/node"+std::to_string(numa_id);
    auto &child=grp.children.emplace_back(CpuGroup{});
    child.numa_id=numa_id;
    child.cache_level=max_cache_level;
    child.cpu_id=read_list_(numa_path+"/cpulist");
    collect_next_level_(child);
  }
  return {std::move(grp), read_list_("/sys/devices/system/cpu/online")};
}

} // namespace dim::cpu_detect::impl_

#endif // DIM_CPU_DETECT_LINUX_HPP

//----------------------------------------------------------------------------
