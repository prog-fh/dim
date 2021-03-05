//----------------------------------------------------------------------------

#ifndef DIM_CPU_DETECT_LINUX_HPP
#define DIM_CPU_DETECT_LINUX_HPP

// included from "cpu_detect.hpp"

#include <fstream>
#include <cctype>

namespace dim::cpu::impl_ {

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
      const auto b=find_if(cbegin(line), cend(line), not_space);
      const auto e=find_if(crbegin(line), crend(line), not_space);
      for_each(b, e.base(),
        [&](const auto & c)
        {
          result+=char(std::tolower(c));
        });
    }
  }
  return result;
}

template<typename SysIdType>
inline
std::vector<SysIdType>
read_list_(const std::string &path,
           const std::vector<SysIdType> &filter={})
{
  auto result=std::vector<SysIdType>{};
  const auto line=first_line_(path);
  for(auto b=cbegin(line), e=b;
      (e=find(b, cend(line), ','))!=b;
      b=(e!=cend(line)) ? e+1 : e)
  {
    const auto dash=find(b, e, '-');
    const auto first=std::stoi(std::string{b, dash});
    const auto last=(dash==e) ? first : std::stoi(std::string{dash+1, e});
    for(auto i=first; i<=last; ++i)
    {
      const auto id=SysIdType{i};
      if(empty(filter)||
         (find(cbegin(filter), cend(filter), id)!=cend(filter)))
      {
        result.emplace_back(id);
      }
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
iterate_cache_(CpuId cpu,
               CacheFnct fnct)
{
  const auto cpu_path=
    "/sys/devices/system/cpu/cpu"+std::to_string(cpu.id);
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
collect_next_level_(TopologyGroup &grp)
{
  for(const auto &cpu: grp.cpus)
  {
    if(grp.cache_level<=1)
    {
      auto &child=grp.children.emplace_back(TopologyGroup{});
      child.cpus.emplace_back(cpu);
    }
    else
    {
      iterate_cache_(cpu,
        [&](const auto &cache_path, const auto &cache_level)
        {
          auto cache_size=read_int_(cache_path+"/size");
          auto cache_line=read_int_(cache_path+"/coherency_line_size");
          if(cache_level==grp.cache_level)
          {
            grp.cache_size=cache_size;
            grp.cache_line=cache_line;
          }
          else if(cache_level==grp.cache_level-1)
          {
            auto cpu_list=read_list_(cache_path+"/shared_cpu_list",
                                     grp.cpus);
            auto found=find_if(
              cbegin(grp.children), cend(grp.children),
              [&](const auto &child)
              {
                return child.cpus==cpu_list;
              });
            if(found==cend(grp.children))
            {
              auto &child=grp.children.emplace_back(TopologyGroup{});
              child.cache_level=cache_level;
              child.cache_size=cache_size;
              child.cache_line=cache_line;
              child.cpus=std::move(cpu_list);
              collect_next_level_(child);
            }
          }
        });
    }
  }
}

inline
TopologyGroup
detect_()
{
  auto root=TopologyGroup{};
  root.cpus=read_list_<CpuId>("/sys/devices/system/cpu/online");
  const auto online_numas=
    read_list_<NumaId>("/sys/devices/system/node/online");
  if(!empty(online_numas))
  {
    auto max_cache_level=0;
    for(const auto &cpu: root.cpus)
    {
      iterate_cache_(cpu,
        [&](const auto &cache_path, const auto &cache_level)
        {
          (void)cache_path;
          max_cache_level=std::max(max_cache_level, cache_level);
        });
    }
    for(const auto &numa: online_numas)
    {
      const auto numa_path=
        "/sys/devices/system/node/node"+std::to_string(numa.id);
      auto &child=root.children.emplace_back(TopologyGroup{});
      child.numa=numa;
      child.cpus=read_list_(numa_path+"/cpulist", root.cpus);
      child.cache_level=max_cache_level;
      collect_next_level_(child);
    }
  }
  else // probably Windows-Subsystem-for-Linux or Raspbian
  {
    for(const auto &cpu: root.cpus)
    {
      const auto pkg=read_int_("/sys/devices/system/cpu/cpu"+
                               std::to_string(cpu.id)+
                               "/topology/physical_package_id");
      const auto it=find_if(begin(root.children), end(root.children),
        [&](const auto &grp)
        {
          return grp.numa.id==pkg;
        });
      auto &l3_grp=(it!=end(root.children)) ? *it
                   : root.children.emplace_back(TopologyGroup{});
      l3_grp.numa=NumaId{pkg}; // dummy numa node from package id
      l3_grp.cpus.emplace_back(cpu);
    }
    for(auto &l3_grp: root.children)
    {
      l3_grp.cache_level=3; // hardcoded
      for(const auto &cpu: l3_grp.cpus)
      {
        const auto core=read_int_("/sys/devices/system/cpu/cpu"+
                                  std::to_string(cpu.id)+
                                  "/topology/core_id");
        const auto it=find_if(begin(l3_grp.children), end(l3_grp.children),
          [&](const auto &grp)
          {
            return grp.numa==NumaId{core}; // temporary
          });
        auto &l2_grp=(it!=end(l3_grp.children)) ? *it
                     : l3_grp.children.emplace_back(TopologyGroup{});
        l2_grp.numa=NumaId{core}; // temporary
        l2_grp.cpus.emplace_back(cpu);
      }
      for(auto &l2_grp: l3_grp.children)
      {
        l2_grp.numa=NumaId{}; // reset temporary
        l2_grp.cache_level=2; // hardcoded
        auto &l1_grp=l2_grp.children.emplace_back(TopologyGroup{});
        l1_grp.cache_level=1; // hardcoded
        l1_grp.cpus=l2_grp.cpus;
        for(const auto &cpu: l1_grp.cpus)
        {
          auto &cpu_grp=l1_grp.children.emplace_back(TopologyGroup{});
          cpu_grp.cpus.emplace_back(cpu);
        }
      }
    }
  }
  return root;
}

} // namespace dim::cpu::impl_

#endif // DIM_CPU_DETECT_LINUX_HPP

//----------------------------------------------------------------------------
