//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifndef DIM_CPU_DETECT_WINDOWS_HPP
#define DIM_CPU_DETECT_WINDOWS_HPP

// included from "cpu_detect.hpp"

#include <cstdlib>

namespace dim::cpu::impl_ {

struct ProcInfoBuffer_
{
  using info_t = SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX;
  info_t *storage;
  DWORD size;

  ProcInfoBuffer_()
  : storage{}
  , size{}
  {
    if(!::GetLogicalProcessorInformationEx(RelationAll, nullptr, &size)&&
       (::GetLastError()==ERROR_INSUFFICIENT_BUFFER))
    {
      storage=(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)std::malloc(size);
    }
    if(!::GetLogicalProcessorInformationEx(RelationAll, storage, &size))
    {
      std::free(storage);
      storage=nullptr;
      size=0;
    }
  }

  ~ProcInfoBuffer_()
  {
    std::free(storage);
  }

  ProcInfoBuffer_(const ProcInfoBuffer_ &) =delete;
  ProcInfoBuffer_& operator=(const ProcInfoBuffer_ &) =delete;
  ProcInfoBuffer_(ProcInfoBuffer_ &&) =delete;
  ProcInfoBuffer_& operator=(ProcInfoBuffer_ &&) =delete;
};

template<typename Fnct>
inline
void
iterate_proc_info_(const ProcInfoBuffer_ &info_buffer,
                   Fnct fnct)
{
  const auto *ptr=reinterpret_cast<const char *>(info_buffer.storage);
  const auto *limit=ptr+info_buffer.size;
  while(ptr<limit)
  {
    const auto &info=*reinterpret_cast<const ProcInfoBuffer_::info_t*>(ptr);
    if(!fnct(info))
    {
      break;
    }
    ptr+=info.Size;
  }
}

inline
std::vector<CpuId>
read_list_(const GROUP_AFFINITY &ga)
{
  auto result=std::vector<CpuId>{};
  const auto group=int(ga.Group)<<6;
  auto mask=ga.Mask;
  for(auto i=0; mask; ++i, mask>>=1)
  {
    if(mask&1)
    {
      result.emplace_back(CpuId{group+i});
    }
  }
  return result;
}

inline
bool
has_cpu_(const GROUP_AFFINITY &ga,
         CpuId cpu)
{
  const auto cpu_group=cpu.id>>6;
  const auto cpu_shift=cpu.id&63;
  return (cpu_group==int(ga.Group))&&((ga.Mask>>cpu_shift)&1);
}

inline
void
collect_next_level_(const ProcInfoBuffer_ &info_buffer,
                    TopologyGroup &grp)
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
      iterate_proc_info_(info_buffer,
        [&](const auto &info)
        {
          if((info.Relationship==RelationCache)&&
             ((info.Cache.Type==CacheUnified)||(info.Cache.Type==CacheData))&&
             has_cpu_(info.Cache.GroupMask, cpu))
          {
            auto cache_level=int(info.Cache.Level);
            auto cache_size=int(info.Cache.CacheSize);
            auto cache_line=int(info.Cache.LineSize);
            if(cache_level==grp.cache_level)
            {
              grp.cache_size=cache_size;
              grp.cache_line=cache_line;
            }
            else if(cache_level==grp.cache_level-1)
            {
              auto cpu_list=read_list_(info.Cache.GroupMask);
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
                collect_next_level_(info_buffer, child);
              }
            }
          }
          return true;
        });
    }
  }
}

inline
TopologyGroup
detect_()
{
  auto root=TopologyGroup{};
  const auto info_buffer=ProcInfoBuffer_{};
  auto max_cache_level=0;
  iterate_proc_info_(info_buffer,
    [&](const auto &info)
    {
      if((info.Relationship==RelationCache)&&
         ((info.Cache.Type==CacheUnified)||(info.Cache.Type==CacheData)))
      {
        max_cache_level=std::max(max_cache_level, int(info.Cache.Level));
      }
      return true;
    });
  iterate_proc_info_(info_buffer,
    [&](const auto &info)
    {
      if(info.Relationship==RelationNumaNode)
      {
        auto &child=root.children.emplace_back(TopologyGroup{});
        child.numa=NumaId{int(info.NumaNode.NodeNumber)};
        child.cpus=read_list_(info.NumaNode.GroupMask);
        root.cpus.insert(cend(root.cpus),
                         cbegin(child.cpus), cend(child.cpus));
        child.cache_level=max_cache_level;
        collect_next_level_(info_buffer, child);
      }
      return true;
    });
  return root;
}

} // namespace dim::cpu::impl_

#endif // DIM_CPU_DETECT_WINDOWS_HPP

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
