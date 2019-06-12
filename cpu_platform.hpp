//----------------------------------------------------------------------------

#ifndef DIM_CPU_PLATFORM_HPP
#define DIM_CPU_PLATFORM_HPP 1

#include "cpu_detect.hpp"

#include <utility>
#include <memory>

namespace dim::cpu {

class Platform
{
public:

  Platform()
  : root_{detect()}
  , max_cache_level_{}
  , max_cache_line_{}
  , numa_count_{}
  , cpu_count_{}
  , numas_{}
  , cpus_{}
  , numa_indices_{}
  , distances_{}
  , proximities_{}
  , roundtrips_{}
  {
    use_sys_cpu(root_.cpus);
  }

  int // max cache level or 0 if unknown
  max_cache_level() const
  {
    return max_cache_level_;
  }

  int // maximum cacheline size in bytes
  max_cache_line() const
  {
    return max_cache_line_;
  }

  int // number of numa nodes
  numa_count() const
  {
    return numa_count_;
  }

  NumaId // system id of numa node or -1 if unknown
  numa_id(int numa_index) const
  {
    return numas_[numa_index];
  }

  int // number of used cpu
  cpu_count() const
  {
    return cpu_count_;
  }

  CpuId // system id of cpu or -1 if unknown
  cpu_id(int cpu_index) const
  {
    return cpus_[cpu_index];
  }

  int // numa index of cpu
  numa(int cpu_index) const
  {
    return numa_indices_[cpu_index];
  }

  int // cache distance from one cpu to another
  distance(int cpu_index,
           int other_cpu_index) const
  {
    return distances_[cpu_index][other_cpu_index];
  }

  int // cache proximity from one cpu to another
  proximity(int cpu_index,
            int other_cpu_index) const
  {
    return proximities_[cpu_index][other_cpu_index];
  }

  const int * // cache-aware roundtrip starting from cpu
  roundtrip(int cpu_index) const
  {
    return roundtrips_[cpu_index].get();
  }

  const TopologyGroup &
  topology() const
  {
    return root_;
  }

  void
  use_sys_cpu(const std::vector<CpuId> &used_cpus)
  {
    // collect paths of used cpu and global properties
    auto cpu_paths=std::vector<TopologyGroup::Path>{};
    auto numas=std::vector<NumaId>{};
    for(const auto *used=&used_cpus;
        empty(cpu_paths); // if no cpu is actually used
        used=&root_.cpus) // then use all of them
    {
      max_cache_level_=0;
      max_cache_line_=0;
      numas.clear();
      visit(root_,
        [&](const auto &grp, const auto &path)
        {
          if(is_cpu(grp)&&(find_(*used, grp.cpus.front())!=-1))
          {
            for(const auto *pgrp: path)
            {
              max_cache_level_=std::max(max_cache_level_, pgrp->cache_level);
              max_cache_line_=std::max(max_cache_line_, pgrp->cache_line);
              const auto numa=pgrp->numa;
              if(valid(numa)&&(find_(numas, numa)==-1))
              {
                numas.emplace_back(numa);
              }
            }
            cpu_paths.emplace_back(path);
          }
          return true;
        });
    }
    if(!max_cache_line_)
    {
      max_cache_line_=64; // suitable assumed default for current hardware
    }
    if(empty(numas))
    {
      numas.emplace_back(NumaId{}); // at least one unknown node
    }
    numa_count_=int(size(numas));
    numas_=std::make_unique<NumaId[]>(numa_count_);
    copy(cbegin(numas), cend(numas), numas_.get());
    cpu_count_=int(size(cpu_paths));
    // collect specific properties of each cpu
    cpus_=std::make_unique<CpuId[]>(cpu_count_);
    numa_indices_=std::make_unique<int[]>(cpu_count_);
    for(auto cpu=0; cpu<cpu_count_; ++cpu)
    {
      const auto &path=cpu_paths[cpu];
      cpus_[cpu]=path.back()->cpus.front();
      numa_indices_[cpu]=-1;
      for(const auto *pgrp: path)
      {
        const auto numa=pgrp->numa;
        for(auto node=0; node<numa_count_; ++node)
        {
          if(numa==numas_[node])
          {
            numa_indices_[cpu]=node;
          }
        }
      }
    }
    // compute distance, proximity and roudtrip
    auto max_distance=0;
    distances_=std::make_unique<std::unique_ptr<int[]>[]>(cpu_count_);
    for(auto cpu=0; cpu<cpu_count_; ++cpu)
    {
      const auto &path=cpu_paths[cpu];
      distances_[cpu]=std::make_unique<int[]>(cpu_count_);
      for(auto other=0; other<cpu_count_; ++other)
      {
        const auto &opath=cpu_paths[other];
        const auto common=mismatch(cbegin(path), cend(path), cbegin(opath));
        const auto first_len=int(std::distance(common.first, cend(path)));
        const auto second_len=int(std::distance(common.second, cend(opath)));
        const auto numa_penalty=
          2*int(numa_indices_[cpu]!=numa_indices_[other]);
        const auto distance=(first_len+second_len+numa_penalty+1)/2;
        max_distance=std::max(max_distance, distance);
        distances_[cpu][other]=distance;
      }
    }
    proximities_=std::make_unique<std::unique_ptr<int[]>[]>(cpu_count_);
    for(auto cpu=0; cpu<cpu_count_; ++cpu)
    {
      proximities_[cpu]=std::make_unique<int[]>(cpu_count_);
      for(auto other=0; other<cpu_count_; ++other)
      {
        proximities_[cpu][other]=(2<<max_distance)-
                                 (1<<distances_[cpu][other]);
      }
    }
    roundtrips_=std::make_unique<std::unique_ptr<int[]>[]>(cpu_count_);
    for(auto cpu=0; cpu<cpu_count_; ++cpu)
    {
      const auto &path=cpu_paths[cpu];
      roundtrips_[cpu]=std::make_unique<int[]>(cpu_count_);
      auto trip_count=0;
      roundtrips_[cpu][trip_count++]=cpu;
      const auto *pgrp=path.back();
      for_each(crbegin(path)+1, crend(path),
        [&](const auto *parent)
        {
          for(auto delta=0; delta<cpu_count_; ++delta)
          {
            const auto other=(delta+cpu)%cpu_count_;
            const auto &opath=cpu_paths[other];
            if((find(cbegin(opath), cend(opath), parent)!=cend(opath))&&
               (find(cbegin(opath), cend(opath), pgrp)==cend(opath)))
            {
              roundtrips_[cpu][trip_count++]=other;
            }
          }
          pgrp=parent;
        });
    }
  }

  void
  filter_sys_cpu(const std::vector<CpuId> &cpus,
                 bool exclude=false)
  {
    auto used_cpus=std::vector<CpuId>{};
    for(auto cpu=0; cpu<cpu_count_; ++cpu)
    {
      const auto sys_id=cpus_[cpu];
      const auto found=(find_(cpus, sys_id)!=-1);
      if(found!=exclude)
      {
        used_cpus.emplace_back(sys_id);
      }
    }
    use_sys_cpu(used_cpus);
  }

private:

  template<typename T>
  static
  int
  find_(const T *values,
        int count,
        const T &value)
  {
    const auto end=values+count;
    const auto it=std::find(values, end, value);
    return (it==end) ? -1 : int(std::distance(values, it));
  }

  template<typename T>
  static
  int
  find_(const std::vector<T> &values,
        const T &value)
  {
    return find_(data(values), int(size(values)), value);
  }

  TopologyGroup root_;
  int max_cache_level_;
  int max_cache_line_;
  int numa_count_;
  int cpu_count_;
  std::unique_ptr<NumaId[]> numas_;
  std::unique_ptr<CpuId[]> cpus_;
  std::unique_ptr<int[]> numa_indices_;
  std::unique_ptr<std::unique_ptr<int[]>[]> distances_;
  std::unique_ptr<std::unique_ptr<int[]>[]> proximities_;
  std::unique_ptr<std::unique_ptr<int[]>[]> roundtrips_;
};

inline
int // index of used cpu or -1 if not found
find_index_from_sys_id(const Platform &p,
                       CpuId cpu)
{
  for(auto index=0; index<p.cpu_count(); ++index)
  {
    if(p.cpu_id(index)==cpu)
    {
      return index;
    }
  }
  return -1;
}

inline
int // index of used numa node or -1 if not found
find_index_from_sys_id(const Platform &p,
                       NumaId numa)
{
  for(auto index=0; index<p.numa_count(); ++index)
  {
    if(p.numa_id(index)==numa)
    {
      return index;
    }
  }
  return -1;
}

inline
void
disable_smt(Platform &p)
{
  p.filter_sys_cpu(collect_indexth_cpu_of_cache_level(p.topology(), 0, 1));
}

inline
int // cache size divided by number of cpus in shared cache
compute_partial_cache_size(const Platform &p,
                           int cpu_index,
                           int level)
{
  const auto *found=
    find_cache(p.topology(), p.cpu_id(cpu_index), level);
  if(!found)
  {
    return -1;
  }
  auto count=0;
  for(const auto &id: found->cpus)
  {
    if(find_index_from_sys_id(p, id)!=-1)
    {
      ++count;
    }
  }
  return found->cache_size/count;
}

inline
std::string
to_string(const Platform &p)
{
  auto txt=std::string{};
  const auto show_values=
    [&](const auto &title, const auto &count, const auto &fnct)
    {
      txt+=title;
      txt+=":";
      for(auto i=0; i<count; ++i)
      {
        txt+=' ';
        txt+=std::to_string(fnct(i));
      }
      txt+='\n';
    };
  const auto show_values_2=
    [&](const auto &title, const auto &low, const auto &high,
        const auto &count, const auto &fnct)
    {
      for(auto n=low; n<high; ++n)
      {
        txt+=title;
        txt+="[";
        txt+=std::to_string(n);
        txt+="]:";
        for(auto i=0; i<count; ++i)
        {
          txt+=' ';
          txt+=std::to_string(fnct(n, i));
        }
        txt+='\n';
      }
    };
  txt+="max_cache_level: "+std::to_string(p.max_cache_level())+'\n';
  txt+="max_cache_line: "+std::to_string(p.max_cache_line())+'\n';
  txt+="numa_count: "+std::to_string(p.numa_count())+'\n';
  txt+="cpu_count: "+std::to_string(p.cpu_count())+'\n';
  show_values("numa_sys_ids", p.numa_count(),
    [&](const auto &numa)
    {
      return p.numa_id(numa).id;
    });
  show_values("cpu_sys_ids", p.cpu_count(),
    [&](const auto &cpu)
    {
      return p.cpu_id(cpu).id;
    });
  show_values("numas", p.cpu_count(),
    [&](const auto &cpu)
    {
      return p.numa(cpu);
    });
  show_values_2("distances", 0, p.cpu_count(), p.cpu_count(),
    [&](const auto &cpu, const auto &other)
    {
      return p.distance(cpu, other);
    });
  show_values_2("proximities", 0, p.cpu_count(), p.cpu_count(),
    [&](const auto &cpu, const auto &other)
    {
      return p.proximity(cpu, other);
    });
  show_values_2("roundtrips", 0, p.cpu_count(), p.cpu_count(),
    [&](const auto &cpu, const auto &other)
    {
      return p.roundtrip(cpu)[other];
    });
  return txt;
}

inline
std::ostream &
operator<<(std::ostream &output,
           const Platform &p)
{
  return output << to_string(p);
}

} // namespace dim::cpu

#endif // DIM_CPU_PLATFORM_HPP

//----------------------------------------------------------------------------
