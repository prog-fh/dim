//----------------------------------------------------------------------------

#ifndef DIM_CPU_PLATFORM_HPP
#define DIM_CPU_PLATFORM_HPP 1

#include "cpu_detect.hpp"

#include <utility>
#include <memory>

namespace dim {

class CpuPlatform
{
public:

  CpuPlatform()
  : root_{}
  , cpu_paths_{}
  , numa_count_{}
  , cpu_count_{}
  , max_cache_level_{}
  , max_cache_line_{}
  , numa_sys_id_{}
  , cpu_sys_id_{}
  , numa_node_{}
  , cache_size_{}
  , cache_line_{}
  , distance_{}
  , proximity_{}
  , roundtrip_{}
  {
    auto [root, online]=cpu_detect::detect();
    root_=std::make_unique<cpu_detect::CpuGroup>(std::move(root));
    max_cache_level_=0;
    max_cache_line_=0;
    auto numa_nodes=std::vector<int>{};
    cpu_paths_.clear();
    root_->visit([&](const auto &grp, const auto &path)
    {
      max_cache_level_=std::max(max_cache_level_, grp.cache_level);
      max_cache_line_=std::max(max_cache_line_, grp.cache_line);
      if(grp.numa_id!=-1)
      {
        numa_nodes.emplace_back(grp.numa_id);
      }
      if((size(grp.cpu_id)==1)&&empty(grp.children))
      {
        auto p=path;
        p.emplace_back(&grp);
        cpu_paths_.emplace_back(std::move(p));
      }
      return true;
    });
    if(!max_cache_line_)
    {
      max_cache_line_=64; // suitable assumed default for current hardware
    }
    if(empty(numa_nodes))
    {
      numa_nodes.emplace_back(-1); // at least one unknown node
    }
    numa_count_=int(size(numa_nodes));
    numa_sys_id_=std::make_unique<int[]>(numa_count_);
    std::copy(cbegin(numa_nodes), cend(numa_nodes), numa_sys_id_.get());
    cpu_count_=int(size(cpu_paths_));
    cpu_sys_id_=std::make_unique<int[]>(cpu_count_);
    numa_node_=std::make_unique<int[]>(cpu_count_);
    cache_size_=std::make_unique<std::unique_ptr<int[]>[]>(max_cache_level_);
    cache_line_=std::make_unique<std::unique_ptr<int[]>[]>(max_cache_level_);
    for(auto lvl=0; lvl<max_cache_level_; ++lvl)
    {
      cache_size_[lvl]=std::make_unique<int[]>(cpu_count_);
      cache_line_[lvl]=std::make_unique<int[]>(cpu_count_);
    }
    for(auto cpu=0; cpu<cpu_count_; ++cpu)
    {
      const auto &path=cpu_paths_[cpu];
      cpu_sys_id_[cpu]=path.back()->cpu_id[0];
      numa_node_[cpu]=-1;
      for(auto lvl=0; lvl<max_cache_level_; ++lvl)
      {
        cache_size_[lvl][cpu]=0;
        cache_line_[lvl][cpu]=0;
      }
      for(const auto *grp: path)
      {
        const auto numa_id=grp->numa_id;
        for(auto node=0; node<numa_count_; ++node)
        {
          if(numa_id==numa_sys_id_[node])
          {
            numa_node_[cpu]=node;
          }
        }
        const auto cache_level=grp->cache_level;
        if(cache_level>0)
        {
          cache_size_[cache_level-1][cpu]=grp->cache_size;
          cache_line_[cache_level-1][cpu]=grp->cache_line;
        }
      }
    }
    auto max_distance=0;
    distance_=std::make_unique<std::unique_ptr<int[]>[]>(cpu_count_);
    for(auto cpu=0; cpu<cpu_count_; ++cpu)
    {
      const auto &path=cpu_paths_[cpu];
      distance_[cpu]=std::make_unique<int[]>(cpu_count_);
      for(auto other=0; other<cpu_count_; ++other)
      {
        const auto &opath=cpu_paths_[other];
        const auto common=
          std::mismatch(cbegin(path), cend(path), cbegin(opath));
        const auto first_len=int(std::distance(common.first, cend(path)));
        const auto second_len=int(std::distance(common.second, cend(opath)));
        const auto numa_penalty=2*int(numa_node_[cpu]!=numa_node_[other]);
        const auto distance=(first_len+second_len+numa_penalty+1)/2;
        max_distance=std::max(max_distance, distance);
        distance_[cpu][other]=distance;
      }
    }
    proximity_=std::make_unique<std::unique_ptr<int[]>[]>(cpu_count_);
    for(auto cpu=0; cpu<cpu_count_; ++cpu)
    {
      proximity_[cpu]=std::make_unique<int[]>(cpu_count_);
      for(auto other=0; other<cpu_count_; ++other)
      {
        proximity_[cpu][other]=(2<<max_distance)-(1<<distance_[cpu][other]);
      }
    }
    roundtrip_=std::make_unique<std::unique_ptr<int[]>[]>(cpu_count_);
    for(auto cpu=0; cpu<cpu_count_; ++cpu)
    {
      const auto &path=cpu_paths_[cpu];
      roundtrip_[cpu]=std::make_unique<int[]>(cpu_count_);
      auto trip_count=0;
      roundtrip_[cpu][trip_count++]=cpu;
      const auto *grp=path.back();
      std::for_each(crbegin(path)+1, crend(path),
        [&](const auto *parent)
        {
          for(auto delta=0; delta<cpu_count_; ++delta)
          {
            const auto other=(delta+cpu)%cpu_count_;
            const auto &opath=cpu_paths_[other];
            if((std::find(cbegin(opath), cend(opath), parent)!=cend(opath))&&
               (std::find(cbegin(opath), cend(opath), grp)==cend(opath)))
            {
              roundtrip_[cpu][trip_count++]=other;
            }
          }
          grp=parent;
        });
    }
    filter_sys_cpu(online);
  }

  int // number of cpu
  cpu_count() const
  {
    return cpu_count_;
  }

  int // system id of cpu or -1 if unknown
  cpu_sys_id(int cpu) const
  {
    return cpu_sys_id_[cpu];
  }

  int // cache size in bytes of cpu or 0 if unavailable
  cache_size(int cpu,
             int level) const
  {
    return ((level>0)&&(level<=max_cache_level_))
           ? cache_size_[level-1][cpu] : 0;
  }

  int // cacheline size in bytes of cpu or 0 if unavailable
  cache_line(int cpu,
             int level) const
  {
    return ((level>0)&&(level<=max_cache_level_))
           ? cache_line_[level-1][cpu] : 0;
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

  int // numa node of cpu
  numa_node(int cpu) const
  {
    return numa_node_[cpu];
  }

  int // number of numa nodes
  numa_count() const
  {
    return numa_count_;
  }

  int // system id of numa node or -1 if unknown
  numa_sys_id(int numa_node) const
  {
    return numa_sys_id_[numa_node];
  }

  int // cache distance from one cpu to another
  distance(int cpu,
           int other_cpu) const
  {
    return distance_[cpu][other_cpu];
  }

  int // cache proximity from one cpu to another
  proximity(int cpu,
            int other_cpu) const
  {
    return proximity_[cpu][other_cpu];
  }

  const int * // cache-aware roundtrip starting from cpu
  roundtrip(int cpu) const
  {
    return roundtrip_[cpu].get();
  }

  const cpu_detect::CpuGroup &
  topology() const
  {
    return *root_;
  }

  void
  filter_sys_cpu(const std::vector<int> kept_sys_cpu)
  {
    auto next_cpu_count=0;
    auto tr=std::vector<int>{};
    for(auto cpu=0; cpu<cpu_count_; ++cpu)
    {
      tr.emplace_back((find_(kept_sys_cpu, cpu_sys_id_[cpu])==-1)
                      ? -1 : next_cpu_count++);
    }
    if((next_cpu_count==0)||(next_cpu_count==cpu_count_))
    {
      return;
    }
    const auto translate=
      [&](auto &previous)
      {
        using elem_t = std::decay_t<decltype(previous[0])>;
        auto next=std::make_unique<elem_t[]>(next_cpu_count);
        for(auto cpu=0; cpu<cpu_count_; ++cpu)
        {
          if(auto next_cpu=tr[cpu]; next_cpu!=-1)
          {
            next[next_cpu]=std::move(previous[cpu]);
          }
        }
        previous=std::move(next);
      };
    translate(cpu_sys_id_);
    translate(numa_node_);
    for(auto lvl=0; lvl<max_cache_level_; ++lvl)
    {
      translate(cache_size_[lvl]);
      translate(cache_line_[lvl]);
    }
    translate(distance_);
    translate(proximity_);
    translate(roundtrip_);
    for(auto cpu=0; cpu<next_cpu_count; ++cpu)
    {
      translate(distance_[cpu]);
      translate(proximity_[cpu]);
      for(auto other=0, next=0; other<cpu_count_; ++other)
      {
        if(auto next_other=tr[roundtrip_[cpu][other]]; next_other!=-1)
        {
          roundtrip_[cpu][next++]=next_other;
        }
      }
    }
    cpu_count_=next_cpu_count;
  }

  void
  filter_out_sys_cpu(const std::vector<int> excluded_sys_cpu)
  {
    auto kept_sys_cpu=std::vector<int>{};
    for(auto cpu=0; cpu<cpu_count_; ++cpu)
    {
      const auto sys_id=cpu_sys_id_[cpu];
      if(find_(excluded_sys_cpu, sys_id)==-1)
      {
        kept_sys_cpu.emplace_back(sys_id);
      }
    }
    filter_sys_cpu(kept_sys_cpu);
  }

  std::vector<int>
  first_sys_cpu_of_cache_level(int level) const
  {
    auto result=std::vector<int>{};
    if((level>0)&&(level<=max_cache_level_))
    {
      for(const auto &path: cpu_paths_)
      {
        for(const auto *grp: path)
        {
          if(grp->cache_level==level)
          {
            if(&grp->first_leaf()==path.back())
            {
              result.emplace_back(path.back()->cpu_id.front());
            }
          }
        }
      }
    }
    return result;
  }

  std::vector<int>
  first_sys_cpu_of_sys_numa(int sys_numa) const
  {
    auto result=std::vector<int>{};
    for(const auto &path: cpu_paths_)
    {
      for(const auto *grp: path)
      {
        if(grp->numa_id==sys_numa)
        {
          if(&grp->first_leaf()==path.back())
          {
            result.emplace_back(path.back()->cpu_id.front());
          }
        }
      }
    }
    return result;
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

  // address of root group is taken in cpu_paths_, thus in case of
  // move-copy/assign this address must not change, thats why this
  // root group is stored in a unique_ptr
  std::unique_ptr<cpu_detect::CpuGroup> root_;
  std::vector<std::vector<const cpu_detect::CpuGroup *>> cpu_paths_;
  int numa_count_;
  int cpu_count_;
  int max_cache_level_;
  int max_cache_line_;
  std::unique_ptr<int[]> numa_sys_id_;
  std::unique_ptr<int[]> cpu_sys_id_;
  std::unique_ptr<int[]> numa_node_;
  std::unique_ptr<std::unique_ptr<int[]>[]> cache_size_;
  std::unique_ptr<std::unique_ptr<int[]>[]> cache_line_;
  std::unique_ptr<std::unique_ptr<int[]>[]> distance_;
  std::unique_ptr<std::unique_ptr<int[]>[]> proximity_;
  std::unique_ptr<std::unique_ptr<int[]>[]> roundtrip_;
};

std::string
to_string(const CpuPlatform &p)
{
  auto txt=std::string{};
  auto show_values=
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
  auto show_values_2=
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
  show_values("numa_sys_id", p.numa_count(),
    [&](const auto &numa)
    {
      return p.numa_sys_id(numa);
    });
  show_values("cpu_sys_id", p.cpu_count(),
    [&](const auto &cpu)
    {
      return p.cpu_sys_id(cpu);
    });
  show_values("numa_node", p.cpu_count(),
    [&](const auto &cpu)
    {
      return p.numa_node(cpu);
    });
  txt+="max_cache_level: "+std::to_string(p.max_cache_level())+'\n';
  txt+="max_cache_line: "+std::to_string(p.max_cache_line())+'\n';
  show_values_2("cache_size", 1, p.max_cache_level()+1, p.cpu_count(),
    [&](const auto &lvl, const auto &cpu)
    {
      return p.cache_size(cpu, lvl);
    });
  show_values_2("cache_line", 1, p.max_cache_level()+1, p.cpu_count(),
    [&](const auto &lvl, const auto &cpu)
    {
      return p.cache_line(cpu, lvl);
    });
  show_values_2("distance", 0, p.cpu_count(), p.cpu_count(),
    [&](const auto &cpu, const auto &other)
    {
      return p.distance(cpu, other);
    });
  show_values_2("proximity", 0, p.cpu_count(), p.cpu_count(),
    [&](const auto &cpu, const auto &other)
    {
      return p.proximity(cpu, other);
    });
  show_values_2("roundtrip", 0, p.cpu_count(), p.cpu_count(),
    [&](const auto &cpu, const auto &other)
    {
      return p.roundtrip(cpu)[other];
    });
  return txt;
}

std::ostream &
operator<<(std::ostream &output,
           const CpuPlatform &p)
{
  return output << to_string(p);
}

} // namespace dim

#endif // DIM_CPU_PLATFORM_HPP

//----------------------------------------------------------------------------
