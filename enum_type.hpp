//----------------------------------------------------------------------------

#ifndef DIM_ENUM_TYPE_HPP
#define DIM_ENUM_TYPE_HPP

/**
inspired from
  https://www.reddit.com/r/cpp/comments/7pya5s/stdvisit_overhead/
  https://godbolt.org/#g:!((g:!((g:!((h:codeEditor,i:(j:1,lang:c%2B%2B,source:'%23include+%3Cvariant%3E%0A%0Atemplate+%3Ctypename+Function,+typename+Variant,+typename+T,+typename+...Ts%3E%0Ainline+auto+dvisit_impl(%0A++++const+Variant%26+v,%0A++++const+Function%26+f%0A)+%7B%0A++++if+(auto+x+%3D+std::get_if%3CT%3E(%26v))+%7B%0A++++++++return+f(*x)%3B%0A++++%7D+else+if+constexpr+(sizeof...(Ts)+%3E+0)+%7B%0A++++++++return+dvisit_impl%3CFunction,+Variant,+Ts...%3E(v,+f)%3B%0A++++%7D%0A%7D%0A%0Atemplate+%3Ctypename+Function,+typename+...Ts%3E%0Ainline+auto+dvisit(%0A++++const+std::variant%3CTs...%3E%26+v,%0A++++const+Function%26+f%0A)+%7B%0A++++if+(v.valueless_by_exception())%0A++++++++__builtin_unreachable()%3B%0A%0A++++return+dvisit_impl%3CFunction,+std::variant%3CTs...%3E,+Ts...%3E(v,+f)%3B%0A%7D%0A%0Avoid+*getPtr(const+std::variant%3Cchar*,+unsigned+char*%3E%26+v)+%7B%0A++++return+dvisit(v,+%5B%26%5D(auto+x)+-%3E+void*+%7B+%0A++++++++return+x%3B+%0A++++%7D)%3B%0A%7D%0A'),l:'5',n:'0',o:'C%2B%2B+source+%231',t:'0')),k:53.11284046692607,l:'4',n:'0',o:'',s:0,t:'0'),(g:!((h:compiler,i:(compiler:gsnapshot,filters:(b:'0',binary:'1',commentOnly:'0',demangle:'0',directives:'0',execute:'1',intel:'0',trim:'0'),lang:c%2B%2B,libs:!(),options:'-std%3Dc%2B%2B17+-O3',source:1),l:'5',n:'0',o:'x86-64+gcc+(trunk)+(Editor+%231,+Compiler+%231)+C%2B%2B',t:'0')),k:46.88715953307393,l:'4',n:'0',o:'',s:0,t:'0')),l:'2',n:'0',o:'',t:'0')),version:4
**/

#include <type_traits>
#include <utility>
#include <typeinfo>

namespace dim {

template<typename ...Types>
class EnumType
{
public:

  template<typename T>
  using prevent_enum_type =
    typename std::enable_if_t<!std::is_convertible_v<T, EnumType>>;

  EnumType()
  : rank_{0}
  , storage_{}
  {
    // nothing more to be done
  }

  template<typename T,
           typename = prevent_enum_type<T>>
  explicit
  EnumType(const T &rhs)
  : EnumType{}
  {
    set<T>(rhs);
  }

  template<typename T,
           typename = prevent_enum_type<T>>
  EnumType &
  operator=(const T &rhs)
  {
    if(reinterpret_cast<const void *>(&rhs)!=
       reinterpret_cast<const void *>(&storage_))
    {
      set<T>(rhs);
    }
    return *this;
  }

  template<typename T,
           typename = prevent_enum_type<T>>
  explicit
  EnumType(T &&rhs) noexcept
  : EnumType{}
  {
    set<T>(std::move(rhs));
  }

  template<typename T,
           typename = prevent_enum_type<T>>
  EnumType &
  operator=(T &&rhs) noexcept
  {
    if(reinterpret_cast<const void *>(&rhs)!=
       reinterpret_cast<const void *>(&storage_))
    {
      set<T>(std::move(rhs));
    }
    return *this;
  }

  EnumType(const EnumType &rhs)
  : EnumType{}
  {
    copy_init_<Types...>(rank_, rhs);
    rank_=rhs.rank_;
  }

  EnumType &
  operator=(const EnumType &rhs)
  {
    if(&rhs!=this)
    {
      destroy_<Types...>(rank_);
      rank_=0;
      copy_init_<Types...>(rank_, rhs);
      rank_=rhs.rank_;
    }
    return *this;
  }

  EnumType(EnumType &&rhs) noexcept
  : EnumType{}
  {
    move_init_<Types...>(rank_, std::move(rhs));
    rank_=rhs.rank_;
    rhs.rank_=0;
  }

  EnumType &
  operator=(EnumType &&rhs) noexcept
  {
    if(&rhs!=this)
    {
      destroy_<Types...>(rank_);
      rank_=0;
      move_init_<Types...>(rank_, std::move(rhs));
      rank_=rhs.rank_;
      rhs.rank_=0;
    }
    return *this;
  }

  ~EnumType()
  {
    destroy_<Types...>(rank_);
  }

  template<typename T>
  bool
  is() const
  {
    return rank_==TypeRank_v<T, void, Types...>;
  }

  template<typename T>
  T &
  get()
  {
    if(rank_==TypeRank_v<T, void, Types...>)
    {
      return reinterpret_cast<T &>(storage_);
    }
    throw std::bad_cast{};
  }

  template<typename T>
  const T &
  get() const
  {
    if(rank_==TypeRank_v<T, void, Types...>)
    {
      return reinterpret_cast<const T &>(storage_);
    }
    throw std::bad_cast{};
  }

  template<typename T,
           typename ...Args>
  void
  set(Args &&...args)
  {
    destroy_<Types...>(rank_);
    rank_=0;
    new (&storage_) T(std::forward<Args>(args)...);
    rank_=TypeRank_v<T, void, Types...>;
  }

  void
  clear()
  {
    destroy_<Types...>(rank_);
    rank_=0;
  }

  template<typename ...Functions>
  decltype(auto)
  use(Functions &&...fncts)
  {
    if(rank_)
    {
      struct Dispatcher : Functions... { using Functions::operator()...; };
      return use_<Dispatcher, Types...>
             (rank_, Dispatcher{std::forward<Functions>(fncts)...});
    }
    throw std::bad_cast{};
  }

  template<typename ...Functions>
  decltype(auto)
  use(Functions &&...fncts) const
  {
    if(rank_)
    {
      struct Dispatcher : Functions... { using Functions::operator()...; };
      return use_<Dispatcher, Types...>
             (rank_, Dispatcher{std::forward<Functions>(fncts)...});
    }
    throw std::bad_cast{};
  }

private:

  using Rank = unsigned char;

  static_assert(sizeof...(Types)>0,
    "at least one type is required for EnumType<>");
  static_assert(sizeof...(Types)>>(8*sizeof(Rank))==0,
    "too many types for EnumType<>");

  template<typename ...>
  struct TypeRank;

  template<typename ...Tail>
  static constexpr Rank TypeRank_v=TypeRank<Tail...>::value;

  template<typename T,
           typename ...Tail>
  struct TypeRank<T, T, Tail...>
    : std::integral_constant<Rank, 0> {};

  template<typename T,
           typename Head,
           typename ...Tail>
  struct TypeRank<T, Head, Tail...>
    : std::integral_constant<Rank, 1+TypeRank_v<T, Tail...>> {};

  template<typename Head,
           typename ...Tail>
  void
  copy_init_(Rank rank,
             const EnumType &rhs)
  {
    if(rank==1)
    {
      new (&storage_) Head{reinterpret_cast<const Head &>(rhs.storage_)};
      return;
    }
    if constexpr(sizeof...(Tail)>0)
    {
      copy_init_<Tail...>(--rank, rhs);
    }
  }

  template<typename Head,
           typename ...Tail>
  void
  move_init_(Rank rank,
             EnumType &&rhs)
  {
    if(rank==1)
    {
      new (&storage_) Head{std::move(reinterpret_cast<Head &>(rhs.storage_))};
      return;
    }
    if constexpr(sizeof...(Tail)>0)
    {
      move_init_<Tail...>(--rank, std::move(rhs));
    }
  }

  template<typename Head,
           typename ...Tail>
  void
  destroy_(Rank rank)
  {
    if(rank==1)
    {
      reinterpret_cast<Head &>(storage_).~Head();
      return;
    }
    if constexpr(sizeof...(Tail)>0)
    {
      destroy_<Tail...>(--rank);
    }
  }

  template<typename Function,
           typename Head,
           typename ...Tail>
  decltype(auto)
  use_(Rank rank,
       Function &&fnct)
  {
    if(rank>1)
    {
      if constexpr(sizeof...(Tail)>0)
      {
        return use_<Function, Tail...>(--rank, std::forward<Function>(fnct));
      }
    }
    return fnct(reinterpret_cast<Head &>(storage_));
  }

  template<typename Function,
           typename Head,
           typename ...Tail>
  decltype(auto)
  use_(Rank rank,
       Function &&fnct) const
  {
    if(rank>1)
    {
      if constexpr(sizeof...(Tail)>0)
      {
        return use_<Function, Tail...>(--rank, std::forward<Function>(fnct));
      }
    }
    return fnct(reinterpret_cast<const Head &>(storage_));
  }

  Rank rank_;
  std::aligned_union_t<0, Types...> storage_;
};

} // namespace dim

#endif // DIM_ENUM_TYPE_HPP

//----------------------------------------------------------------------------
