//----------------------------------------------------------------------------

#ifndef DIM_ENUMTYPE_HPP
#define DIM_ENUMTYPE_HPP 1

/**
inspired from
  https://gist.github.com/S6066/f726a37b2b703efea7ee27103e5bec89

see also
  https://www.reddit.com/r/cpp/comments/7pya5s/stdvisit_overhead/

https://godbolt.org/#g:!((g:!((g:!((h:codeEditor,i:(j:1,lang:c%2B%2B,source:'%23include+%3Cvariant%3E%0A%0Atemplate+%3Ctypename+Function,+typename+Variant,+typename+T,+typename+...Ts%3E%0Ainline+auto+dvisit_impl(%0A++++const+Variant%26+v,%0A++++const+Function%26+f%0A)+%7B%0A++++if+(auto+x+%3D+std::get_if%3CT%3E(%26v))+%7B%0A++++++++return+f(*x)%3B%0A++++%7D+else+if+constexpr+(sizeof...(Ts)+%3E+0)+%7B%0A++++++++return+dvisit_impl%3CFunction,+Variant,+Ts...%3E(v,+f)%3B%0A++++%7D%0A%7D%0A%0Atemplate+%3Ctypename+Function,+typename+...Ts%3E%0Ainline+auto+dvisit(%0A++++const+std::variant%3CTs...%3E%26+v,%0A++++const+Function%26+f%0A)+%7B%0A++++if+(v.valueless_by_exception())%0A++++++++__builtin_unreachable()%3B%0A%0A++++return+dvisit_impl%3CFunction,+std::variant%3CTs...%3E,+Ts...%3E(v,+f)%3B%0A%7D%0A%0Avoid+*getPtr(const+std::variant%3Cchar*,+unsigned+char*%3E%26+v)+%7B%0A++++return+dvisit(v,+%5B%26%5D(auto+x)+-%3E+void*+%7B+%0A++++++++return+x%3B+%0A++++%7D)%3B%0A%7D%0A'),l:'5',n:'0',o:'C%2B%2B+source+%231',t:'0')),k:53.11284046692607,l:'4',n:'0',o:'',s:0,t:'0'),(g:!((h:compiler,i:(compiler:gsnapshot,filters:(b:'0',binary:'1',commentOnly:'0',demangle:'0',directives:'0',execute:'1',intel:'0',trim:'0'),lang:c%2B%2B,libs:!(),options:'-std%3Dc%2B%2B17+-O3',source:1),l:'5',n:'0',o:'x86-64+gcc+(trunk)+(Editor+%231,+Compiler+%231)+C%2B%2B',t:'0')),k:46.88715953307393,l:'4',n:'0',o:'',s:0,t:'0')),l:'2',n:'0',o:'',t:'0')),version:4
**/

#include <type_traits>

namespace dim {

template<typename ...Types>
class EnumType
{
public:

  EnumType();

  template<typename T,
           typename = std::enable_if_t<!std::is_convertible_v<T, EnumType>>>
  explicit
  EnumType(const T &rhs);

  template<typename T,
           typename = std::enable_if_t<!std::is_convertible_v<T, EnumType>>>
  EnumType &
  operator=(const T &rhs);

  template<typename T,
           typename = std::enable_if_t<!std::is_convertible_v<T, EnumType>>>
  explicit
  EnumType(T &&rhs) noexcept;

  template<typename T,
           typename = std::enable_if_t<!std::is_convertible_v<T, EnumType>>>
  EnumType &
  operator=(T &&rhs) noexcept;

  EnumType(const EnumType &rhs);

  EnumType &
  operator=(const EnumType &rhs);
  
  EnumType(EnumType &&rhs) noexcept;

  EnumType &
  operator=(EnumType &&rhs) noexcept;

  ~EnumType();

  template<typename T>
  bool
  is() const;

  template<typename T>
  T &
  get();

  template<typename T>
  const T &
  get() const;

  template<typename T,
           typename ...Args>
  void
  set(Args &&...args);

  void
  clear();

  template<typename ...Functions>
  decltype(auto)
  use(Functions &&...fncts);

  template<typename ...Functions>
  decltype(auto)
  use(Functions &&...fncts) const;

private:

  using Rank = unsigned char;

  static_assert(sizeof...(Types)>0,
    "at least one type is required for EnumType<>");
  static_assert(sizeof...(Types)>>(8*sizeof(Rank))==0,
    "too many types for EnumType<>");

  template<typename ...>
  struct TypeRank;

  template<typename ...Tail>
  static constexpr unsigned char TypeRank_v=TypeRank<Tail...>::value;

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
  copyInit_(Rank rank,
            const EnumType &rhs);

  template<typename Head,
           typename ...Tail>
  void
  moveInit_(Rank rank,
            EnumType &&rhs);

  template<typename Head,
           typename ...Tail>
  void
  destroy_(Rank rank);

  template<typename Function,
           typename Head,
           typename ...Tail>
  decltype(auto)
  use_(Rank rank,
       Function &&fnct);

  template<typename Function,
           typename Head,
           typename ...Tail>
  decltype(auto)
  use_(Rank rank,
       Function &&fnct) const;

  Rank rank_;
  std::aligned_union_t<0, Types...> storage_;
};

} // namespace dim

#endif // DIM_ENUMTYPE_HPP

//----------------------------------------------------------------------------
// implementation details (don't look below!)
//----------------------------------------------------------------------------

#ifndef DIM_ENUMTYPE_HPP_IMPL
#define DIM_ENUMTYPE_HPP_IMPL 1

#include <utility>
#include <typeinfo>

namespace dim {

template<typename ...Types>
inline
EnumType<Types...>::EnumType()
: rank_{0}
, storage_{}
{
  // nothing more to be done
}

template<typename ...Types>
template<typename T,
         typename>
inline
EnumType<Types...>::EnumType(const T &rhs)
: EnumType{}
{
  set<T>(rhs);
}

template<typename ...Types>
template<typename T,
         typename>
inline
EnumType<Types...> &
EnumType<Types...>::operator=(const T &rhs)
{
  if(reinterpret_cast<const void *>(&rhs)!=
     reinterpret_cast<const void *>(&storage_))
  {
    set<T>(rhs);
  }
  return *this;
}

template<typename ...Types>
template<typename T,
         typename>
inline
EnumType<Types...>::EnumType(T &&rhs) noexcept
: EnumType{}
{
  set<T>(std::move(rhs));
}

template<typename ...Types>
template<typename T,
         typename>
inline
EnumType<Types...> &
EnumType<Types...>::operator=(T &&rhs) noexcept
{
  if(reinterpret_cast<const void *>(&rhs)!=
     reinterpret_cast<const void *>(&storage_))
  {
    set<T>(std::move(rhs));
  }
  return *this;
}

template<typename ...Types>
inline
EnumType<Types...>::EnumType(const EnumType &rhs)
: rank_{rhs.rank_}
, storage_{}
{
  copyInit_<Types...>(rank_, rhs);
}

template<typename ...Types>
inline
EnumType<Types...> &
EnumType<Types...>::operator=(const EnumType &rhs)
{
  if(&rhs!=this)
  {
    if(rank_)
    {
      destroy_<Types...>(rank_);
    }
    rank_=rhs.rank_;
    copyInit_<Types...>(rank_, rhs);
  }
  return *this;
}

template<typename ...Types>
inline
EnumType<Types...>::EnumType(EnumType &&rhs) noexcept
: rank_{rhs.rank_}
, storage_{}
{
  moveInit_<Types...>(rank_, std::move(rhs));
  rhs.rank_=0;
}

template<typename ...Types>
inline
EnumType<Types...> &
EnumType<Types...>::operator=(EnumType &&rhs) noexcept
{
  if(&rhs!=this)
  {
    if(rank_)
    {
      destroy_<Types...>(rank_);
    }
    rank_=rhs.rank_;
    moveInit_<Types...>(rank_, std::move(rhs));
    rhs.rank_=0;
  }
  return *this;
}

template<typename ...Types>
inline
EnumType<Types...>::~EnumType()
{
  if(rank_)
  {
    destroy_<Types...>(rank_);
  }
}

template<typename ...Types>
template<typename T>
inline
bool
EnumType<Types...>::is() const
{
  return rank_==TypeRank_v<T, void, Types...>;
}

template<typename ...Types>
template<typename T>
inline
T &
EnumType<Types...>::get()
{
  if(rank_==TypeRank_v<T, void, Types...>)
  {
    return reinterpret_cast<T &>(storage_);
  }
  else
  {
    throw std::bad_cast{};
  }
}

template<typename ...Types>
template<typename T>
inline
const T &
EnumType<Types...>::get() const
{
  if(rank_==TypeRank_v<T, void, Types...>)
  {
    return reinterpret_cast<const T &>(storage_);
  }
  else
  {
    throw std::bad_cast{};
  }
}

template<typename ...Types>
template<typename T,
         typename ...Args>
inline
void
EnumType<Types...>::set(Args &&...args)
{
  if(rank_)
  {
    destroy_<Types...>(rank_);
  }
  rank_=TypeRank_v<T, void, Types...>;
  new (&storage_) T(std::forward<Args>(args)...);
}

template<typename ...Types>
inline
void
EnumType<Types...>::clear()
{
  destroy_<Types...>(rank_);
  rank_=0;
}

template<typename ...Types>
template<typename ...Functions>
inline
decltype(auto)
EnumType<Types...>::use(Functions &&...fncts)
{
  struct Dispatcher : Functions... { using Functions::operator()...; };
  return use_<Dispatcher, Types...>
         (rank_, Dispatcher{std::forward<Functions>(fncts)...});
}

template<typename ...Types>
template<typename ...Functions>
inline
decltype(auto)
EnumType<Types...>::use(Functions &&...fncts) const
{
  struct Dispatcher : Functions... { using Functions::operator()...; };
  return use_<Dispatcher, Types...>
         (rank_, Dispatcher{std::forward<Functions>(fncts)...});
}

template<typename ...Types>
template<typename Head,
         typename ...Tail>
inline
void
EnumType<Types...>::copyInit_(Rank rank,
                              const EnumType &rhs)
{
  if(rank==1)
  {
    new (&storage_) Head{reinterpret_cast<const Head &>(rhs.storage_)};
    return;
  }
  if constexpr(sizeof...(Tail)>0)
  {
    copyInit_<Tail...>(--rank, rhs);
  }
}

template<typename ...Types>
template<typename Head,
         typename ...Tail>
inline
void
EnumType<Types...>::moveInit_(Rank rank,
                              EnumType &&rhs)
{
  if(rank==1)
  {
    new (&storage_) Head{std::move(reinterpret_cast<Head &>(rhs.storage_))};
    return;
  }
  if constexpr(sizeof...(Tail)>0)
  {
    moveInit_<Tail...>(--rank, std::move(rhs));
  }
}

template<typename ...Types>
template<typename Head,
         typename ...Tail>
inline
void
EnumType<Types...>::destroy_(Rank rank)
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

template<typename ...Types>
template<typename Function,
         typename Head,
         typename ...Tail>
inline
decltype(auto)
EnumType<Types...>::use_(Rank rank,
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

template<typename ...Types>
template<typename Function,
         typename Head,
         typename ...Tail>
inline
decltype(auto)
EnumType<Types...>::use_(Rank rank,
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

} // namespace dim

#endif // DIM_ENUMTYPE_HPP_IMPL

//----------------------------------------------------------------------------
