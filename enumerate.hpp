//----------------------------------------------------------------------------

#ifndef DIM_ENUMERATE_HPP
#define DIM_ENUMERATE_HPP 1

/**
inspired from
  http://reedbeta.com/blog/python-like-enumerate-in-cpp17/
**/

#include <type_traits>
#include <iterator>

namespace dim {

template<typename, typename = std::void_t<>>
struct has_begin : std::false_type { };
template<typename Container>
struct has_begin<Container,
  std::void_t<decltype(*begin(std::declval<Container>()))>>
  : std::true_type { };

template<typename, typename = std::void_t<>>
struct has_end : std::false_type { };
template<typename Container>
struct has_end<Container,
  std::void_t<decltype(*end(std::declval<Container>()))>>
  : std::true_type { };

template<typename, typename = std::void_t<>>
struct has_size : std::false_type { };
template<typename Container>
struct has_size<Container,
  std::void_t<decltype(size(std::declval<Container>()))>>
  : std::true_type { };

template<typename Container>
using can_enumerate = std::conjunction<
  has_begin<Container>,
  has_end<Container>,
  has_size<Container>>;

template<typename Container>
inline constexpr auto can_enumerate_v = can_enumerate<Container>::value;

template<typename Container,
         typename = std::enable_if_t<can_enumerate_v<Container>>>
auto // iterable 
enumerate(Container &&container);

template<typename Container,
         typename = std::enable_if_t<can_enumerate_v<Container>>>
auto // iterable
cenumerate(const Container &container);

template<typename Counter>
using can_count = std::conjunction<
  std::is_integral<Counter>,
  std::negation<std::is_same<bool, Counter>>>;

template<typename Counter>
inline constexpr auto can_count_v = can_count<Counter>::value;

template<typename Counter,
         typename = std::enable_if_t<can_count_v<Counter>>>
auto
enumerate(Counter from_count,
          Counter to_count);

template<typename Counter,
         typename = std::enable_if_t<can_count_v<Counter>>>
auto
enumerate(Counter to_count);

} // namespace dim

#endif // DIM_ENUMERATE_HPP

//----------------------------------------------------------------------------
// inline implementation details (don't look below!)
//----------------------------------------------------------------------------

#ifndef DIM_ENUMERATE_HPP_IMPL
#define DIM_ENUMERATE_HPP_IMPL 1

#include <utility>

namespace dim {

template<typename Container,
         typename = std::enable_if_t<can_enumerate_v<Container>>>
inline
auto
enumerate(Container &&container)
{
  using std::begin;
  using std::end;
  using std::size;
  using begin_t = decltype(begin(container));
  using end_t = decltype(end(container));
  using index_t = decltype(size(container));
  using value_t = decltype(*begin(container));
  struct Value
  {
    index_t index;
    value_t value;
  };
  struct Iter
  {
    index_t count;
    begin_t it;
    bool operator!=(const end_t &rhs) const { return it!=rhs; }
    void operator++() { ++count; ++it; }
    auto operator*() const { return Value{count, *it}; }
  };
  struct Enumerate
  {
    Container c;
    auto begin() { using std::begin; return Iter{0, begin(c)}; }
    auto end() { using std::end; return end(c); }
  };
  return Enumerate{std::forward<Container>(container)};
}

template<typename Container,
         typename = std::enable_if_t<can_enumerate_v<Container>>>
inline
auto
cenumerate(const Container &container)
{
  return enumerate(container);
}

template<typename Counter,
         typename = std::enable_if_t<can_count_v<Counter>>>
inline
auto
enumerate(Counter from_count,
          Counter to_count)
{
  struct Iter
  {
    Counter count;
    bool operator!=(const Counter &rhs) const { return count!=rhs; }
    void operator++() { ++count; }
    auto operator*() const { return count; }
  };
  struct Enumerate
  {
    Counter from_count;
    Counter to_count;
    auto begin() { return Iter{from_count}; }
    auto end() { return to_count; }
  };
  return Enumerate{from_count, to_count};
}

template<typename Counter,
         typename = std::enable_if_t<can_count_v<Counter>>>
inline
auto
enumerate(Counter to_count)
{
  return enumerate(Counter{}, to_count);
}

} // namespace dim

#endif // DIM_ENUMERATE_HPP_IMPL

//----------------------------------------------------------------------------
