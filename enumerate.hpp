//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifndef DIM_ENUMERATE_HPP
#define DIM_ENUMERATE_HPP

/**
inspired from
  http://reedbeta.com/blog/python-like-enumerate-in-cpp17/
**/

#include <type_traits>
#include <iterator>
#include <utility>

namespace dim {

namespace impl_ {

template<typename Container>
using enumerate_container_support =
  decltype(*begin(std::declval<Container>()),
           *end(std::declval<Container>()),
           size(std::declval<Container>()),
           void(), 1);

template<typename Counter>
using enumerate_counter_support =
  decltype((std::declval<Counter>()!=std::declval<Counter>()),
           ++std::declval<std::add_lvalue_reference_t<Counter>>(),
           void(), 1);

} // namespace impl_

template<typename Container,
         impl_::enumerate_container_support<Container> =1>
inline
auto // iterable
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
         impl_::enumerate_container_support<Container> =1>
inline
auto // iterable
cenumerate(const Container &container)
{
  return enumerate(container);
}

template<typename Counter,
         impl_::enumerate_counter_support<Counter> =1>
inline
auto // iterable
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
         impl_::enumerate_counter_support<Counter> =1>
inline
auto // iterable
enumerate(Counter to_count)
{
  return enumerate(Counter{}, to_count);
}

} // namespace dim

#endif // DIM_ENUMERATE_HPP

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
