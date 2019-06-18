//----------------------------------------------------------------------------

#ifndef DIM_UTILS_HPP
#define DIM_UTILS_HPP 1

#include <cstdint>
#include <chrono>
#include <tuple>
#include <type_traits>
#include <algorithm>

#define DIM_RESTRICT __restrict__

#define DIM_ASSUME_ALIGNED(a) __attribute__((assume_aligned(a)))

namespace dim {

constexpr auto assumed_cacheline_size = 64;

inline
std::int64_t // microseconds since 1970/01/01 00:00:00 UTC
system_time_us()
{
  const auto now=std::chrono::system_clock::now().time_since_epoch();
  return std::int64_t(std::chrono::duration_cast
                      <std::chrono::microseconds>(now).count());
}

double // seconds (1e-6 precision) since 1970/01/01 00:00:00 UTC
system_time()
{
  return 1e-6*double(system_time_us());
}

template<typename T>
inline
std::tuple<T, // part_begin
           T> // part_end
sequence_part(T seq_begin,
              T seq_end,
              int part_id,
              int part_count)
{
  // use a wide integer to prevent overflow in multiplication
  using wide_t =
    std::conditional_t<std::is_unsigned_v<T>, std::uintmax_t, std::intmax_t>;
  const auto seq_size=wide_t{seq_end-seq_begin};
  auto part_begin=T(seq_size*part_id/part_count);
  auto part_end=T(std::min(seq_size, seq_size*(part_id+1)/part_count));
  return {seq_begin+part_begin, seq_begin+part_end};
}

template<typename T>
inline
std::tuple<T, // part_begin
           T> // part_end
sequence_part(T seq_size,
              int part_id,
              int part_count)
{
  return sequence_part(T{}, seq_size, part_id, part_count);
}

} // namespace dim

#endif // DIM_UTILS_HPP

//----------------------------------------------------------------------------
