//----------------------------------------------------------------------------

#ifndef DIM_UTILS_HPP
#define DIM_UTILS_HPP 1

#include <cstdint>

namespace dim {

std::int64_t // microseconds since 1970/01/01 00:00:00 UTC
system_time_us();

double // seconds (1e-6 precision) since 1970/01/01 00:00:00 UTC
system_time();

} // namespace dim

#endif // DIM_UTILS_HPP

//----------------------------------------------------------------------------
// inline implementation details (don't look below!)
//----------------------------------------------------------------------------

#ifndef DIM_UTILS_HPP_IMPL
#define DIM_UTILS_HPP_IMPL 1

#include <chrono>

namespace dim {

inline
std::int64_t // microseconds since 1970/01/01 00:00:00 UTC
system_time_ms()
{
  const auto now=std::chrono::system_clock::now().time_since_epoch();
  return std::int64_t(std::chrono::duration_cast
                      <std::chrono::microseconds>(now).count());
}

inline
double // seconds (1e-6 precision) since 1970/01/01 00:00:00 UTC
system_time()
{
  return 1e-6*double(system_time_ms());
}

} // namespace dim

#endif // DIM_UTILS_HPP_IMPL

//----------------------------------------------------------------------------
