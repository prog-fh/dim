//----------------------------------------------------------------------------

#ifndef DIM_REAL3_HPP
#define DIM_REAL3_HPP

#include <limits>
#include <cmath>
#include <algorithm>
#include <string>
#include <iostream>

namespace dim {

template<typename RealType>
struct Real3
{
  using real_t = RealType;

  static_assert(std::is_floating_point_v<real_t>,
                "floating point expected for template type");

  real_t x, y, z;

  constexpr Real3() : x{}, y{}, z{} {}
  constexpr explicit Real3(real_t r) : x{r}, y{r}, z{r} {}
  constexpr Real3(real_t x, real_t y, real_t z) : x{x}, y{y}, z{z} {}
};

template<typename RealType>
inline constexpr
Real3<RealType>
operator-(const Real3<RealType> &rhs)
{
  return Real3<RealType>{-rhs.x,
                         -rhs.y,
                         -rhs.z};
}

template<typename RealType>
inline constexpr
Real3<RealType> &
operator+=(Real3<RealType> &lhs,
           const Real3<RealType> &rhs)
{
  lhs.x+=rhs.x;
  lhs.y+=rhs.y;
  lhs.z+=rhs.z;
  return lhs;
}

template<typename RealType>
inline constexpr
Real3<RealType>
operator+(const Real3<RealType> &lhs,
          const Real3<RealType> &rhs)
{
  return Real3<RealType>{lhs.x+rhs.x,
                         lhs.y+rhs.y,
                         lhs.z+rhs.z};
}

template<typename RealType>
inline constexpr
Real3<RealType> &
operator-=(Real3<RealType> &lhs,
           const Real3<RealType> &rhs)
{
  lhs.x-=rhs.x;
  lhs.y-=rhs.y;
  lhs.z-=rhs.z;
  return lhs;
}

template<typename RealType>
inline constexpr
Real3<RealType>
operator-(const Real3<RealType> &lhs,
          const Real3<RealType> &rhs)
{
  return Real3<RealType>{lhs.x-rhs.x,
                         lhs.y-rhs.y,
                         lhs.z-rhs.z};
}

template<typename RealType>
inline constexpr
Real3<RealType> &
operator*=(Real3<RealType> &lhs,
           const Real3<RealType> &rhs)
{
  lhs.x*=rhs.x;
  lhs.y*=rhs.y;
  lhs.z*=rhs.z;
  return lhs;
}

template<typename RealType>
inline constexpr
Real3<RealType>
operator*(const Real3<RealType> &lhs,
          const Real3<RealType> &rhs)
{
  return Real3<RealType>{lhs.x*rhs.x,
                         lhs.y*rhs.y,
                         lhs.z*rhs.z};
}

template<typename RealType>
inline constexpr
Real3<RealType> &
operator/=(Real3<RealType> &lhs,
           const Real3<RealType> &rhs)
{
  lhs.x/=rhs.x;
  lhs.y/=rhs.y;
  lhs.z/=rhs.z;
  return lhs;
}

template<typename RealType>
inline constexpr
Real3<RealType>
operator/(const Real3<RealType> &lhs,
          const Real3<RealType> &rhs)
{
  return Real3<RealType>{lhs.x/rhs.x,
                         lhs.y/rhs.y,
                         lhs.z/rhs.z};
}

template<typename RealType>
inline constexpr
Real3<RealType>
cross(const Real3<RealType> &lhs,
      const Real3<RealType> &rhs)
{
  return Real3<RealType>{lhs.y*rhs.z-lhs.z*rhs.y,
                         lhs.z*rhs.x-lhs.x*rhs.z,
                         lhs.x*rhs.y-lhs.y*rhs.x};
}

template<typename RealType>
inline constexpr
RealType
dot(const Real3<RealType> &lhs,
    const Real3<RealType> &rhs)
{
  return lhs.x*rhs.x+
         lhs.y*rhs.y+
         lhs.z*rhs.z;
}

template<typename RealType>
inline constexpr
RealType
sqr_magnitude(const Real3<RealType> &r3)
{
  return dot(r3, r3);
}

template<typename RealType>
inline constexpr
RealType
magnitude(const Real3<RealType> &r3)
{
  return std::sqrt(sqr_magnitude(r3));
}

template<typename RealType>
inline constexpr
void
normalise(Real3<RealType> &r3)
{
  constexpr auto eps=std::numeric_limits<RealType>::epsilon();
  constexpr auto one=RealType(1.0);
  const auto mag=magnitude(r3);
  r3*=Real3<RealType>{mag>eps ? one/mag : one};
}

template<typename RealType>
inline constexpr
Real3<RealType>
normalised(const Real3<RealType> &r3)
{
  auto result=r3;
  normalise(result);
  return result;
}

template<typename RealType>
inline constexpr
Real3<RealType>
min_coord(const Real3<RealType> &lhs,
          const Real3<RealType> &rhs)
{
  return Real3<RealType>{std::min(lhs.x, rhs.x),
                         std::min(lhs.y, rhs.y),
                         std::min(lhs.z, rhs.z)};
}

template<typename RealType>
inline constexpr
Real3<RealType>
max_coord(const Real3<RealType> &lhs,
          const Real3<RealType> &rhs)
{
  return Real3<RealType>{std::max(lhs.x, rhs.x),
                         std::max(lhs.y, rhs.y),
                         std::max(lhs.z, rhs.z)};
}

template<typename RealType>
inline
std::string
to_string(const Real3<RealType> &r3)
{
  return '{'+std::to_string(r3.x)+", "+
             std::to_string(r3.y)+", "+
             std::to_string(r3.z)+'}';
}

template<typename RealType>
inline
std::ostream &
operator<<(std::ostream &os,
           const Real3<RealType> &r3)
{
  return os << '{' << r3.x << ", "
                   << r3.y << ", "
                   << r3.z << '}';
}

} // namespace dim

#endif // DIM_REAL3_HPP

//----------------------------------------------------------------------------
