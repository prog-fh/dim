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
  static_assert(std::is_floating_point_v<RealType>,
                "floating point expected for template type");

  using real_t = RealType;

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
  return {-rhs.x,
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
  return {lhs.x+rhs.x,
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
  return {lhs.x-rhs.x,
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
  return {lhs.x*rhs.x,
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
  return {lhs.x/rhs.x,
          lhs.y/rhs.y,
          lhs.z/rhs.z};
}

template<typename RealType>
inline constexpr
Real3<RealType>
cross(const Real3<RealType> &lhs,
      const Real3<RealType> &rhs)
{
  return {lhs.y*rhs.z-lhs.z*rhs.y,
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
fmin(const Real3<RealType> &lhs,
     const Real3<RealType> &rhs)
{
  return {std::fmin(lhs.x, rhs.x),
          std::fmin(lhs.y, rhs.y),
          std::fmin(lhs.z, rhs.z)};
}

template<typename RealType>
inline constexpr
Real3<RealType>
fmax(const Real3<RealType> &lhs,
     const Real3<RealType> &rhs)
{
  return {std::fmax(lhs.x, rhs.x),
          std::fmax(lhs.y, rhs.y),
          std::fmax(lhs.z, rhs.z)};
}

template<typename RealType>
inline constexpr
void
rotate_x(Real3<RealType> &r3,
         RealType angle)
{
  const auto ca=std::cos(angle), sa=std::sin(angle);
  const auto y=r3.y*ca-r3.z*sa,
             z=r3.y*sa+r3.z*ca;
  r3.y=y;
  r3.z=z;
}

template<typename RealType>
inline constexpr
Real3<RealType>
rotated_x(const Real3<RealType> &r3,
          RealType angle)
{
  auto result=r3;
  rotate_x(result, angle);
  return result;
}

template<typename RealType>
inline constexpr
void
rotate_y(Real3<RealType> &r3,
         RealType angle)
{
  const auto ca=std::cos(angle), sa=std::sin(angle);
  const auto x=r3.z*sa+r3.x*ca,
             z=r3.z*ca-r3.x*sa;
  r3.x=x;
  r3.z=z;
}

template<typename RealType>
inline constexpr
Real3<RealType>
rotated_y(const Real3<RealType> &r3,
          RealType angle)
{
  auto result=r3;
  rotate_y(result, angle);
  return result;
}

template<typename RealType>
inline constexpr
void
rotate_z(Real3<RealType> &r3,
         RealType angle)
{
  const auto ca=std::cos(angle), sa=std::sin(angle);
  const auto x=r3.x*ca-r3.y*sa,
             y=r3.x*sa+r3.y*ca;
  r3.x=x;
  r3.y=y;
}

template<typename RealType>
inline constexpr
Real3<RealType>
rotated_z(const Real3<RealType> &r3,
          RealType angle)
{
  auto result=r3;
  rotate_z(result, angle);
  return result;
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
