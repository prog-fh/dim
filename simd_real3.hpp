//----------------------------------------------------------------------------

#ifndef DIM_SIMD_REAL3_HPP
#define DIM_SIMD_REAL3_HPP

#include "simd.hpp"

namespace dim::simd {

template<typename SimdType>
struct Real3
{
  using real_t = typename SimdType::value_type;
  using simd_t = simd::simd_t<real_t, SimdType::vector_size>;

  static_assert(std::is_floating_point_v<real_t>,
                "simd floating point expected for template type");

  simd_t x, y, z;

  constexpr Real3() : x{}, y{}, z{} {}
  constexpr explicit Real3(simd_t r) : x{r}, y{r}, z{r} {}
  constexpr Real3(simd_t x, simd_t y, simd_t z) : x{x}, y{y}, z{z} {}
};

template<typename SimdType>
inline constexpr
Real3<SimdType>
operator-(const Real3<SimdType> &rhs)
{
  return {-rhs.x,
          -rhs.y,
          -rhs.z};
}

template<typename SimdType>
inline constexpr
Real3<SimdType> &
operator+=(Real3<SimdType> &lhs,
           const Real3<SimdType> &rhs)
{
  lhs.x+=rhs.x;
  lhs.y+=rhs.y;
  lhs.z+=rhs.z;
  return lhs;
}

template<typename SimdType>
inline constexpr
Real3<SimdType>
operator+(const Real3<SimdType> &lhs,
          const Real3<SimdType> &rhs)
{
  return {lhs.x+rhs.x,
          lhs.y+rhs.y,
          lhs.z+rhs.z};
}

template<typename SimdType>
inline constexpr
Real3<SimdType> &
operator-=(Real3<SimdType> &lhs,
           const Real3<SimdType> &rhs)
{
  lhs.x-=rhs.x;
  lhs.y-=rhs.y;
  lhs.z-=rhs.z;
  return lhs;
}

template<typename SimdType>
inline constexpr
Real3<SimdType>
operator-(const Real3<SimdType> &lhs,
          const Real3<SimdType> &rhs)
{
  return {lhs.x-rhs.x,
          lhs.y-rhs.y,
          lhs.z-rhs.z};
}

template<typename SimdType>
inline constexpr
Real3<SimdType> &
operator*=(Real3<SimdType> &lhs,
           const Real3<SimdType> &rhs)
{
  lhs.x*=rhs.x;
  lhs.y*=rhs.y;
  lhs.z*=rhs.z;
  return lhs;
}

template<typename SimdType>
inline constexpr
Real3<SimdType>
operator*(const Real3<SimdType> &lhs,
          const Real3<SimdType> &rhs)
{
  return {lhs.x*rhs.x,
          lhs.y*rhs.y,
          lhs.z*rhs.z};
}

template<typename SimdType>
inline constexpr
Real3<SimdType> &
operator/=(Real3<SimdType> &lhs,
           const Real3<SimdType> &rhs)
{
  lhs.x/=rhs.x;
  lhs.y/=rhs.y;
  lhs.z/=rhs.z;
  return lhs;
}

template<typename SimdType>
inline constexpr
Real3<SimdType>
operator/(const Real3<SimdType> &lhs,
          const Real3<SimdType> &rhs)
{
  return {lhs.x/rhs.x,
          lhs.y/rhs.y,
          lhs.z/rhs.z};
}

template<typename SimdType>
inline constexpr
Real3<SimdType>
cross(const Real3<SimdType> &lhs,
      const Real3<SimdType> &rhs)
{
  return {lhs.y*rhs.z-lhs.z*rhs.y,
          lhs.z*rhs.x-lhs.x*rhs.z,
          lhs.x*rhs.y-lhs.y*rhs.x};
}

template<typename SimdType>
inline constexpr
SimdType
dot(const Real3<SimdType> &lhs,
    const Real3<SimdType> &rhs)
{
  return lhs.x*rhs.x+
         lhs.y*rhs.y+
         lhs.z*rhs.z;
}

template<typename SimdType>
inline constexpr
SimdType
sqr_magnitude(const Real3<SimdType> &r3)
{
  return dot(r3, r3);
}

template<typename SimdType>
inline constexpr
SimdType
magnitude(const Real3<SimdType> &r3)
{
  return sqrt(sqr_magnitude(r3));
}

template<typename SimdType>
inline constexpr
void
normalise(Real3<SimdType> &r3)
{
  using simd_t = typename Real3<SimdType>::simd_t;
  using real_t = typename Real3<SimdType>::real_t;
  constexpr auto eps=simd_t{std::numeric_limits<real_t>::epsilon()};
  constexpr auto one=simd_t{real_t(1.0)};
  const auto mag=magnitude(r3);
  r3*=Real3<simd_t>{select(mag>eps, one/mag, one)};
}

template<typename SimdType>
inline constexpr
Real3<SimdType>
normalised(const Real3<SimdType> &r3)
{
  auto result=r3;
  normalise(result);
  return result;
}

template<typename SimdType>
inline constexpr
Real3<SimdType>
min_coord(const Real3<SimdType> &lhs,
          const Real3<SimdType> &rhs)
{
  return {min(lhs.x, rhs.x),
          min(lhs.y, rhs.y),
          min(lhs.z, rhs.z)};
}

template<typename SimdType>
inline constexpr
Real3<SimdType>
max_coord(const Real3<SimdType> &lhs,
          const Real3<SimdType> &rhs)
{
  return {max(lhs.x, rhs.x),
          max(lhs.y, rhs.y),
          max(lhs.z, rhs.z)};
}

template<typename SimdType>
inline constexpr
void
rotate_x(Real3<SimdType> &r3,
         typename SimdType::real_t angle)
{
  const auto ca=std::cos(angle), sa=std::sin(angle);
  const auto y=r3.y*ca-r3.z*sa,
             z=r3.y*sa+r3.z*ca;
  r3.y=y;
  r3.z=z;
}

template<typename SimdType>
inline constexpr
Real3<SimdType>
rotated_x(const Real3<SimdType> &r3,
          typename SimdType::real_t angle)
{
  auto result=r3;
  rotate_x(result, angle);
  return result;
}

template<typename SimdType>
inline constexpr
void
rotate_y(Real3<SimdType> &r3,
         typename SimdType::real_t angle)
{
  const auto ca=std::cos(angle), sa=std::sin(angle);
  const auto x=r3.z*sa+r3.x*ca,
             z=r3.z*ca-r3.x*sa;
  r3.x=x;
  r3.z=z;
}

template<typename SimdType>
inline constexpr
Real3<SimdType>
rotated_y(const Real3<SimdType> &r3,
          typename SimdType::real_t angle)
{
  auto result=r3;
  rotate_y(result, angle);
  return result;
}

template<typename SimdType>
inline constexpr
void
rotate_z(Real3<SimdType> &r3,
         typename SimdType::real_t angle)
{
  const auto ca=std::cos(angle), sa=std::sin(angle);
  const auto x=r3.x*ca-r3.y*sa,
             y=r3.x*sa+r3.y*ca;
  r3.x=x;
  r3.y=y;
}

template<typename SimdType>
inline constexpr
Real3<SimdType>
rotated_z(const Real3<SimdType> &r3,
          typename SimdType::real_t angle)
{
  auto result=r3;
  rotate_z(result, angle);
  return result;
}

template<typename SimdType>
inline
std::string
to_string(const Real3<SimdType> &r3)
{
  return '{'+to_string(r3.x)+", "+
             to_string(r3.y)+", "+
             to_string(r3.z)+'}';
}

template<typename SimdType>
inline
std::ostream &
operator<<(std::ostream &os,
           const Real3<SimdType> &r3)
{
  return os << '{' << r3.x << ", "
                   << r3.y << ", "
                   << r3.z << '}';
}

} // namespace dim::simd

#endif // DIM_SIMD_REAL3_HPP

//----------------------------------------------------------------------------
