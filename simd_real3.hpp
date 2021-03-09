//----------------------------------------------------------------------------

#ifndef DIM_SIMD_REAL3_HPP
#define DIM_SIMD_REAL3_HPP

#include "simd.hpp"

namespace dim::simd {

template<typename VectorType>
struct Real3
{
  using simd_t = Simd<VectorType>;
  using real_t = typename simd_t::value_type;

  static_assert(std::is_floating_point_v<real_t>,
                "floating point expected for template type");

  simd_t x, y, z;

  constexpr Real3() : x{}, y{}, z{} {}
  constexpr explicit Real3(simd_t r) : x{r}, y{r}, z{r} {}
  constexpr Real3(simd_t x, simd_t y, simd_t z) : x{x}, y{y}, z{z} {}
};

template<typename VectorType>
inline constexpr
Real3<VectorType>
operator-(const Real3<VectorType> &rhs)
{
  return Real3<VectorType>{-rhs.x,
                           -rhs.y,
                           -rhs.z};
}

template<typename VectorType>
inline constexpr
Real3<VectorType> &
operator+=(Real3<VectorType> &lhs,
           const Real3<VectorType> &rhs)
{
  lhs.x+=rhs.x;
  lhs.y+=rhs.y;
  lhs.z+=rhs.z;
  return lhs;
}

template<typename VectorType>
inline constexpr
Real3<VectorType>
operator+(const Real3<VectorType> &lhs,
          const Real3<VectorType> &rhs)
{
  return Real3<VectorType>{lhs.x+rhs.x,
                           lhs.y+rhs.y,
                           lhs.z+rhs.z};
}

template<typename VectorType>
inline constexpr
Real3<VectorType> &
operator-=(Real3<VectorType> &lhs,
           const Real3<VectorType> &rhs)
{
  lhs.x-=rhs.x;
  lhs.y-=rhs.y;
  lhs.z-=rhs.z;
  return lhs;
}

template<typename VectorType>
inline constexpr
Real3<VectorType>
operator-(const Real3<VectorType> &lhs,
          const Real3<VectorType> &rhs)
{
  return Real3<VectorType>{lhs.x-rhs.x,
                           lhs.y-rhs.y,
                           lhs.z-rhs.z};
}

template<typename VectorType>
inline constexpr
Real3<VectorType> &
operator*=(Real3<VectorType> &lhs,
           const Real3<VectorType> &rhs)
{
  lhs.x*=rhs.x;
  lhs.y*=rhs.y;
  lhs.z*=rhs.z;
  return lhs;
}

template<typename VectorType>
inline constexpr
Real3<VectorType>
operator*(const Real3<VectorType> &lhs,
          const Real3<VectorType> &rhs)
{
  return Real3<VectorType>{lhs.x*rhs.x,
                           lhs.y*rhs.y,
                           lhs.z*rhs.z};
}

template<typename VectorType>
inline constexpr
Real3<VectorType> &
operator/=(Real3<VectorType> &lhs,
           const Real3<VectorType> &rhs)
{
  lhs.x/=rhs.x;
  lhs.y/=rhs.y;
  lhs.z/=rhs.z;
  return lhs;
}

template<typename VectorType>
inline constexpr
Real3<VectorType>
operator/(const Real3<VectorType> &lhs,
          const Real3<VectorType> &rhs)
{
  return Real3<VectorType>{lhs.x/rhs.x,
                           lhs.y/rhs.y,
                           lhs.z/rhs.z};
}

template<typename VectorType>
inline constexpr
Real3<VectorType>
cross(const Real3<VectorType> &lhs,
      const Real3<VectorType> &rhs)
{
  return Real3<VectorType>{lhs.y*rhs.z-lhs.z*rhs.y,
                           lhs.z*rhs.x-lhs.x*rhs.z,
                           lhs.x*rhs.y-lhs.y*rhs.x};
}

template<typename VectorType>
inline constexpr
VectorType
dot(const Real3<VectorType> &lhs,
    const Real3<VectorType> &rhs)
{
  return lhs.x*rhs.x+
         lhs.y*rhs.y+
         lhs.z*rhs.z;
}

template<typename VectorType>
inline constexpr
VectorType
sqr_magnitude(const Real3<VectorType> &r3)
{
  return dot(r3, r3);
}

template<typename VectorType>
inline constexpr
VectorType
magnitude(const Real3<VectorType> &r3)
{
  return std::sqrt(sqr_magnitude(r3));
}

template<typename VectorType>
inline constexpr
void
normalise(Real3<VectorType> &r3)
{
  using simd_t = typename Real3<VectorType>::simd_t;
  using real_t = typename Real3<VectorType>::real_t;
  constexpr auto eps=simd_t{std::numeric_limits<real_t>::epsilon()};
  constexpr auto one=simd_t{real_t(1.0)};
  const auto mag=magnitude(r3);
  r3*=Real3<VectorType>{select(mag>eps, one/mag, one)};
}

template<typename VectorType>
inline constexpr
Real3<VectorType>
normalised(const Real3<VectorType> &r3)
{
  auto result=r3;
  normalise(result);
  return result;
}

template<typename VectorType>
inline constexpr
Real3<VectorType>
min_coord(const Real3<VectorType> &lhs,
          const Real3<VectorType> &rhs)
{
  return Real3<VectorType>{min(lhs.x, rhs.x),
                           min(lhs.y, rhs.y),
                           min(lhs.z, rhs.z)};
}

template<typename VectorType>
inline constexpr
Real3<VectorType>
max_coord(const Real3<VectorType> &lhs,
          const Real3<VectorType> &rhs)
{
  return Real3<VectorType>{max(lhs.x, rhs.x),
                           max(lhs.y, rhs.y),
                           max(lhs.z, rhs.z)};
}

template<typename VectorType>
inline
std::string
to_string(const Real3<VectorType> &r3)
{
  return '{'+to_string(r3.x)+", "+
             to_string(r3.y)+", "+
             to_string(r3.z)+'}';
}

template<typename VectorType>
inline
std::ostream &
operator<<(std::ostream &os,
           const Real3<VectorType> &r3)
{
  return os << '{' << r3.x << ", "
                   << r3.y << ", "
                   << r3.z << '}';
}

} // namespace dim::simd

#endif // DIM_SIMD_REAL3_HPP

//----------------------------------------------------------------------------
