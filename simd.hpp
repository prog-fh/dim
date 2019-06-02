//----------------------------------------------------------------------------

#ifndef DIM_SIMD_HPP
#define DIM_SIMD_HPP 1

#include <immintrin.h>
#include <cstdint>
#include <type_traits>
#include <tuple>
#include <cmath>
#include <string>
#include <iostream>

namespace dim::simd {

namespace impl_ {

template<typename ValueType,
         int VectorSize>
struct simd_helper { using type = void; };

} // namespace impl_

template<typename ValueType,
         int VectorSize>
using simd_t = typename impl_::simd_helper<
  typename std::decay<ValueType>::type, VectorSize>::type;

template<typename VectorType>
class Simd
{
public:

  using vector_type = VectorType;
  using value_type =
    typename std::decay<decltype(std::declval<vector_type>()[0])>::type;

  static constexpr auto vector_size = int(sizeof(vector_type));
  static constexpr auto value_size  = int(sizeof(value_type));
  static constexpr auto value_count = vector_size/value_size;

  using mask_type = simd_t<
    typename std::conditional<value_size==1, std::int8_t,
    typename std::conditional<value_size==2, std::int16_t,
    typename std::conditional<value_size==4, std::int32_t,
    typename std::conditional<value_size==8, std::int64_t,
    void>::type>::type>::type>::type, vector_size>;

  constexpr Simd() : v_{} {};
  constexpr Simd(vector_type v) : v_{v} {};
  constexpr Simd & operator=(vector_type v) { v_=v; return *this; }
  constexpr Simd(value_type v)
  : v_{
    [&v]()
    {
      constexpr auto c=value_count;
      if constexpr(c==64)
      {
        return vector_type{v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v,
                           v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v,
                           v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v,
                           v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v};
      }
      else if constexpr(c==32)
      {
        return vector_type{v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v,
                           v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v};
      }
      else if constexpr(c==16)
      {
        return vector_type{v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v};
      }
      else if constexpr(c==8)
      {
        return vector_type{v, v, v, v, v, v, v, v};
      }
      else if constexpr(c==4)
      {
        return vector_type{v, v, v, v};
      }
      else if constexpr(c==2)
      {
        return vector_type{v, v};
      }
      else
      {
        return vector_type{v};
      }
    }()} {}
  constexpr Simd & operator=(value_type v) { v_=Simd{v}; return *this; }

  constexpr Simd(const Simd &) =default;
  constexpr Simd & operator=(const Simd &) =default;
  constexpr Simd(Simd &&) =default;
  constexpr Simd & operator=(Simd &&) =default;
  ~Simd() =default;

  constexpr const vector_type & vec() const { return v_; }
  constexpr       vector_type & vec()       { return v_; }

  constexpr value_type operator[](int i) const { return v_[i]; }

private:
  vector_type v_;
};

template<typename SimdType,
         typename ValueType>
using expect_value_type =
  typename std::enable_if<
    std::is_same<
      typename SimdType::value_type,
      typename std::decay<ValueType>::type
      >::value
    >::type;

template<typename SimdType,
         int VectorSize>
using expect_vector_size =
  typename std::enable_if<
    std::is_same<
      typename std::integral_constant<int, SimdType::vector_size>,
      typename std::integral_constant<int, VectorSize>
      >::value
    >::type;

template<typename SimdType,
         int ValueSize>
using expect_value_size =
  typename std::enable_if<
    std::is_same<
      typename std::integral_constant<int, SimdType::value_size>,
      typename std::integral_constant<int, ValueSize>
      >::value
    >::type;

template<typename SimdType,
         int ValueCount>
using expect_value_count =
  typename std::enable_if<
    std::is_same<
      typename std::integral_constant<int, SimdType::value_count>,
      typename std::integral_constant<int, ValueCount>
      >::value
    >::type;

//~~~~ enumerate available simd types ~~~~

#if __AVX512F__
# define DIM_SIMD_MAX_VECTOR_SIZE 64
#elif __AVX__
# define DIM_SIMD_MAX_VECTOR_SIZE 32
#else
# define DIM_SIMD_MAX_VECTOR_SIZE 16
#endif

constexpr auto max_vector_size=DIM_SIMD_MAX_VECTOR_SIZE;

#define DIM_SIMD_DEFINE_TYPE(name, base, vec_size) \
  using name = Simd<base __attribute__((__vector_size__(vec_size)))>; \
  template<> struct impl_::simd_helper<base, vec_size> { using type = name; };

#if DIM_SIMD_MAX_VECTOR_SIZE>=64
  DIM_SIMD_DEFINE_TYPE( u8x64_t,  std::uint8_t, 64)
  DIM_SIMD_DEFINE_TYPE( i8x64_t,   std::int8_t, 64)
  DIM_SIMD_DEFINE_TYPE(u16x32_t, std::uint16_t, 64)
  DIM_SIMD_DEFINE_TYPE(i16x32_t,  std::int16_t, 64)
  DIM_SIMD_DEFINE_TYPE(u32x16_t, std::uint32_t, 64)
  DIM_SIMD_DEFINE_TYPE(i32x16_t,  std::int32_t, 64)
  DIM_SIMD_DEFINE_TYPE( u64x8_t, std::uint64_t, 64)
  DIM_SIMD_DEFINE_TYPE( i64x8_t,  std::int64_t, 64)
  DIM_SIMD_DEFINE_TYPE(r32x16_t,         float, 64)
  DIM_SIMD_DEFINE_TYPE( r64x8_t,        double, 64)
#endif

#if DIM_SIMD_MAX_VECTOR_SIZE>=32
  DIM_SIMD_DEFINE_TYPE( u8x32_t,  std::uint8_t, 32)
  DIM_SIMD_DEFINE_TYPE( i8x32_t,   std::int8_t, 32)
  DIM_SIMD_DEFINE_TYPE(u16x16_t, std::uint16_t, 32)
  DIM_SIMD_DEFINE_TYPE(i16x16_t,  std::int16_t, 32)
  DIM_SIMD_DEFINE_TYPE( u32x8_t, std::uint32_t, 32)
  DIM_SIMD_DEFINE_TYPE( i32x8_t,  std::int32_t, 32)
  DIM_SIMD_DEFINE_TYPE( u64x4_t, std::uint64_t, 32)
  DIM_SIMD_DEFINE_TYPE( i64x4_t,  std::int64_t, 32)
  DIM_SIMD_DEFINE_TYPE( r32x8_t,         float, 32)
  DIM_SIMD_DEFINE_TYPE( r64x4_t,        double, 32)
#endif

#if DIM_SIMD_MAX_VECTOR_SIZE>=16
  DIM_SIMD_DEFINE_TYPE( u8x16_t,  std::uint8_t, 16)
  DIM_SIMD_DEFINE_TYPE( i8x16_t,   std::int8_t, 16)
  DIM_SIMD_DEFINE_TYPE( u16x8_t, std::uint16_t, 16)
  DIM_SIMD_DEFINE_TYPE( i16x8_t,  std::int16_t, 16)
  DIM_SIMD_DEFINE_TYPE( u32x4_t, std::uint32_t, 16)
  DIM_SIMD_DEFINE_TYPE( i32x4_t,  std::int32_t, 16)
  DIM_SIMD_DEFINE_TYPE( u64x2_t, std::uint64_t, 16)
  DIM_SIMD_DEFINE_TYPE( i64x2_t,  std::int64_t, 16)
  DIM_SIMD_DEFINE_TYPE( r32x4_t,         float, 16)
  DIM_SIMD_DEFINE_TYPE( r64x2_t,        double, 16)
#endif

#undef DIM_SIMD_MAX_VECTOR_SIZE
#undef DIM_SIMD_DEFINE_TYPE

using u8_t  = simd_t< std::uint8_t, max_vector_size>;
using i8_t  = simd_t<  std::int8_t, max_vector_size>;
using u16_t = simd_t<std::uint16_t, max_vector_size>;
using i16_t = simd_t< std::int16_t, max_vector_size>;
using u32_t = simd_t<std::uint32_t, max_vector_size>;
using i32_t = simd_t< std::int32_t, max_vector_size>;
using u64_t = simd_t<std::uint64_t, max_vector_size>;
using i64_t = simd_t< std::int64_t, max_vector_size>;
using r32_t = simd_t<        float, max_vector_size>;
using r64_t = simd_t<       double, max_vector_size>;

//~~~~ arithmetic and logic operators ~~~~

#define DIM_SIMD_FORWARD_UNARY(op) \
        template<typename VectorType> \
        inline constexpr \
        auto \
        operator op(Simd<VectorType> rhs) \
        { \
          return Simd{op rhs.vec()}; \
        }
DIM_SIMD_FORWARD_UNARY(+)
DIM_SIMD_FORWARD_UNARY(-)
DIM_SIMD_FORWARD_UNARY(~)
#undef DIM_SIMD_FORWARD_UNARY
#define DIM_SIMD_FORWARD_BINARY(op) \
        template<typename LhsVectorType, \
                 typename RhsVectorType> \
        inline constexpr \
        auto \
        operator op(Simd<LhsVectorType> lhs, \
                    Simd<RhsVectorType> rhs) \
        { \
          return Simd{lhs.vec() op rhs.vec()}; \
        } \
        template<typename VectorType> \
        inline constexpr \
        auto \
        operator op(Simd<VectorType> lhs, \
                    typename Simd<VectorType>::value_type rhs) \
        { \
          return Simd{lhs.vec() op rhs}; \
        } \
        template<typename VectorType> \
        inline constexpr \
        auto \
        operator op(typename Simd<VectorType>::value_type lhs, \
                    Simd<VectorType> rhs) \
        { \
          return Simd{lhs op rhs.vec()}; \
        }
#define DIM_SIMD_FORWARD_BINARY_ASSIGN(op) \
        DIM_SIMD_FORWARD_BINARY(op) \
        template<typename LhsVectorType, \
                 typename RhsVectorType> \
        inline constexpr \
        auto & \
        operator op##=(Simd<LhsVectorType> &lhs, \
                       Simd<RhsVectorType> rhs) \
        { \
          lhs.vec() op##= rhs.vec(); \
          return lhs; \
        } \
        template<typename VectorType> \
        inline constexpr \
        auto & \
        operator op##=(Simd<VectorType> &lhs, \
                       typename Simd<VectorType>::value_type rhs) \
        { \
          lhs.vec() op##= rhs; \
          return lhs; \
        }
DIM_SIMD_FORWARD_BINARY_ASSIGN(+)
DIM_SIMD_FORWARD_BINARY_ASSIGN(-)
DIM_SIMD_FORWARD_BINARY_ASSIGN(*)
DIM_SIMD_FORWARD_BINARY_ASSIGN(/)
DIM_SIMD_FORWARD_BINARY_ASSIGN(%)
DIM_SIMD_FORWARD_BINARY_ASSIGN(&)
DIM_SIMD_FORWARD_BINARY_ASSIGN(|)
DIM_SIMD_FORWARD_BINARY_ASSIGN(^)
DIM_SIMD_FORWARD_BINARY_ASSIGN(<<)
DIM_SIMD_FORWARD_BINARY_ASSIGN(>>)
#undef DIM_SIMD_FORWARD_BINARY_ASSIGN
DIM_SIMD_FORWARD_BINARY(==)
DIM_SIMD_FORWARD_BINARY(!=)
DIM_SIMD_FORWARD_BINARY(<)
DIM_SIMD_FORWARD_BINARY(<=)
DIM_SIMD_FORWARD_BINARY(>)
DIM_SIMD_FORWARD_BINARY(>=)
DIM_SIMD_FORWARD_BINARY(&&)
DIM_SIMD_FORWARD_BINARY(||)
#undef DIM_SIMD_FORWARD_BINARY

//~~~~ selection ~~~~

template<typename ConditionVectorType,
         typename ValueVectorType>
inline constexpr
auto
select(Simd<ConditionVectorType> condition,
       Simd<ValueVectorType> true_value,
       Simd<ValueVectorType> false_value)
{
  using mask_t = typename Simd<ValueVectorType>::mask_type::vector_type;
  const auto t_mask=reinterpret_cast<mask_t>(true_value.vec());
  const auto f_mask=reinterpret_cast<mask_t>(false_value.vec());
#if defined __clang__
  const auto result=(t_mask&condition.vec())|(f_mask&~condition.vec());
#else
  const auto result=condition.vec() ? t_mask : f_mask;
#endif
  using vector_t = typename Simd<ValueVectorType>::vector_type;
  return Simd{reinterpret_cast<vector_t>(result)};
}

template<typename VectorType>
inline constexpr
auto
min(Simd<VectorType> a,
    Simd<VectorType> b)
{
#if defined __clang__
  return select(a<b, a , b);
#else
  return Simd{a.vec()<b.vec() ? a.vec() : b.vec()};
#endif
}

template<typename VectorType>
inline constexpr
auto
min(Simd<VectorType> a,
    typename Simd<VectorType>::value_type b)
{
  return min(a, Simd<VectorType>{b});
}

template<typename VectorType>
inline constexpr
auto
min(typename Simd<VectorType>::value_type a,
    Simd<VectorType> b)
{
  return min(Simd<VectorType>{a}, b);
}

template<typename VectorType>
inline constexpr
auto
max(Simd<VectorType> a,
    Simd<VectorType> b)
{
#if defined __clang__
  return select(a>b, a , b);
#else
  return Simd{a.vec()>b.vec() ? a.vec() : b.vec()};
#endif
}

template<typename VectorType>
inline constexpr
auto
max(Simd<VectorType> a,
    typename Simd<VectorType>::value_type b)
{
  return max(a, Simd<VectorType>{b});
}

template<typename VectorType>
inline constexpr
auto
max(typename Simd<VectorType>::value_type a,
    Simd<VectorType> b)
{
  return max(Simd<VectorType>{a}, b);
}

//~~~~ explicit initialisation ~~~~

#define DIM_SIMD_X64(x) \
        x##0,  x##1,  x##2,  x##3,  x##4,  x##5,  x##6,  x##7,  \
        x##8,  x##9,  x##10, x##11, x##12, x##13, x##14, x##15, \
        x##16, x##17, x##18, x##19, x##20, x##21, x##22, x##23, \
        x##24, x##25, x##26, x##27, x##28, x##29, x##30, x##31, \
        x##32, x##33, x##34, x##35, x##36, x##37, x##38, x##39, \
        x##40, x##41, x##42, x##43, x##44, x##45, x##46, x##47, \
        x##48, x##49, x##50, x##51, x##52, x##53, x##54, x##55, \
        x##56, x##57, x##58, x##59, x##60, x##61, x##62, x##63
#define DIM_SIMD_X32(x) \
        x##0,  x##1,  x##2,  x##3,  x##4,  x##5,  x##6,  x##7,  \
        x##8,  x##9,  x##10, x##11, x##12, x##13, x##14, x##15, \
        x##16, x##17, x##18, x##19, x##20, x##21, x##22, x##23, \
        x##24, x##25, x##26, x##27, x##28, x##29, x##30, x##31
#define DIM_SIMD_X16(x) \
        x##0,  x##1,  x##2,  x##3,  x##4,  x##5,  x##6,  x##7,  \
        x##8,  x##9,  x##10, x##11, x##12, x##13, x##14, x##15
#define DIM_SIMD_X8(x) \
        x##0,  x##1,  x##2,  x##3,  x##4,  x##5,  x##6,  x##7
#define DIM_SIMD_X4(x) \
        x##0,  x##1,  x##2,  x##3
#define DIM_SIMD_X2(x) \
        x##0,  x##1

#define DIM_SIMD_MAKE(count) \
        template<typename SimdType, \
                 typename =expect_value_count<SimdType, count>> \
        inline constexpr \
        auto \
        make(DIM_SIMD_X##count(typename SimdType::value_type arg)) \
        { \
          using vector_t = typename SimdType::vector_type; \
          return Simd{vector_t{DIM_SIMD_X##count(arg)}}; \
        }
DIM_SIMD_MAKE(64)
DIM_SIMD_MAKE(32)
DIM_SIMD_MAKE(16)
DIM_SIMD_MAKE(8)
DIM_SIMD_MAKE(4)
DIM_SIMD_MAKE(2)
#undef DIM_SIMD_MAKE

//~~~~ shuffle ~~~~

#if defined __clang__
# define DIM_SIMD_BUILTIN_SHUFFLE_1(mask) \
         __builtin_shufflevector(s.vec(), s.vec(), mask)
# define DIM_SIMD_BUILTIN_SHUFFLE_2(mask) \
         __builtin_shufflevector(low.vec(), high.vec(), mask)
#else // GCC
# define DIM_SIMD_BUILTIN_SHUFFLE_1(mask) \
         __builtin_shuffle(s.vec(), \
                           typename SimdType::mask_type::vector_type{mask})
# define DIM_SIMD_BUILTIN_SHUFFLE_2(mask) \
         __builtin_shuffle(low.vec(), high.vec(), \
                           typename SimdType::mask_type::vector_type{mask})
#endif

using idx_t = std::uint8_t;

#define DIM_SIMD_SHUFFLE(count) \
        template<DIM_SIMD_X##count(idx_t N), \
                 typename SimdType, \
                 typename =expect_value_count<SimdType, count>> \
        inline constexpr \
        auto \
        shuffle(SimdType s) \
        { \
          return Simd{DIM_SIMD_BUILTIN_SHUFFLE_1(DIM_SIMD_X##count(N))}; \
        } \
        template<DIM_SIMD_X##count(idx_t N), \
                 typename SimdType, \
                 typename =expect_value_count<SimdType, count>> \
        inline constexpr \
        auto \
        shuffle(SimdType low, \
                SimdType high) \
        { \
          return Simd{DIM_SIMD_BUILTIN_SHUFFLE_2(DIM_SIMD_X##count(N))}; \
        }
DIM_SIMD_SHUFFLE(64)
DIM_SIMD_SHUFFLE(32)
DIM_SIMD_SHUFFLE(16)
DIM_SIMD_SHUFFLE(8)
DIM_SIMD_SHUFFLE(4)
DIM_SIMD_SHUFFLE(2)
#undef DIM_SIMD_SHUFFLE
#undef DIM_SIMD_BUILTIN_SHUFFLE_1
#undef DIM_SIMD_BUILTIN_SHUFFLE_2

#undef DIM_SIMD_X64
#undef DIM_SIMD_X32
#undef DIM_SIMD_X16
#undef DIM_SIMD_X8
#undef DIM_SIMD_X4
#undef DIM_SIMD_X2

template<idx_t N,
         typename VectorType>
inline constexpr
auto
down(Simd<VectorType> s)
{
  constexpr auto c=s.value_count;
  if constexpr(c==64)
  {
    return shuffle<( 0+N)%c, ( 1+N)%c, ( 2+N)%c, ( 3+N)%c,
                   ( 4+N)%c, ( 5+N)%c, ( 6+N)%c, ( 7+N)%c,
                   ( 8+N)%c, ( 9+N)%c, (10+N)%c, (11+N)%c,
                   (12+N)%c, (13+N)%c, (14+N)%c, (15+N)%c,
                   (16+N)%c, (17+N)%c, (18+N)%c, (19+N)%c,
                   (20+N)%c, (21+N)%c, (22+N)%c, (23+N)%c,
                   (24+N)%c, (25+N)%c, (26+N)%c, (27+N)%c,
                   (28+N)%c, (29+N)%c, (30+N)%c, (31+N)%c,
                   (32+N)%c, (33+N)%c, (34+N)%c, (35+N)%c,
                   (36+N)%c, (37+N)%c, (38+N)%c, (39+N)%c,
                   (40+N)%c, (41+N)%c, (42+N)%c, (43+N)%c,
                   (44+N)%c, (45+N)%c, (46+N)%c, (47+N)%c,
                   (48+N)%c, (49+N)%c, (50+N)%c, (51+N)%c,
                   (52+N)%c, (53+N)%c, (54+N)%c, (55+N)%c,
                   (56+N)%c, (57+N)%c, (58+N)%c, (59+N)%c,
                   (60+N)%c, (61+N)%c, (62+N)%c, (63+N)%c>(s);
  }
  else if constexpr(c==32)
  {
    return shuffle<( 0+N)%c, ( 1+N)%c, ( 2+N)%c, ( 3+N)%c,
                   ( 4+N)%c, ( 5+N)%c, ( 6+N)%c, ( 7+N)%c,
                   ( 8+N)%c, ( 9+N)%c, (10+N)%c, (11+N)%c,
                   (12+N)%c, (13+N)%c, (14+N)%c, (15+N)%c,
                   (16+N)%c, (17+N)%c, (18+N)%c, (19+N)%c,
                   (20+N)%c, (21+N)%c, (22+N)%c, (23+N)%c,
                   (24+N)%c, (25+N)%c, (26+N)%c, (27+N)%c,
                   (28+N)%c, (29+N)%c, (30+N)%c, (31+N)%c>(s);
  }
  else if constexpr(c==16)
  {
    return shuffle<( 0+N)%c, ( 1+N)%c, ( 2+N)%c, ( 3+N)%c,
                   ( 4+N)%c, ( 5+N)%c, ( 6+N)%c, ( 7+N)%c,
                   ( 8+N)%c, ( 9+N)%c, (10+N)%c, (11+N)%c,
                   (12+N)%c, (13+N)%c, (14+N)%c, (15+N)%c>(s);
  }
  else if constexpr(c==8)
  {
    return shuffle<( 0+N)%c, ( 1+N)%c, ( 2+N)%c, ( 3+N)%c,
                   ( 4+N)%c, ( 5+N)%c, ( 6+N)%c, ( 7+N)%c>(s);
  }
  else if constexpr(c==4)
  {
    return shuffle<( 0+N)%c, ( 1+N)%c, ( 2+N)%c, ( 3+N)%c>(s);
  }
  else if constexpr(c==2)
  {
    return shuffle<( 0+N)%c, ( 1+N)%c>(s);
  }
  else
  {
    return s;
  }
}

template<idx_t N,
         typename VectorType>
inline constexpr
auto
up(Simd<VectorType> s)
{
  return down<s.value_count-N>(s);
}

template<typename VectorType>
inline constexpr
auto
even(Simd<VectorType> low,
     Simd<VectorType> high)
{
  constexpr auto c=low.value_count;
  if constexpr(c==64)
  {
    return shuffle<  0,   2,   4,   6,   8,  10,  12,  14,
                    16,  18,  20,  22,  24,  26,  28,  30,
                    32,  34,  36,  38,  40,  42,  44,  46,
                    48,  50,  52,  54,  56,  58,  60,  62,
                    64,  66,  68,  70,  72,  74,  76,  78,
                    80,  82,  84,  86,  88,  90,  92,  94,
                    96,  98, 100, 102, 104, 106, 108, 110,
                   112, 114, 116, 118, 120, 122, 124, 126>(low, high);
  }
  else if constexpr(c==32)
  {
    return shuffle<  0,   2,   4,   6,   8,  10,  12,  14,
                    16,  18,  20,  22,  24,  26,  28,  30,
                    32,  34,  36,  38,  40,  42,  44,  46,
                    48,  50,  52,  54,  56,  58,  60,  62>(low, high);
  }
  else if constexpr(c==16)
  {
    return shuffle<  0,   2,   4,   6,   8,  10,  12,  14,
                    16,  18,  20,  22,  24,  26,  28,  30>(low, high);
  }
  else if constexpr(c==8)
  {
    return shuffle<  0,   2,   4,   6,   8,  10,  12,  14>(low, high);
  }
  else if constexpr(c==4)
  {
    return shuffle<  0,   2,   4,   6>(low, high);
  }
  else if constexpr(c==2)
  {
    return shuffle<  0,   2>(low, high);
  }
  else
  {
    return low;
  }
}

template<typename VectorType>
inline constexpr
auto
odd(Simd<VectorType> low,
    Simd<VectorType> high)
{
  constexpr auto c=low.value_count;
  if constexpr(c==64)
  {
    return shuffle<  1,   3,   5,   7,   9,  11,  13,  15,
                    17,  19,  21,  23,  25,  27,  29,  31,
                    33,  35,  37,  39,  41,  43,  45,  47,
                    49,  51,  53,  55,  57,  59,  61,  63,
                    65,  67,  69,  71,  73,  75,  77,  79,
                    81,  83,  85,  87,  89,  91,  93,  95,
                    97,  99, 101, 103, 105, 107, 109, 111,
                   113, 115, 117, 119, 121, 123, 125, 127>(low, high);
  }
  else if constexpr(c==32)
  {
    return shuffle<  1,   3,   5,   7,   9,  11,  13,  15,
                    17,  19,  21,  23,  25,  27,  29,  31,
                    33,  35,  37,  39,  41,  43,  45,  47,
                    49,  51,  53,  55,  57,  59,  61,  63>(low, high);
  }
  else if constexpr(c==16)
  {
    return shuffle<  1,   3,   5,   7,   9,  11,  13,  15,
                    17,  19,  21,  23,  25,  27,  29,  31>(low, high);
  }
  else if constexpr(c==8)
  {
    return shuffle<  1,   3,   5,   7,   9,  11,  13,  15>(low, high);
  }
  else if constexpr(c==4)
  {
    return shuffle<  1,   3,   5,   7>(low, high);
  }
  else if constexpr(c==2)
  {
    return shuffle<  1,   3>(low, high);
  }
  else
  {
    return high;
  }
}

//~~~~ load/store ~~~~

template<typename VectorType>
inline
auto
load_u(const Simd<VectorType> *unaligned_addr)
{
  struct unaligned { VectorType v; } __attribute__((__packed__));
  return Simd{reinterpret_cast<const unaligned *>(unaligned_addr)->v};
}

template<typename SimdType>
inline
auto
load_u(const typename SimdType::value_type *unaligned_addr)
{
  return load_u(reinterpret_cast<const SimdType *>(unaligned_addr));
}

template<typename VectorType>
inline
auto
load_a(const Simd<VectorType> *aligned_addr)
{
  return *aligned_addr;
}

template<typename SimdType>
inline
auto
load_a(const typename SimdType::value_type *aligned_addr)
{
  return load_a(reinterpret_cast<const SimdType *>(aligned_addr));
}

template<typename VectorType>
inline
void
store_u(Simd<VectorType> *unaligned_addr,
        Simd<VectorType> s)
{
  struct unaligned { VectorType v; } __attribute__((__packed__));
  reinterpret_cast<unaligned *>(unaligned_addr)->v=s.vec();
}

template<typename VectorType>
inline
void
store_u(typename Simd<VectorType>::value_type *unaligned_addr,
        Simd<VectorType> s)
{
  store_u(reinterpret_cast<Simd<VectorType> *>(unaligned_addr), s);
}

template<typename VectorType>
inline
void
store_a(Simd<VectorType> *aligned_addr,
        Simd<VectorType> s)
{
  *aligned_addr=s;
}

template<typename VectorType>
inline
void
store_a(typename Simd<VectorType>::value_type *aligned_addr,
        Simd<VectorType> s)
{
  store_a(reinterpret_cast<Simd<VectorType> *>(aligned_addr), s);
}

template<typename SimdType>
inline
auto
split(const typename SimdType::value_type *values,
      int count)
{
  constexpr auto vector_size=SimdType::vector_size;
  constexpr auto value_size=SimdType::value_size;
  constexpr auto value_count=SimdType::value_count;
  const auto offset=int(reinterpret_cast<std::intptr_t>(values)%vector_size);
  const auto prefix=offset ? (vector_size-offset)/value_size : 0;
  const auto simd_count=(count-prefix)/value_count;
  const auto suffix=(count-prefix)%value_count;
  return std::make_tuple(prefix, simd_count, suffix);
}

template<typename SimdType>
inline
auto
load_prefix(const typename SimdType::value_type *values,
            int prefix_length)
{
  using vector_t = typename SimdType::vector_type;
  auto result=vector_t{};
  const auto offset=SimdType::value_count-prefix_length;
  for(auto i=0; i<prefix_length; ++i)
  {
    result[i+offset]=values[i];
  }
  return Simd{result};
}

template<typename VectorType>
inline
void
store_prefix(typename Simd<VectorType>::value_type *values,
             int prefix_length,
             Simd<VectorType> s)
{
  const auto offset=Simd<VectorType>::value_count-prefix_length;
  for(auto i=0; i<prefix_length; ++i)
  {
    values[i]=s[i+offset];
  }
}

template<typename SimdType>
inline
auto
load_suffix(const typename SimdType::value_type *values,
            int suffix_length)
{
  using vector_t = typename SimdType::vector_type;
  auto result=vector_t{};
  for(auto i=0; i<suffix_length; ++i)
  {
    result[i]=values[i];
  }
  return Simd{result};
}

template<typename VectorType>
inline
void
store_suffix(typename Simd<VectorType>::value_type *values,
             int suffix_length,
             Simd<VectorType> s)
{
  for(auto i=0; i<suffix_length; ++i)
  {
    values[i]=s[i];
  }
}

//~~~~ math functions ~~~~

template<typename VectorType,
         typename Fnct>
auto
transform(Simd<VectorType> s,
          Fnct fnct)
{
  // hopefuly, for simple math functions, the compiler will be able
  // to call the appropriate simd instruction; in other cases it
  // will serialise the calls
  VectorType result;
  for(auto i=0; i<s.value_count; ++i)
  {
    result[i]=fnct(s[i]);
  }
  return Simd{result};
}

#define DIM_SIMD_TRANSFORM(name, fnct) \
        template<typename VectorType> \
        auto \
        name(Simd<VectorType> s) \
        { \
          using value_t = typename Simd<VectorType>::value_type; \
          return transform(s, static_cast<value_t (*)(value_t)>(fnct)); \
        }

DIM_SIMD_TRANSFORM(abs,   std::abs)
DIM_SIMD_TRANSFORM(exp,   std::exp)
DIM_SIMD_TRANSFORM(log,   std::log)
DIM_SIMD_TRANSFORM(sqrt,  std::sqrt)
DIM_SIMD_TRANSFORM(cbrt,  std::cbrt)
DIM_SIMD_TRANSFORM(sin,   std::sin)
DIM_SIMD_TRANSFORM(cos,   std::cos)
DIM_SIMD_TRANSFORM(tan,   std::tan)
DIM_SIMD_TRANSFORM(asin,  std::asin)
DIM_SIMD_TRANSFORM(acos,  std::acos)
DIM_SIMD_TRANSFORM(atan,  std::atan)
DIM_SIMD_TRANSFORM(sinh,  std::sinh)
DIM_SIMD_TRANSFORM(cosh,  std::cosh)
DIM_SIMD_TRANSFORM(tanh,  std::tanh)
DIM_SIMD_TRANSFORM(ceil,  std::sinh)
DIM_SIMD_TRANSFORM(floor, std::floor)
DIM_SIMD_TRANSFORM(trunc, std::trunc)
DIM_SIMD_TRANSFORM(round, std::round)

#undef DIM_SIMD_TRANSFORM

//~~~~ display operations ~~~~

template<typename VectorType>
inline
std::string
to_string(const Simd<VectorType> &s)
{
  auto result=std::string{'{'};
  for(auto i=0; i<s.value_count; ++i)
  {
    if(i!=0)
    {
      result+=", ";
    }
    result+=std::to_string(s[i]);
  }
  result+='}';
  return result;;
}

template<typename VectorType>
inline
std::ostream &
operator<<(std::ostream &os,
           const Simd<VectorType> &s)
{
  os << '{';
  for(auto i=0; i<s.value_count; ++i)
  {
    if(i!=0)
    {
      os << ", ";
    }
    os << s[i];
  }
  return os << '}';
}

} // namespace dim::simd

#endif // DIM_SIMD_HPP

//----------------------------------------------------------------------------
