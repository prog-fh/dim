//----------------------------------------------------------------------------

#ifndef DIM_SIMD_HPP
#define DIM_SIMD_HPP 1

#include <immintrin.h>
#include <cstdint>
#include <type_traits>
#include <tuple>
#include <string>
#include <iostream>

namespace dim::simd {

template<typename ValueType,
         int VectorSize>
struct vector_helper { using type = void; };

template<typename ValueType,
         int VectorSize>
using vector_t = typename vector_helper<
  typename std::decay<ValueType>::type, VectorSize>::type;

template<typename T>
class Simd
{
public:

  using vector_type = T;
  using value_type =
    typename std::decay<decltype(std::declval<vector_type>()[0])>::type;

  static constexpr auto vector_size =
    int(sizeof(vector_type));
  static constexpr auto value_size =
    int(sizeof(value_type));
  static constexpr auto value_count =
    vector_size/value_size;
  static constexpr auto is_integral =
    std::is_integral<value_type>::value;
  static constexpr auto is_floating_point =
    std::is_floating_point<value_type>::value;

  using mask_type = vector_t<
    typename std::conditional<value_size==1, std::int8_t,
    typename std::conditional<value_size==2, std::int16_t,
    typename std::conditional<value_size==4, std::int32_t,
    typename std::conditional<value_size==8, std::int64_t,
    void>::type>::type>::type>::type, vector_size>;

  constexpr Simd() : v_{} {};
  constexpr Simd(vector_type v) : v_{v} {};
  constexpr Simd & operator=(vector_type v) { v_=v; return *this; }
  constexpr Simd(value_type v)
  { // skip member-initialiser-list
    constexpr auto c=value_count;
    if constexpr (c==64)
    {
      v_=vector_type{v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v,
                     v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v,
                     v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v,
                     v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v};
    }
    else if constexpr (c==32)
    {
      v_=vector_type{v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v,
                     v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v};
    }
    else if constexpr (c==16)
    {
      v_=vector_type{v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v};
    }
    else if constexpr (c==8)
    {
      v_=vector_type{v, v, v, v, v, v, v, v};
    }
    else if constexpr (c==4)
    {
      v_=vector_type{v, v, v, v};
    }
    else if constexpr (c==2)
    {
      v_=vector_type{v, v};
    }
    else
    {
      v_=vector_type{v};
    }
  }
  constexpr Simd & operator=(value_type v) { v_=Simd{v}; return *this; }

  constexpr Simd(const Simd &) =default;
  constexpr Simd & operator=(const Simd &) =default;
  constexpr Simd(Simd &&) =default;
  constexpr Simd & operator=(Simd &&) =default;
  ~Simd() =default;

  constexpr operator    vector_type() const { return v_; }
  constexpr const vector_type & vec() const { return v_; }
  constexpr       vector_type & vec()       { return v_; }

  constexpr value_type   operator[](int i) const { return v_[i]; }
  constexpr value_type & operator[](int i)       { return v_[i]; }

private:
  vector_type v_;
};

template<typename T,
         typename ValueType>
using expect_value_type =
  typename std::enable_if<
    std::is_same<
      typename T::value_type,
      typename std::decay<ValueType>::type
      >::value
    >::type;

template<typename T,
         int VectorSize>
using expect_vector_size =
  typename std::enable_if<
    std::is_same<
      typename std::integral_constant<int, T::vector_size>,
      typename std::integral_constant<int, VectorSize>
      >::value
    >::type;

template<typename T,
         int ValueSize>
using expect_value_size =
  typename std::enable_if<
    std::is_same<
      typename std::integral_constant<int, T::value_size>,
      typename std::integral_constant<int, ValueSize>
      >::value
    >::type;

template<typename T,
         int ValueCount>
using expect_value_count =
  typename std::enable_if<
    std::is_same<
      typename std::integral_constant<int, T::value_count>,
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

#define DIM_SIMD_DEFINE_TYPE(vec, base, vec_size) \
  using vec = Simd<base __attribute__((__vector_size__(vec_size)))>; \
  template<> struct vector_helper<base, vec_size> { using type = vec; };

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

using u8_t  = vector_t< std::uint8_t, max_vector_size>;
using i8_t  = vector_t<  std::int8_t, max_vector_size>;
using u16_t = vector_t<std::uint16_t, max_vector_size>;
using i16_t = vector_t< std::int16_t, max_vector_size>;
using u32_t = vector_t<std::uint32_t, max_vector_size>;
using i32_t = vector_t< std::int32_t, max_vector_size>;
using u64_t = vector_t<std::uint64_t, max_vector_size>;
using i64_t = vector_t< std::int64_t, max_vector_size>;
using r32_t = vector_t<        float, max_vector_size>;
using r64_t = vector_t<       double, max_vector_size>;

//~~~~ arithmetic and logic operators ~~~~

#define DIM_SIMD_FORWARD_UNARY(op) \
        template<typename T> \
        inline constexpr \
        auto \
        operator op(Simd<T> rhs) \
        { \
          return Simd{op rhs.vec()}; \
        }
DIM_SIMD_FORWARD_UNARY(+)
DIM_SIMD_FORWARD_UNARY(-)
DIM_SIMD_FORWARD_UNARY(~)
#undef DIM_SIMD_FORWARD_UNARY
#define DIM_SIMD_FORWARD_BINARY(op) \
        template<typename T1, \
                 typename T2> \
        inline constexpr \
        auto \
        operator op(Simd<T1> lhs, \
                    Simd<T2> rhs) \
        { \
          return Simd{lhs.vec() op rhs.vec()}; \
        } \
        template<typename T> \
        inline constexpr \
        auto \
        operator op(Simd<T> lhs, \
                    typename Simd<T>::value_type rhs) \
        { \
          return Simd{lhs.vec() op rhs}; \
        } \
        template<typename T> \
        inline constexpr \
        auto \
        operator op(typename Simd<T>::value_type lhs, \
                    Simd<T> rhs) \
        { \
          return Simd{lhs op rhs.vec()}; \
        }
#define DIM_SIMD_FORWARD_BINARY_ASSIGN(op) \
        DIM_SIMD_FORWARD_BINARY(op) \
        template<typename T1, \
                 typename T2> \
        inline constexpr \
        auto & \
        operator op##=(Simd<T1> &lhs, \
                       Simd<T2> rhs) \
        { \
          lhs.vec() op##= rhs.vec(); \
          return lhs; \
        } \
        template<typename T> \
        inline constexpr \
        auto & \
        operator op##=(Simd<T> &lhs, \
                       typename Simd<T>::value_type rhs) \
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

template<typename T1,
         typename T2>
inline constexpr
auto
select(Simd<T1> condition,
       Simd<T2> true_value,
       Simd<T2> false_value)
{
  using mask_t=typename Simd<T2>::mask_type::vector_type;
  const auto t_mask=reinterpret_cast<mask_t>(true_value.vec());
  const auto f_mask=reinterpret_cast<mask_t>(false_value.vec());
  const auto result=(t_mask&condition.vec())|(f_mask&~condition.vec());
  return Simd{reinterpret_cast<typename Simd<T2>::vector_type>(result)};
}

template<typename T>
inline constexpr
auto
min(Simd<T> a,
    Simd<T> b)
{
  return select(a<b, a , b);
}

template<typename T>
inline constexpr
auto
max(Simd<T> a,
    Simd<T> b)
{
  return select(a>b, a , b);
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
        template<typename T, \
                 typename =expect_value_count<T, count>> \
        inline constexpr \
        auto \
        make(DIM_SIMD_X##count(typename T::value_type arg)) \
        { \
          return Simd{typename T::vector_type{DIM_SIMD_X##count(arg)}}; \
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
         __builtin_shufflevector(v.vec(), v.vec(), mask)
# define DIM_SIMD_BUILTIN_SHUFFLE_2(mask) \
         __builtin_shufflevector(low.vec(), high.vec(), mask)
#else // GCC
# define DIM_SIMD_BUILTIN_SHUFFLE_1(mask) \
         __builtin_shuffle(v.vec(), \
                           typename T::mask_type::vector_type{mask})
# define DIM_SIMD_BUILTIN_SHUFFLE_2(mask) \
         __builtin_shuffle(low.vec(), high.vec(), \
                           typename T::mask_type::vector_type{mask})
#endif

using idx_t = std::uint8_t;

#define DIM_SIMD_SHUFFLE(count) \
        template<DIM_SIMD_X##count(idx_t N), \
                 typename T, \
                 typename =expect_value_count<T, count>> \
        inline constexpr \
        auto \
        shuffle(T v) \
        { \
          return Simd{DIM_SIMD_BUILTIN_SHUFFLE_1(DIM_SIMD_X##count(N))}; \
        } \
        template<DIM_SIMD_X##count(idx_t N), \
                 typename T, \
                 typename =expect_value_count<T, count>> \
        inline constexpr \
        auto \
        shuffle(T low, \
                T high) \
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
         typename T>
inline constexpr
auto
down(Simd<T> v)
{
  constexpr auto c=v.value_count;
  if constexpr (c==64)
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
                   (60+N)%c, (61+N)%c, (62+N)%c, (63+N)%c>(v);
  }
  else if constexpr (c==32)
  {
    return shuffle<( 0+N)%c, ( 1+N)%c, ( 2+N)%c, ( 3+N)%c,
                   ( 4+N)%c, ( 5+N)%c, ( 6+N)%c, ( 7+N)%c,
                   ( 8+N)%c, ( 9+N)%c, (10+N)%c, (11+N)%c,
                   (12+N)%c, (13+N)%c, (14+N)%c, (15+N)%c,
                   (16+N)%c, (17+N)%c, (18+N)%c, (19+N)%c,
                   (20+N)%c, (21+N)%c, (22+N)%c, (23+N)%c,
                   (24+N)%c, (25+N)%c, (26+N)%c, (27+N)%c,
                   (28+N)%c, (29+N)%c, (30+N)%c, (31+N)%c>(v);
  }
  else if constexpr (c==16)
  {
    return shuffle<( 0+N)%c, ( 1+N)%c, ( 2+N)%c, ( 3+N)%c,
                   ( 4+N)%c, ( 5+N)%c, ( 6+N)%c, ( 7+N)%c,
                   ( 8+N)%c, ( 9+N)%c, (10+N)%c, (11+N)%c,
                   (12+N)%c, (13+N)%c, (14+N)%c, (15+N)%c>(v);
  }
  else if constexpr (c==8)
  {
    return shuffle<(0+N)%c, (1+N)%c, (2+N)%c, (3+N)%c,
                   (4+N)%c, (5+N)%c, (6+N)%c, (7+N)%c>(v);
  }
  else if constexpr (c==4)
  {
    return shuffle<(0+N)%c, (1+N)%c, (2+N)%c, (3+N)%c>(v);
  }
  else if constexpr (c==2)
  {
    return shuffle<(0+N)%c, (1+N)%c>(v);
  }
  else
  {
    return v;
  }
}

template<idx_t N,
         typename T>
inline constexpr
auto
up(Simd<T> v)
{
  constexpr auto c=v.value_count;
  return down<c-N>(v);
}

template<typename T>
inline constexpr
auto
even(Simd<T> low,
     Simd<T> high)
{
  constexpr auto c=low.value_count;
  if constexpr (c==64)
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
  else if constexpr (c==32)
  {
    return shuffle<  0,   2,   4,   6,   8,  10,  12,  14,
                    16,  18,  20,  22,  24,  26,  28,  30,
                    32,  34,  36,  38,  40,  42,  44,  46,
                    48,  50,  52,  54,  56,  58,  60,  62>(low, high);
  }
  else if constexpr (c==16)
  {
    return shuffle<  0,   2,   4,   6,   8,  10,  12,  14,
                    16,  18,  20,  22,  24,  26,  28,  30>(low, high);
  }
  else if constexpr (c==8)
  {
    return shuffle<  0,   2,   4,   6,   8,  10,  12,  14>(low, high);
  }
  else if constexpr (c==4)
  {
    return shuffle<  0,   2,   4,   6>(low, high);
  }
  else if constexpr (c==2)
  {
    return shuffle<  0,   2>(low, high);
  }
  else
  {
    return low;
  }
}

template<typename T>
inline constexpr
auto
odd(Simd<T> low,
    Simd<T> high)
{
  constexpr auto c=low.value_count;
  if constexpr (c==64)
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
  else if constexpr (c==32)
  {
    return shuffle<  1,   3,   5,   7,   9,  11,  13,  15,
                    17,  19,  21,  23,  25,  27,  29,  31,
                    33,  35,  37,  39,  41,  43,  45,  47,
                    49,  51,  53,  55,  57,  59,  61,  63>(low, high);
  }
  else if constexpr (c==16)
  {
    return shuffle<  1,   3,   5,   7,   9,  11,  13,  15,
                    17,  19,  21,  23,  25,  27,  29,  31>(low, high);
  }
  else if constexpr (c==8)
  {
    return shuffle<  1,   3,   5,   7,   9,  11,  13,  15>(low, high);
  }
  else if constexpr (c==4)
  {
    return shuffle<  1,   3,   5,   7>(low, high);
  }
  else if constexpr (c==2)
  {
    return shuffle<  1,   3>(low, high);
  }
  else
  {
    return high;
  }
}

//~~~~ load/store ~~~~

template<typename T>
inline
auto
load_u(const Simd<T> *unaligned_addr)
{
  using vec_t = typename Simd<T>::vector_type;
  struct unaligned { vec_t v; } __attribute__((__packed__));
  return Simd{reinterpret_cast<const unaligned *>(unaligned_addr)->v};
}

template<typename T>
inline
auto
load_a(const Simd<T> *aligned_addr)
{
  return *aligned_addr;
}

template<typename T>
inline
void
store_u(Simd<T> *unaligned_addr,
        Simd<T> v)
{
  using vec_t = typename Simd<T>::vector_type;
  struct unaligned { vec_t v; } __attribute__((__packed__));
  reinterpret_cast<unaligned *>(unaligned_addr)->v=v.vec();
}

template<typename T>
inline
void
store_a(Simd<T> *aligned_addr,
        Simd<T> v)
{
  *aligned_addr=v;
}

template<typename T>
inline
auto
split(const typename T::value_type *values,
      int count)
{
  constexpr auto vector_size=T::vector_size;
  constexpr auto value_size=T::value_size;
  constexpr auto value_count=T::value_count;
  const auto offset=int(reinterpret_cast<std::intptr_t>(values)%vector_size);
  const auto prefix=offset ? (vector_size-offset)/value_size : 0;
  const auto vector_count=(count-prefix)/value_count;
  const auto suffix=(count-prefix)%value_count;
  return std::make_tuple(prefix, vector_count, suffix);
}

template<typename T>
inline
auto
load_prefix(const typename T::value_type *values,
            int prefix_length)
{
  auto v=T{};
  const auto offset=v.value_count-prefix_length;
  for(auto i=0; i<prefix_length; ++i)
  {
    v[i+offset]=values[i];
  }
  return v;
}

template<typename T>
inline
void
store_prefix(typename T::value_type *values,
             int prefix_length,
             Simd<T> v)
{
  const auto offset=v.value_count-prefix_length;
  for(auto i=0; i<prefix_length; ++i)
  {
    values[i]=v[i+offset];
  }
}

template<typename T>
inline
auto
load_suffix(const typename T::value_type *values,
            int suffix_length)
{
  auto v=T{};
  for(auto i=0; i<suffix_length; ++i)
  {
    v[i]=values[i];
  }
  return v;
}

template<typename T>
inline
void
store_suffix(typename T::value_type *values,
             int suffix_length,
             Simd<T> v)
{
  for(auto i=0; i<suffix_length; ++i)
  {
    values[i]=v[i];
  }
}

//~~~~ display operations ~~~~

template<typename T>
inline
std::string
to_string(const Simd<T> &v)
{
  auto result=std::string{'{'};
  for(auto i=0; i<v.value_count; ++i)
  {
    if(i!=0)
    {
      result+=", ";
    }
    result+=std::to_string(v[i]);
  }
  result+='}';
  return result;;
}

template<typename T>
inline
std::ostream &
operator<<(std::ostream &os,
           const Simd<T> &v)
{
  os << '{';
  for(auto i=0; i<v.value_count; ++i)
  {
    if(i!=0)
    {
      os << ", ";
    }
    os << v[i];
  }
  return os << '}';
}

} // namespace dim::simd

#endif // DIM_SIMD_HPP

//----------------------------------------------------------------------------
