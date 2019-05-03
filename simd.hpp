//----------------------------------------------------------------------------

#ifndef DIM_SIMD_HPP
#define DIM_SIMD_HPP 1

#include <immintrin.h>
#include <stdint.h>
#include <type_traits>
#include <string>
#include <iostream>

namespace dim::simd {

//~~~~ type traits ~~~~

template<typename T>
struct is_simd_helper : std::false_type { };
template<typename T>
struct is_simd : is_simd_helper<typename std::decay<T>::type> { };
template<typename T>
constexpr int is_simd_v = is_simd<T>::value;

template<typename T>
using expect_simd = typename std::enable_if<is_simd_v<T>>::type;

template<typename T,
         typename =expect_simd<T>>
struct shuffle_mask_helper { using type = void; };
template<typename T>
using shuffle_mask_t = typename shuffle_mask_helper<T>::type;

template<typename T,
         typename =expect_simd<T>>
using value_type_t = typename std::decay<decltype((*(T *)0)[0])>::type;

template<typename T,
         typename =expect_simd<T>>
constexpr auto vector_size_v = int(sizeof(T));

template<typename T>
constexpr auto value_size_v = int(sizeof(value_type_t<T>));

template<typename T>
constexpr auto value_count_v = vector_size_v<T>/value_size_v<T>;

template<typename T,
         int VectorSize>
using expect_vector_size =
  typename std::enable_if<
    std::is_same<
      typename std::integral_constant<int, vector_size_v<T>>,
      typename std::integral_constant<int, VectorSize>
      >::value
    >::type;

template<typename T,
         int ValueSize>
using expect_value_size =
  typename std::enable_if<
    std::is_same<
      typename std::integral_constant<int, value_size_v<T>>,
      typename std::integral_constant<int, ValueSize>
      >::value
    >::type;

template<typename T,
         int ValueCount>
using expect_value_count =
  typename std::enable_if<
    std::is_same<
      typename std::integral_constant<int, value_count_v<T>>,
      typename std::integral_constant<int, ValueCount>
      >::value
    >::type;

//~~~~ enumerate available simd types ~~~~

#if defined __clang__
# define DIM_SIMD_DEFINE_TYPE(tn, bn, vs, smn) \
         using tn = bn __attribute__((ext_vector_type(vs/sizeof(bn)))); \
         template<> struct is_simd_helper<tn> : std::true_type { }; \
         template<> struct shuffle_mask_helper<tn> { using type = smn; };
#elif defined __GNUC__
# define DIM_SIMD_DEFINE_TYPE(tn, bn, vs, smn) \
         using tn = bn __attribute__((__vector_size__(vs))); \
         template<> struct is_simd_helper<tn> : std::true_type { }; \
         template<> struct shuffle_mask_helper<tn> { using type = smn; };
#else
# error "SIMD vector types not supported by compiler"
#endif

#if __AVX512F__
# define DIM_SIMD_MAX_REGISTER_SIZE 64
#elif __AVX__
# define DIM_SIMD_MAX_REGISTER_SIZE 32
#else
# define DIM_SIMD_MAX_REGISTER_SIZE 16
#endif

#if DIM_SIMD_MAX_REGISTER_SIZE>=64
  DIM_SIMD_DEFINE_TYPE(  u8x64_t,  uint8_t, 64,  u8x64_t)
  DIM_SIMD_DEFINE_TYPE(  i8x64_t,   int8_t, 64,  u8x64_t)
  DIM_SIMD_DEFINE_TYPE( u16x32_t, uint16_t, 64, u16x32_t)
  DIM_SIMD_DEFINE_TYPE( i16x32_t,  int16_t, 64, u16x32_t)
  DIM_SIMD_DEFINE_TYPE( u32x16_t, uint32_t, 64, u32x16_t)
  DIM_SIMD_DEFINE_TYPE( i32x16_t,  int32_t, 64, u32x16_t)
  DIM_SIMD_DEFINE_TYPE(  u64x8_t, uint64_t, 64,  u64x8_t)
  DIM_SIMD_DEFINE_TYPE(  i64x8_t,  int64_t, 64,  u64x8_t)
  DIM_SIMD_DEFINE_TYPE( r32x16_t,    float, 64, u32x16_t)
  DIM_SIMD_DEFINE_TYPE(  r64x8_t,   double, 64,  u64x8_t)
#endif

#if DIM_SIMD_MAX_REGISTER_SIZE>=32
  DIM_SIMD_DEFINE_TYPE( u8x32_t,  uint8_t, 32,  u8x32_t)
  DIM_SIMD_DEFINE_TYPE( i8x32_t,   int8_t, 32,  u8x32_t)
  DIM_SIMD_DEFINE_TYPE(u16x16_t, uint16_t, 32, u16x16_t)
  DIM_SIMD_DEFINE_TYPE(i16x16_t,  int16_t, 32, u16x16_t)
  DIM_SIMD_DEFINE_TYPE( u32x8_t, uint32_t, 32,  u32x8_t)
  DIM_SIMD_DEFINE_TYPE( i32x8_t,  int32_t, 32,  u32x8_t)
  DIM_SIMD_DEFINE_TYPE( u64x4_t, uint64_t, 32,  u64x4_t)
  DIM_SIMD_DEFINE_TYPE( i64x4_t,  int64_t, 32,  u64x4_t)
  DIM_SIMD_DEFINE_TYPE( r32x8_t,    float, 32,  u32x8_t)
  DIM_SIMD_DEFINE_TYPE( r64x4_t,   double, 32,  u64x4_t)
#endif

#if DIM_SIMD_MAX_REGISTER_SIZE>=16
  DIM_SIMD_DEFINE_TYPE(u8x16_t,  uint8_t, 16, u8x16_t)
  DIM_SIMD_DEFINE_TYPE(i8x16_t,   int8_t, 16, u8x16_t)
  DIM_SIMD_DEFINE_TYPE(u16x8_t, uint16_t, 16, u16x8_t)
  DIM_SIMD_DEFINE_TYPE(i16x8_t,  int16_t, 16, u16x8_t)
  DIM_SIMD_DEFINE_TYPE(u32x4_t, uint32_t, 16, u32x4_t)
  DIM_SIMD_DEFINE_TYPE(i32x4_t,  int32_t, 16, u32x4_t)
  DIM_SIMD_DEFINE_TYPE(u64x2_t, uint64_t, 16, u64x2_t)
  DIM_SIMD_DEFINE_TYPE(i64x2_t,  int64_t, 16, u64x2_t)
  DIM_SIMD_DEFINE_TYPE(r32x4_t,    float, 16, u32x4_t)
  DIM_SIMD_DEFINE_TYPE(r64x2_t,   double, 16, u64x2_t)
#endif

#undef DIM_SIMD_DEFINE_TYPE

//~~~~ define simd types with maximal register size ~~~~

#if DIM_SIMD_MAX_REGISTER_SIZE==64
  using i8_t  = i8x64_t;
  using u8_t  = u8x64_t;
  using i16_t = i16x32_t;
  using u16_t = u16x32_t;
  using i32_t = i32x16_t;
  using u32_t = u32x16_t;
  using i64_t = i64x8_t;
  using u64_t = u64x8_t;
  using r32_t = r32x16_t;
  using r64_t = r64x8_t;
#endif

#if DIM_SIMD_MAX_REGISTER_SIZE==32
  using i8_t  = i8x32_t;
  using u8_t  = u8x32_t;
  using i16_t = i16x16_t;
  using u16_t = u16x16_t;
  using i32_t = i32x8_t;
  using u32_t = u32x8_t;
  using i64_t = i64x4_t;
  using u64_t = u64x4_t;
  using r32_t = r32x8_t;
  using r64_t = r64x4_t;
#endif

#if DIM_SIMD_MAX_REGISTER_SIZE==16
  using i8_t  = i8x16_t;
  using u8_t  = u8x16_t;
  using i16_t = i16x8_t;
  using u16_t = u16x8_t;
  using i32_t = i32x4_t;
  using u32_t = u32x4_t;
  using i64_t = i64x2_t;
  using u64_t = u64x2_t;
  using r32_t = r32x4_t;
  using r64_t = r64x2_t;
#endif

#undef DIM_SIMD_MAX_REGISTER_SIZE

//~~~~ properties ~~~~

template<typename T,
         typename =expect_simd<T>>
inline constexpr
int
vector_size(T v)
{
  return int(sizeof(v));
}

template<typename T,
         typename =expect_simd<T>>
inline constexpr
int
value_size(T v)
{
  return int(sizeof(v[0]));
}

template<typename T,
         typename =expect_simd<T>>
inline constexpr
int
value_count(T v)
{
  return vector_size(v)/value_size(v);
}

template<typename T,
         typename =expect_simd<T>>
inline
std::string
to_string(T v)
{
  auto r=std::string{"{ "};
  for(int i=0; i<value_count(v); ++i)
  {
    if(i!=0)
    {
      r+=", ";
    }
    r+=std::to_string(v[i]);
  }
  r+=" }";
  return r;
}

//~~~~ load/store ~~~~

template<typename T,
         typename =expect_simd<T>>
inline
T
load_u(const T *unaligned_addr)
{
  struct unaligned { T v; } __attribute__((__packed__));
  return reinterpret_cast<const unaligned *>(unaligned_addr)->v;
}

template<typename T,
         typename =expect_simd<T>>
inline
T
load_a(const T *aligned_addr)
{
  return *aligned_addr;
}

template<typename T,
         typename =expect_simd<T>>
inline
void
store_u(T *unaligned_addr,
        T v)
{
  struct unaligned { T v; } __attribute__((__packed__));
  reinterpret_cast<unaligned *>(unaligned_addr)->v=v;
}

template<typename T,
         typename =expect_simd<T>>
inline
void
store_a(T *aligned_addr,
        T v)
{
  *aligned_addr=v;
}

//~~~~ fill ~~~~

template<typename T,
         typename =expect_simd<T>>
inline constexpr
T
zero()
{
  return T{0};
}

template<typename T,
         typename =expect_simd<T>>
inline constexpr
T
fill(value_type_t<T> v)
{
  return zero<T>()+v;
}

//~~~~ min/max ~~~~

template<typename T,
         typename =expect_simd<T>>
inline constexpr
T
min(T a,
    T b)
{
  const auto cmp=a<b;
  return (a&cmp)|(b&~cmp);
}

template<typename T,
         typename =expect_simd<T>>
inline constexpr
T
max(T a,
    T b)
{
  const auto cmp=a>b;
  return (a&cmp)|(b&~cmp);
}

//~~~~ shuffle ~~~~

#if defined __clang__
# define DIM_SIMD_SHUFFLE_1(mask) \
         __builtin_shufflevector(v, v, mask)
# define DIM_SIMD_SHUFFLE_2(mask) \
         __builtin_shufflevector(low, high, mask)
#else
# define DIM_SIMD_SHUFFLE_1(mask) \
         __builtin_shuffle(v, shuffle_mask_t<T>{mask})
# define DIM_SIMD_SHUFFLE_2(mask) \
         __builtin_shuffle(low, high, shuffle_mask_t<T>{mask})
#endif

#define DIM_SIMD_MASK_64  N0,  N1,  N2,  N3,  N4,  N5,  N6,  N7, \
                          N8,  N9, N10, N11, N12, N13, N14, N15, \
                         N16, N17, N18, N19, N20, N21, N22, N23, \
                         N24, N25, N26, N27, N28, N29, N30, N31, \
                         N32, N33, N34, N35, N36, N37, N38, N39, \
                         N40, N41, N42, N43, N44, N45, N46, N47, \
                         N48, N49, N50, N51, N52, N53, N54, N55, \
                         N56, N57, N58, N59, N60, N61, N62, N63
#define DIM_SIMD_MASK_32  N0,  N1,  N2,  N3,  N4,  N5,  N6,  N7, \
                          N8,  N9, N10, N11, N12, N13, N14, N15, \
                         N16, N17, N18, N19, N20, N21, N22, N23, \
                         N24, N25, N26, N27, N28, N29, N30, N31
#define DIM_SIMD_MASK_16  N0,  N1,  N2,  N3,  N4,  N5,  N6,  N7, \
                          N8,  N9, N10, N11, N12, N13, N14, N15
#define DIM_SIMD_MASK_8   N0,  N1,  N2,  N3,  N4,  N5,  N6,  N7
#define DIM_SIMD_MASK_4   N0,  N1,  N2,  N3
#define DIM_SIMD_MASK_2   N0,  N1

template<uint8_t  N0, uint8_t  N1, uint8_t  N2, uint8_t  N3,
         uint8_t  N4, uint8_t  N5, uint8_t  N6, uint8_t  N7,
         uint8_t  N8, uint8_t  N9, uint8_t N10, uint8_t N11,
         uint8_t N12, uint8_t N13, uint8_t N14, uint8_t N15,
         uint8_t N16, uint8_t N17, uint8_t N18, uint8_t N19,
         uint8_t N20, uint8_t N21, uint8_t N22, uint8_t N23,
         uint8_t N24, uint8_t N25, uint8_t N26, uint8_t N27,
         uint8_t N28, uint8_t N29, uint8_t N30, uint8_t N31,
         uint8_t N32, uint8_t N33, uint8_t N34, uint8_t N35,
         uint8_t N36, uint8_t N37, uint8_t N38, uint8_t N39,
         uint8_t N40, uint8_t N41, uint8_t N42, uint8_t N43,
         uint8_t N44, uint8_t N45, uint8_t N46, uint8_t N47,
         uint8_t N48, uint8_t N49, uint8_t N50, uint8_t N51,
         uint8_t N52, uint8_t N53, uint8_t N54, uint8_t N55,
         uint8_t N56, uint8_t N57, uint8_t N58, uint8_t N59,
         uint8_t N60, uint8_t N61, uint8_t N62, uint8_t N63,
         typename T,
         typename =expect_simd<T>,
         typename =expect_value_count<T, 64>>
inline constexpr
T
shuffle(T v)
{
  return DIM_SIMD_SHUFFLE_1(DIM_SIMD_MASK_64);
}

template<uint8_t  N0, uint8_t  N1, uint8_t  N2, uint8_t  N3,
         uint8_t  N4, uint8_t  N5, uint8_t  N6, uint8_t  N7,
         uint8_t  N8, uint8_t  N9, uint8_t N10, uint8_t N11,
         uint8_t N12, uint8_t N13, uint8_t N14, uint8_t N15,
         uint8_t N16, uint8_t N17, uint8_t N18, uint8_t N19,
         uint8_t N20, uint8_t N21, uint8_t N22, uint8_t N23,
         uint8_t N24, uint8_t N25, uint8_t N26, uint8_t N27,
         uint8_t N28, uint8_t N29, uint8_t N30, uint8_t N31,
         uint8_t N32, uint8_t N33, uint8_t N34, uint8_t N35,
         uint8_t N36, uint8_t N37, uint8_t N38, uint8_t N39,
         uint8_t N40, uint8_t N41, uint8_t N42, uint8_t N43,
         uint8_t N44, uint8_t N45, uint8_t N46, uint8_t N47,
         uint8_t N48, uint8_t N49, uint8_t N50, uint8_t N51,
         uint8_t N52, uint8_t N53, uint8_t N54, uint8_t N55,
         uint8_t N56, uint8_t N57, uint8_t N58, uint8_t N59,
         uint8_t N60, uint8_t N61, uint8_t N62, uint8_t N63,
         typename T,
         typename =expect_simd<T>,
         typename =expect_value_count<T, 64>>
inline constexpr
T
shuffle(T low,
        T high)
{
  return DIM_SIMD_SHUFFLE_2(DIM_SIMD_MASK_64);
}

template<uint8_t  N0, uint8_t  N1, uint8_t  N2, uint8_t  N3,
         uint8_t  N4, uint8_t  N5, uint8_t  N6, uint8_t  N7,
         uint8_t  N8, uint8_t  N9, uint8_t N10, uint8_t N11,
         uint8_t N12, uint8_t N13, uint8_t N14, uint8_t N15,
         uint8_t N16, uint8_t N17, uint8_t N18, uint8_t N19,
         uint8_t N20, uint8_t N21, uint8_t N22, uint8_t N23,
         uint8_t N24, uint8_t N25, uint8_t N26, uint8_t N27,
         uint8_t N28, uint8_t N29, uint8_t N30, uint8_t N31,
         typename T,
         typename =expect_simd<T>,
         typename =expect_value_count<T, 32>>
inline constexpr
T
shuffle(T v)
{
  return DIM_SIMD_SHUFFLE_1(DIM_SIMD_MASK_32);
}

template<uint8_t  N0, uint8_t  N1, uint8_t  N2, uint8_t  N3,
         uint8_t  N4, uint8_t  N5, uint8_t  N6, uint8_t  N7,
         uint8_t  N8, uint8_t  N9, uint8_t N10, uint8_t N11,
         uint8_t N12, uint8_t N13, uint8_t N14, uint8_t N15,
         uint8_t N16, uint8_t N17, uint8_t N18, uint8_t N19,
         uint8_t N20, uint8_t N21, uint8_t N22, uint8_t N23,
         uint8_t N24, uint8_t N25, uint8_t N26, uint8_t N27,
         uint8_t N28, uint8_t N29, uint8_t N30, uint8_t N31,
         typename T,
         typename =expect_simd<T>,
         typename =expect_value_count<T, 32>>
inline constexpr
T
shuffle(T low,
        T high)
{
  return DIM_SIMD_SHUFFLE_2(DIM_SIMD_MASK_32);
}

template<uint8_t  N0, uint8_t  N1, uint8_t  N2, uint8_t  N3,
         uint8_t  N4, uint8_t  N5, uint8_t  N6, uint8_t  N7,
         uint8_t  N8, uint8_t  N9, uint8_t N10, uint8_t N11,
         uint8_t N12, uint8_t N13, uint8_t N14, uint8_t N15,
         typename T,
         typename =expect_simd<T>,
         typename =expect_value_count<T, 16>>
inline constexpr
T
shuffle(T v)
{
  return DIM_SIMD_SHUFFLE_1(DIM_SIMD_MASK_16);
}

template<uint8_t  N0, uint8_t  N1, uint8_t  N2, uint8_t  N3,
         uint8_t  N4, uint8_t  N5, uint8_t  N6, uint8_t  N7,
         uint8_t  N8, uint8_t  N9, uint8_t N10, uint8_t N11,
         uint8_t N12, uint8_t N13, uint8_t N14, uint8_t N15,
         typename T,
         typename =expect_simd<T>,
         typename =expect_value_count<T, 16>>
inline constexpr
T
shuffle(T low,
        T high)
{
  return DIM_SIMD_SHUFFLE_2(DIM_SIMD_MASK_16);
}

template<uint8_t  N0, uint8_t  N1, uint8_t  N2, uint8_t  N3,
         uint8_t  N4, uint8_t  N5, uint8_t  N6, uint8_t  N7,
         typename T,
         typename =expect_simd<T>,
         typename =expect_value_count<T, 8>>
inline constexpr
T
shuffle(T v)
{
  return DIM_SIMD_SHUFFLE_1(DIM_SIMD_MASK_8);
}

template<uint8_t  N0, uint8_t  N1, uint8_t  N2, uint8_t  N3,
         uint8_t  N4, uint8_t  N5, uint8_t  N6, uint8_t  N7,
         typename T,
         typename =expect_simd<T>,
         typename =expect_value_count<T, 8>>
inline constexpr
T
shuffle(T low,
        T high)
{
  return DIM_SIMD_SHUFFLE_2(DIM_SIMD_MASK_8);
}

template<uint8_t  N0, uint8_t  N1, uint8_t  N2, uint8_t  N3,
         typename T,
         typename =expect_simd<T>,
         typename =expect_value_count<T, 4>>
inline constexpr
T
shuffle(T v)
{
  return DIM_SIMD_SHUFFLE_1(DIM_SIMD_MASK_4);
}

template<uint8_t  N0, uint8_t  N1, uint8_t  N2, uint8_t  N3,
         typename T,
         typename =expect_simd<T>,
         typename =expect_value_count<T, 4>>
inline constexpr
T
shuffle(T low,
        T high)
{
  return DIM_SIMD_SHUFFLE_2(DIM_SIMD_MASK_4);
}

template<uint8_t  N0, uint8_t  N1,
         typename T,
         typename =expect_simd<T>,
         typename =expect_value_count<T, 2>>
inline constexpr
T
shuffle(T v)
{
  return DIM_SIMD_SHUFFLE_1(DIM_SIMD_MASK_2);
}

template<uint8_t  N0, uint8_t  N1,
         typename T,
         typename =expect_simd<T>,
         typename =expect_value_count<T, 2>>
inline constexpr
T
shuffle(T low,
        T high)
{
  return DIM_SIMD_SHUFFLE_2(DIM_SIMD_MASK_2);
}

#undef DIM_SIMD_SHUFFLE_1
#undef DIM_SIMD_SHUFFLE_2
#undef DIM_SIMD_MASK_64
#undef DIM_SIMD_MASK_32
#undef DIM_SIMD_MASK_16
#undef DIM_SIMD_MASK_8
#undef DIM_SIMD_MASK_4
#undef DIM_SIMD_MASK_2

template<uint8_t N,
         typename T,
         typename =expect_simd<T>>
inline constexpr
T
down(T v)
{
  constexpr auto c=value_count_v<T>;
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

template<uint8_t N,
         typename T,
         typename =expect_simd<T>>
inline constexpr
T
up(T v)
{
  constexpr auto c=value_count_v<T>;
  return down<c-N>(v);
}

template<typename T,
         typename =expect_simd<T>>
inline constexpr
T
even(T low,
     T high)
{
  constexpr auto c=value_count_v<T>;
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

template<typename T,
         typename =expect_simd<T>>
inline constexpr
T
odd(T low,
    T high)
{
  constexpr auto c=value_count_v<T>;
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

} // namespace dim::simd

namespace std {
// FIXME: ugly but convenient way to provide std::operator<< for simd types

template<typename T,
         typename =dim::simd::expect_simd<T>>
inline
ostream &
operator<<(ostream &os,
           const T &v)
{
  return os << dim::simd::to_string(v);
}

} // namespace std

#endif // DIM_SIMD_HPP

//----------------------------------------------------------------------------
