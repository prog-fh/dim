//----------------------------------------------------------------------------

#ifndef DIM_ALIGNED_BUFFER_HPP
#define DIM_ALIGNED_BUFFER_HPP 1

#include "utils.hpp"

#include <memory>
#include <cstdlib>

#if !defined DIM_ALIGNED_BUFFER_DISABLE_SIMD
# define DIM_ALIGNED_BUFFER_DISABLE_SIMD 0
#endif

#if !DIM_ALIGNED_BUFFER_DISABLE_SIMD
# include "simd.hpp"
#endif

namespace dim {

template<typename T,
         int Alignment=assumed_cacheline_size>
class AlignedBuffer
{
public:

  static_assert(std::is_pod_v<T>,
                "plain-old-data type expected");

  static_assert((Alignment>0)&&((Alignment&(Alignment-1))==0),
                "positive power-of-two alignment expected");

  static constexpr auto alignment=Alignment;

  explicit
  AlignedBuffer(int count=0)
  : count_{count}
  , data_{}
  {
    const auto requested=count*int(sizeof(T));
    // align at the end too (so that simd operations can overflow)
    const auto padded=requested+alignment-(requested%alignment);
#if defined __APPLE__ || defined _WIN32
    // FIXME: some systems lack some standard features!
    auto *p=static_cast<unsigned char *>(std::malloc(alignment+padded));
    const auto offset=static_cast<unsigned char>
      (alignment-reinterpret_cast<std::size_t>(p)%alignment);
    p+=offset;
    p[-1]=offset;
#else
    auto *p=std::aligned_alloc(alignment, padded);
#endif
    data_.reset(reinterpret_cast<T *>(p));
    std::fill(data_.get(), data_.get()+(padded/int(sizeof(T))), T{});
  }

  int
  count() const
  {
    return count_;
  }

  T *
  data() DIM_ASSUME_ALIGNED(alignment)
  {
    return data_.get();
  }

  const T *
  cdata() const DIM_ASSUME_ALIGNED(alignment)
  {
    return data_.get();
  }

#if !DIM_ALIGNED_BUFFER_DISABLE_SIMD
  using simd_t = dim::simd::simd_t<T, dim::simd::max_vector_size>;

  static_assert((alignment%simd_t::vector_size)==0,
                "alignment should be a multiple of simd vector size");

  simd_t *
  simd_data() DIM_ASSUME_ALIGNED(alignment)
  {
    return reinterpret_cast<simd_t *>(data_.get());
  }

  const simd_t *
  simd_cdata() const DIM_ASSUME_ALIGNED(alignment)
  {
    return reinterpret_cast<const simd_t *>(data_.get());
  }
#endif

private:

  struct Deleter
  {
    void
    operator()(void *ptr)
    {
#if defined __APPLE__ || defined _WIN32
      // FIXME: some systems lack some standard features!
      auto *p=static_cast<unsigned char *>(ptr);
      const auto offset=p[-1];
      std::free(p-offset);
#else
      std::free(ptr);
#endif
    }
  };

  int count_;
  std::unique_ptr<T[], Deleter> data_;
};

//----------------------------------------------------------------------------

#if DIM_ALIGNED_BUFFER_DISABLE_SIMD
# define DIM_ALIGNED_BUFFER_ACCESS_DATA(id) \
    auto * DIM_RESTRICT d##id=buffer##id.data();
# define DIM_ALIGNED_BUFFER_ACCESS_CDATA(id) \
    const auto * DIM_RESTRICT d##id=buffer##id.cdata();
# define DIM_ALIGNED_BUFFER_ITERATE(call) \
    const auto count=buffer1.count(); \
    for(auto [i, i_end]=dim::sequence_part(count, part_id, part_count); \
        i<i_end; ++i) { call; }
#else
# define DIM_ALIGNED_BUFFER_ACCESS_DATA(id) \
    auto * DIM_RESTRICT d##id=buffer##id.simd_data(); \
    using simd_t##id = typename std::decay_t<decltype(buffer##id)>::simd_t; \
    static_assert(simd_t1::value_count==simd_t##id::value_count);
# define DIM_ALIGNED_BUFFER_ACCESS_CDATA(id) \
    const auto * DIM_RESTRICT d##id=buffer##id.simd_cdata(); \
    using simd_t##id = typename std::decay_t<decltype(buffer##id)>::simd_t; \
    static_assert(simd_t1::value_count==simd_t##id::value_count);
# define DIM_ALIGNED_BUFFER_ITERATE(call) \
    const auto count=(buffer1.count()+simd_t1::value_count-1)/ \
                     simd_t1::value_count; \
    for(auto [i, i_end]= dim::sequence_part(count, part_id, part_count); \
        i<i_end; ++i) { call; }
#endif

template<typename T1,
         typename Fnct>
inline
void
apply0(int part_id, int part_count,
       const AlignedBuffer<T1> &buffer1,
       Fnct fnct)
{
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(1)
  DIM_ALIGNED_BUFFER_ITERATE(fnct(d1[i]))
}

template<typename T1,
         typename T2,
         typename Fnct>
inline
void
apply0(int part_id, int part_count,
       const AlignedBuffer<T1> &buffer1,
       const AlignedBuffer<T2> &buffer2,
       Fnct fnct)
{
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(1)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(2)
  DIM_ALIGNED_BUFFER_ITERATE(fnct(d1[i], d2[i]))
}

template<typename T1,
         typename T2,
         typename T3,
         typename Fnct>
inline
void
apply0(int part_id, int part_count,
       const AlignedBuffer<T1> &buffer1,
       const AlignedBuffer<T2> &buffer2,
       const AlignedBuffer<T3> &buffer3,
       Fnct fnct)
{
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(1)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(2)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(3)
  DIM_ALIGNED_BUFFER_ITERATE(fnct(d1[i], d2[i], d3[i]))
}

template<typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename Fnct>
inline
void
apply0(int part_id, int part_count,
       const AlignedBuffer<T1> &buffer1,
       const AlignedBuffer<T2> &buffer2,
       const AlignedBuffer<T3> &buffer3,
       const AlignedBuffer<T4> &buffer4,
       Fnct fnct)
{
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(1)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(2)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(3)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(4)
  DIM_ALIGNED_BUFFER_ITERATE(fnct(d1[i], d2[i], d3[i], d4[i]))
}

template<typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename T5,
         typename Fnct>
inline
void
apply0(int part_id, int part_count,
       const AlignedBuffer<T1> &buffer1,
       const AlignedBuffer<T2> &buffer2,
       const AlignedBuffer<T3> &buffer3,
       const AlignedBuffer<T4> &buffer4,
       const AlignedBuffer<T5> &buffer5,
       Fnct fnct)
{
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(1)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(2)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(3)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(4)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(5)
  DIM_ALIGNED_BUFFER_ITERATE(fnct(d1[i], d2[i], d3[i], d4[i], d5[i]))
}

template<typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename T5,
         typename T6,
         typename Fnct>
inline
void
apply0(int part_id, int part_count,
       const AlignedBuffer<T1> &buffer1,
       const AlignedBuffer<T2> &buffer2,
       const AlignedBuffer<T3> &buffer3,
       const AlignedBuffer<T4> &buffer4,
       const AlignedBuffer<T5> &buffer5,
       const AlignedBuffer<T6> &buffer6,
       Fnct fnct)
{
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(1)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(2)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(3)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(4)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(5)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(6)
  DIM_ALIGNED_BUFFER_ITERATE(fnct(d1[i], d2[i], d3[i], d4[i], d5[i], d6[i]))
}

//----------------------------------------------------------------------------

template<typename T1,
         typename Fnct>
inline
void
apply1(AlignedBuffer<T1> &buffer1,
       int part_id, int part_count,
       Fnct fnct)
{
  DIM_ALIGNED_BUFFER_ACCESS_DATA(1)
  DIM_ALIGNED_BUFFER_ITERATE(fnct(d1[i]))
}

template<typename T1,
         typename T2,
         typename Fnct>
inline
void
apply1(AlignedBuffer<T1> &buffer1,
       int part_id, int part_count,
       const AlignedBuffer<T2> &buffer2,
       Fnct fnct)
{
  DIM_ALIGNED_BUFFER_ACCESS_DATA(1)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(2)
  DIM_ALIGNED_BUFFER_ITERATE(fnct(d1[i], d2[i]))
}

template<typename T1,
         typename T2,
         typename T3,
         typename Fnct>
inline
void
apply1(AlignedBuffer<T1> &buffer1,
       int part_id, int part_count,
       const AlignedBuffer<T2> &buffer2,
       const AlignedBuffer<T3> &buffer3,
       Fnct fnct)
{
  DIM_ALIGNED_BUFFER_ACCESS_DATA(1)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(2)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(3)
  DIM_ALIGNED_BUFFER_ITERATE(fnct(d1[i], d2[i], d3[i]))
}

template<typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename Fnct>
inline
void
apply1(AlignedBuffer<T1> &buffer1,
       int part_id, int part_count,
       const AlignedBuffer<T2> &buffer2,
       const AlignedBuffer<T3> &buffer3,
       const AlignedBuffer<T4> &buffer4,
       Fnct fnct)
{
  DIM_ALIGNED_BUFFER_ACCESS_DATA(1)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(2)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(3)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(4)
  DIM_ALIGNED_BUFFER_ITERATE(fnct(d1[i], d2[i], d3[i], d4[i]))
}

template<typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename T5,
         typename Fnct>
inline
void
apply1(AlignedBuffer<T1> &buffer1,
       int part_id, int part_count,
       const AlignedBuffer<T2> &buffer2,
       const AlignedBuffer<T3> &buffer3,
       const AlignedBuffer<T4> &buffer4,
       const AlignedBuffer<T5> &buffer5,
       Fnct fnct)
{
  DIM_ALIGNED_BUFFER_ACCESS_DATA(1)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(2)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(3)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(4)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(5)
  DIM_ALIGNED_BUFFER_ITERATE(fnct(d1[i], d2[i], d3[i], d4[i], d5[i]))
}

template<typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename T5,
         typename T6,
         typename Fnct>
inline
void
apply1(AlignedBuffer<T1> &buffer1,
       int part_id, int part_count,
       const AlignedBuffer<T2> &buffer2,
       const AlignedBuffer<T3> &buffer3,
       const AlignedBuffer<T4> &buffer4,
       const AlignedBuffer<T5> &buffer5,
       const AlignedBuffer<T6> &buffer6,
       Fnct fnct)
{
  DIM_ALIGNED_BUFFER_ACCESS_DATA(1)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(2)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(3)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(4)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(5)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(6)
  DIM_ALIGNED_BUFFER_ITERATE(fnct(d1[i], d2[i], d3[i], d4[i], d5[i], d6[i]))
}

//----------------------------------------------------------------------------

template<typename T1,
         typename T2,
         typename Fnct>
inline
void
apply2(AlignedBuffer<T1> &buffer1,
       AlignedBuffer<T2> &buffer2,
       int part_id, int part_count,
       Fnct fnct)
{
  DIM_ALIGNED_BUFFER_ACCESS_DATA(1)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(2)
  DIM_ALIGNED_BUFFER_ITERATE(fnct(d1[i], d2[i]))
}

template<typename T1,
         typename T2,
         typename T3,
         typename Fnct>
inline
void
apply2(AlignedBuffer<T1> &buffer1,
       AlignedBuffer<T2> &buffer2,
       int part_id, int part_count,
       const AlignedBuffer<T3> &buffer3,
       Fnct fnct)
{
  DIM_ALIGNED_BUFFER_ACCESS_DATA(1)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(2)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(3)
  DIM_ALIGNED_BUFFER_ITERATE(fnct(d1[i], d2[i], d3[i]))
}

template<typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename Fnct>
inline
void
apply2(AlignedBuffer<T1> &buffer1,
       AlignedBuffer<T2> &buffer2,
       int part_id, int part_count,
       const AlignedBuffer<T3> &buffer3,
       const AlignedBuffer<T4> &buffer4,
       Fnct fnct)
{
  DIM_ALIGNED_BUFFER_ACCESS_DATA(1)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(2)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(3)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(4)
  DIM_ALIGNED_BUFFER_ITERATE(fnct(d1[i], d2[i], d3[i], d4[i]))
}

template<typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename T5,
         typename Fnct>
inline
void
apply2(AlignedBuffer<T1> &buffer1,
       AlignedBuffer<T2> &buffer2,
       int part_id, int part_count,
       const AlignedBuffer<T3> &buffer3,
       const AlignedBuffer<T4> &buffer4,
       const AlignedBuffer<T5> &buffer5,
       Fnct fnct)
{
  DIM_ALIGNED_BUFFER_ACCESS_DATA(1)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(2)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(3)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(4)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(5)
  DIM_ALIGNED_BUFFER_ITERATE(fnct(d1[i], d2[i], d3[i], d4[i], d5[i]))
}

template<typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename T5,
         typename T6,
         typename Fnct>
inline
void
apply2(AlignedBuffer<T1> &buffer1,
       AlignedBuffer<T2> &buffer2,
       int part_id, int part_count,
       const AlignedBuffer<T3> &buffer3,
       const AlignedBuffer<T4> &buffer4,
       const AlignedBuffer<T5> &buffer5,
       const AlignedBuffer<T6> &buffer6,
       Fnct fnct)
{
  DIM_ALIGNED_BUFFER_ACCESS_DATA(1)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(2)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(3)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(4)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(5)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(6)
  DIM_ALIGNED_BUFFER_ITERATE(fnct(d1[i], d2[i], d3[i], d4[i], d5[i], d6[i]))
}

//----------------------------------------------------------------------------

template<typename T1,
         typename T2,
         typename T3,
         typename Fnct>
inline
void
apply3(AlignedBuffer<T1> &buffer1,
       AlignedBuffer<T2> &buffer2,
       AlignedBuffer<T3> &buffer3,
       int part_id, int part_count,
       Fnct fnct)
{
  DIM_ALIGNED_BUFFER_ACCESS_DATA(1)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(2)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(3)
  DIM_ALIGNED_BUFFER_ITERATE(fnct(d1[i], d2[i], d3[i]))
}

template<typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename Fnct>
inline
void
apply3(AlignedBuffer<T1> &buffer1,
       AlignedBuffer<T2> &buffer2,
       AlignedBuffer<T3> &buffer3,
       int part_id, int part_count,
       const AlignedBuffer<T4> &buffer4,
       Fnct fnct)
{
  DIM_ALIGNED_BUFFER_ACCESS_DATA(1)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(2)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(3)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(4)
  DIM_ALIGNED_BUFFER_ITERATE(fnct(d1[i], d2[i], d3[i], d4[i]))
}

template<typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename T5,
         typename Fnct>
inline
void
apply3(AlignedBuffer<T1> &buffer1,
       AlignedBuffer<T2> &buffer2,
       AlignedBuffer<T3> &buffer3,
       int part_id, int part_count,
       const AlignedBuffer<T4> &buffer4,
       const AlignedBuffer<T5> &buffer5,
       Fnct fnct)
{
  DIM_ALIGNED_BUFFER_ACCESS_DATA(1)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(2)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(3)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(4)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(5)
  DIM_ALIGNED_BUFFER_ITERATE(fnct(d1[i], d2[i], d3[i], d4[i], d5[i]))
}

template<typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename T5,
         typename T6,
         typename Fnct>
inline
void
apply3(AlignedBuffer<T1> &buffer1,
       AlignedBuffer<T2> &buffer2,
       AlignedBuffer<T3> &buffer3,
       int part_id, int part_count,
       const AlignedBuffer<T4> &buffer4,
       const AlignedBuffer<T5> &buffer5,
       const AlignedBuffer<T6> &buffer6,
       Fnct fnct)
{
  DIM_ALIGNED_BUFFER_ACCESS_DATA(1)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(2)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(3)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(4)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(5)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(6)
  DIM_ALIGNED_BUFFER_ITERATE(fnct(d1[i], d2[i], d3[i], d4[i], d5[i], d6[i]))
}

//----------------------------------------------------------------------------

template<typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename Fnct>
inline
void
apply4(AlignedBuffer<T1> &buffer1,
       AlignedBuffer<T2> &buffer2,
       AlignedBuffer<T3> &buffer3,
       AlignedBuffer<T4> &buffer4,
       int part_id, int part_count,
       Fnct fnct)
{
  DIM_ALIGNED_BUFFER_ACCESS_DATA(1)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(2)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(3)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(4)
  DIM_ALIGNED_BUFFER_ITERATE(fnct(d1[i], d2[i], d3[i], d4[i]))
}

template<typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename T5,
         typename Fnct>
inline
void
apply4(AlignedBuffer<T1> &buffer1,
       AlignedBuffer<T2> &buffer2,
       AlignedBuffer<T3> &buffer3,
       AlignedBuffer<T4> &buffer4,
       int part_id, int part_count,
       const AlignedBuffer<T5> &buffer5,
       Fnct fnct)
{
  DIM_ALIGNED_BUFFER_ACCESS_DATA(1)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(2)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(3)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(4)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(5)
  DIM_ALIGNED_BUFFER_ITERATE(fnct(d1[i], d2[i], d3[i], d4[i], d5[i]))
}

template<typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename T5,
         typename T6,
         typename Fnct>
inline
void
apply4(AlignedBuffer<T1> &buffer1,
       AlignedBuffer<T2> &buffer2,
       AlignedBuffer<T3> &buffer3,
       AlignedBuffer<T4> &buffer4,
       int part_id, int part_count,
       const AlignedBuffer<T5> &buffer5,
       const AlignedBuffer<T6> &buffer6,
       Fnct fnct)
{
  DIM_ALIGNED_BUFFER_ACCESS_DATA(1)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(2)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(3)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(4)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(5)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(6)
  DIM_ALIGNED_BUFFER_ITERATE(fnct(d1[i], d2[i], d3[i], d4[i], d5[i], d6[i]))
}

//----------------------------------------------------------------------------

template<typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename T5,
         typename Fnct>
inline
void
apply5(AlignedBuffer<T1> &buffer1,
       AlignedBuffer<T2> &buffer2,
       AlignedBuffer<T3> &buffer3,
       AlignedBuffer<T4> &buffer4,
       AlignedBuffer<T5> &buffer5,
       int part_id, int part_count,
       Fnct fnct)
{
  DIM_ALIGNED_BUFFER_ACCESS_DATA(1)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(2)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(3)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(4)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(5)
  DIM_ALIGNED_BUFFER_ITERATE(fnct(d1[i], d2[i], d3[i], d4[i], d5[i]))
}

template<typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename T5,
         typename T6,
         typename Fnct>
inline
void
apply5(AlignedBuffer<T1> &buffer1,
       AlignedBuffer<T2> &buffer2,
       AlignedBuffer<T3> &buffer3,
       AlignedBuffer<T4> &buffer4,
       AlignedBuffer<T5> &buffer5,
       int part_id, int part_count,
       const AlignedBuffer<T6> &buffer6,
       Fnct fnct)
{
  DIM_ALIGNED_BUFFER_ACCESS_DATA(1)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(2)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(3)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(4)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(5)
  DIM_ALIGNED_BUFFER_ACCESS_CDATA(6)
  DIM_ALIGNED_BUFFER_ITERATE(fnct(d1[i], d2[i], d3[i], d4[i], d5[i], d6[i]))
}

//----------------------------------------------------------------------------

template<typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename T5,
         typename T6,
         typename Fnct>
inline
void
apply6(AlignedBuffer<T1> &buffer1,
       AlignedBuffer<T2> &buffer2,
       AlignedBuffer<T3> &buffer3,
       AlignedBuffer<T4> &buffer4,
       AlignedBuffer<T5> &buffer5,
       AlignedBuffer<T6> &buffer6,
       int part_id, int part_count,
       Fnct fnct)
{
  DIM_ALIGNED_BUFFER_ACCESS_DATA(1)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(2)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(3)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(4)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(5)
  DIM_ALIGNED_BUFFER_ACCESS_DATA(6)
  DIM_ALIGNED_BUFFER_ITERATE(fnct(d1[i], d2[i], d3[i], d4[i], d5[i], d6[i]))
}

#undef DIM_ALIGNED_BUFFER_ACCESS_DATA
#undef DIM_ALIGNED_BUFFER_ACCESS_CDATA
#undef DIM_ALIGNED_BUFFER_ITERATE

//----------------------------------------------------------------------------

template<typename T>
inline
void
fill(AlignedBuffer<T> &dst,
     int part_id, int part_count,
     const T &value)
{
  apply1(dst,
    part_id, part_count,
    [&value](auto &p)
    {
      const std::decay_t<decltype(p)> v{value};
      p=v;
    });
}

template<typename T>
inline
void
fill(AlignedBuffer<T> &dst,
     int part_id, int part_count,
     int width, [[maybe_unused]] int height,
     int x, int y, int w, int h,
     const T &value)
{
  // FIXME: is there a simd-friendly way to do this?
  auto * DIM_RESTRICT d=dst.data();
  for(auto [yid, yid_end]=dim::sequence_part(y, y+h, part_id, part_count);
      yid<yid_end; ++yid)
  {
    const auto xid=yid*width+x;
    for(auto id=xid, id_end=xid+w; id<id_end; ++id)
    {
      d[id]=value;
    }
  }
}

template<typename T>
inline
T
mean(const AlignedBuffer<T> &buffer,
     int part_id, int part_count)
{
  // FIXME: consider horizontal sum for simd
  auto sum=T{};
  apply0(
    buffer,
    part_id, part_count,
    [&sum](const auto & p)
    {
      sum+=p;
    });
  return sum/T(buffer.count());
}

template<typename T>
inline
T
mean(const AlignedBuffer<T> &buffer,
     int part_id, int part_count,
     int width, [[maybe_unused]] int height,
     int x, int y, int w, int h)
{
  // FIXME: is there a simd-friendly way to do this?
  auto sum=T{};
  const auto * DIM_RESTRICT p=buffer.cdata();
  for(auto [yid, yid_end]=dim::sequence_part(y, y+h, part_id, part_count);
      yid<yid_end; ++yid)
  {
    const auto xid=yid*width+x;
    for(auto id=xid, id_end=xid+w; id<id_end; ++id)
    {
      sum+=p[id];
    }
  }
  return sum/T(w*h);
}

} // namespace dim

#endif // DIM_ALIGNED_BUFFER_HPP

//----------------------------------------------------------------------------
