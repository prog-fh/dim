//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#ifndef DIM_TXT_HPP
#define DIM_TXT_HPP

#include <type_traits>
#include <string>
#include <string_view>
#include <vector>
#include <cmath>
#include <utility>
#include <cctype>
#include <limits>
#include <cstring>

#include <unistd.h>

namespace dim::txt {

template<typename T,
         typename = std::enable_if_t<std::is_integral_v<T>>>
inline
std::string
hex(T value)
{
  if constexpr(std::is_unsigned_v<T>)
  {
    constexpr auto digit_count=int(2*sizeof(T));
    auto result=std::string{};
    result.reserve(digit_count);
    constexpr auto digits="0123456789ABCDEF";
    for(auto shift=4*(digit_count-1); shift>=0; shift-=4)
    {
      result+=digits[(value>>shift)&0x0F];
    }
    return result;
  }
  else
  {
    return hex(static_cast<std::make_unsigned_t<T>>(value));
  }
}

inline
std::string
hex(const void *value)
{
  return hex(reinterpret_cast<std::size_t>(value));
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<typename T,
         typename = std::enable_if_t<std::is_integral_v<T>>>
inline
std::string
bin(T value)
{
  if constexpr(std::is_unsigned_v<T>)
  {
    constexpr auto digit_count=int(8*sizeof(T));
    std::string result;
    result.reserve(digit_count);
    for(auto shift=digit_count-1; shift>=0; --shift)
    {
      result+=char('0'+((value>>shift)&1));
    }
    return result;
  }
  else
  {
    return bin(static_cast<std::make_unsigned_t<T>>(value));
  }
}

inline
std::string
bin(const void *value)
{
  return bin(reinterpret_cast<std::size_t>(value));
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inline
void
fmt(std::string &inout_result,
    const char *value)
{
  inout_result+=value;
}

inline
void
fmt(std::string &inout_result,
    std::string_view value)
{
  inout_result+=value;
}

inline
void
fmt(std::string &inout_result,
    const std::string &value)
{
  inout_result+=value;
}

inline
void
fmt(std::string &inout_result,
    bool value)
{
  inout_result+=value ? "true" : "false";
}

inline
void
fmt(std::string &inout_result,
    char value)
{
  inout_result+=value;
}

template<typename T,
         typename = std::enable_if_t<std::is_integral_v<T>>>
inline
void
fmt(std::string &inout_result,
    T value)
{
  if constexpr(std::is_unsigned_v<T>)
  {
    constexpr auto ten=T{10};
    auto div=T{1};
    for(auto d=std::numeric_limits<T>::digits10; d; --d)
    {
      div=static_cast<T>(div*ten);
    }
    auto digit_count=0;
    for(; div!=0; div=static_cast<T>(div/ten))
    {
      const auto digit=int(value/div);
      if(digit||(digit_count!=0))
      {
        inout_result+=char('0'+digit);
        ++digit_count;
      }
      value=static_cast<T>(value-digit*div);
    }
    if(digit_count==0)
    {
      inout_result+='0';
    }
  }
  else
  {
    using unsigned_type = std::make_unsigned_t<T>;
    const auto cast_value=static_cast<unsigned_type>(value);
    if(value<0)
    {
      inout_result+='-';
      constexpr auto one=unsigned_type{1};
      const auto neg_value=static_cast<unsigned_type>(one+~cast_value);
      fmt(inout_result, neg_value);
    }
    else
    {
      fmt(inout_result, cast_value);
    }
  }
}

inline
void
fmt(std::string &inout_result,
    double value)
{
  if(!std::isfinite(value))
  {
    inout_result+="NaN";
  }
  else if(value==0.0)
  {
    inout_result+="0.0";
  }
  else if(value<0.0)
  {
    inout_result+='-';
    fmt(inout_result, -value);
  }
  else if((value>1e-4)&&(value<1e5))
  {
    // FIXME: last digit is not rounded towards nearest
    auto div=1e5;
    for(auto dot_pos=5, digit_count=0; digit_count<6; --dot_pos, div/=10.0)
    {
      const auto digit=int(value/div);
      value=std::fmod(value, div);
      if(digit||(digit_count!=0))
      {
        ++digit_count;
      }
      if((digit_count!=0)||(dot_pos<0))
      {
        inout_result+=char('0'+digit);
      }
      if(dot_pos==0)
      {
        if(digit_count==0)
        {
          inout_result+='0';
        }
        inout_result+='.';
      }
    }
    auto sz=int(size(inout_result));
    while((inout_result[sz-1]=='0')&&(inout_result[sz-2]!='.'))
    {
      inout_result.resize(--sz);
    }
  }
  else
  {
    const auto exponent=std::floor(std::log10(value));
    const auto mantissa=value/std::pow(10.0, exponent);
    fmt(inout_result, mantissa);
    inout_result+='e';
    fmt(inout_result, int(exponent));
  }
}

inline
void
fmt(std::string &inout_result,
    float value)
{
  fmt(inout_result, static_cast<double>(value));
}

template<typename Elem>
inline
void
fmt(std::string &inout_result,
    const std::vector<Elem> &value)
{
  inout_result+='{';
  auto first=true;
  for(const auto &elem: value)
  {
    if(!first)
    {
      inout_result+=", ";
    }
    first=false;
    fmt(inout_result, elem);
  }
  inout_result+='}';
}

template<typename First,
         typename ...Args>
inline
void
fmt(std::string &inout_result,
    const char *format,
    First first,
    Args &&...args)
{
  while(*format)
  {
    if(*format=='%')
    {
      fmt(inout_result, first);
      fmt(inout_result, ++format, std::forward<Args>(args)...);
      return;
    }
    inout_result+=*format++;
  }
}

template<typename ...Args>
inline
std::string
txt(const char *format,
    Args &&...args)
{
  auto result=std::string{};
  fmt(result, format, std::forward<Args>(args)...);
  return result;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<typename ...Args>
inline
int // written bytes
to_fd(int fd,
      const char *format,
      Args &&...args)
{
  const auto str=txt(format, std::forward<Args>(args)...);
  auto ptr=data(str);
  auto remaining=int(size(str));
  while(remaining)
  {
    const auto r=int(::write(fd, ptr, remaining));
    if(r<=0)
    {
      break; // error
    }
    ptr+=r;
    remaining-=r;
  }
  return int(size(str))-remaining;
}

template<typename ...Args>
inline
int // written bytes
out(const char *format,
    Args &&...args)
{
  return to_fd(STDOUT_FILENO, format, std::forward<Args>(args)...);
}

template<typename ...Args>
inline
int // written bytes
err(const char *format,
    Args &&...args)
{
  return to_fd(STDERR_FILENO, format, std::forward<Args>(args)...);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace impl_ {

inline
bool // go on
begin_extract_(const char *&input,
               int &remaining,
               bool &failure)
{
  if(failure)
  {
    return false;
  }
  for(; remaining!=0; ++input, --remaining)
  {
    if(!std::isspace(*input))
    {
      break;
    }
  }
  return true;
}

inline
void
extract_arg_(const char *&input,
             int &remaining,
             bool &failure,
             int &count,
             const char &literalChar)
{
  if(!begin_extract_(input, remaining, failure))
  {
    return;
  }
  if((remaining<1)||(*input!=literalChar))
  {
    failure=true;
    return;
  }
  ++input;
  --remaining;
  ++count;
}

template<int N>
inline
void
extract_arg_(const char *&input,
             int &remaining,
             bool &failure,
             int &count,
             const char(&literalString)[N])
{
  if(!begin_extract_(input, remaining, failure))
  {
    return;
  }
  if(remaining<N)
  {
    failure=true;
    return;
  }
  for(auto i=0; i<N-1; ++i)
  {
    if(input[i]!=literalString[i])
    {
      failure=true;
      return;
    }
  }
  input+=N-1;
  remaining-=N-1;
  ++count;
}

inline
void
extract_arg_(const char *&input,
             int &remaining,
             bool &failure,
             int &count,
             char &value)
{
  if(!begin_extract_(input, remaining, failure))
  {
    return;
  }
  if(remaining<1)
  {
    failure=true;
    return;
  }
  value=*input;
  ++input;
  --remaining;
  ++count;
}

inline
void
extract_arg_(const char *&input,
             int &remaining,
             bool &failure,
             int &count,
             std::string &value)
{
  if(!begin_extract_(input, remaining, failure))
  {
    return;
  }
  auto tmp=std::string{};
  auto start=remaining;
  for(; remaining!=0; ++input, --remaining)
  {
    const auto c=*input;
    if(std::isspace(c))
    {
      break;
    }
    tmp+=c;
  }
  if(remaining==start)
  {
    failure=true;
    return;
  }
  value=std::move(tmp);
  ++count;
}

template<typename T,
         typename = std::enable_if_t<std::is_integral_v<T>||
                                     std::is_floating_point_v<T>>>
inline
void
extract_arg_(const char *&input,
             int &remaining,
             bool &failure,
             int &count,
             T &value)
{
  if(!begin_extract_(input, remaining, failure))
  {
    return;
  }
  if constexpr(std::is_integral_v<T>)
  {
    [[maybe_unused]] auto negative=false;
    if(remaining!=0)
    {
      const auto c=*input;
      if(c=='+')
      {
        ++input;
        --remaining;
      }
      else if(c=='-')
      {
        if constexpr(std::is_unsigned_v<T>)
        {
          failure=true;
          return;
        }
        else
        {
          ++input;
          --remaining;
          negative=true;
        }
      }
    }
    const auto consume_digits=
      [&](const auto &limit)
      {
        using tmp_type = std::remove_const_t<std::remove_reference_t<
                                             decltype(limit)>>;
        auto tmp=tmp_type{};
        const auto prev_limit=limit/10;
        auto start=remaining;
        for(; remaining!=0; ++input, --remaining)
        {
          const auto c=*input;
          if(!std::isdigit(c))
          {
            break;
          }
          if(failure)
          {
            continue;
          }
          if(tmp>prev_limit)
          {
            failure=true;
            continue;
          }
          tmp=static_cast<tmp_type>(tmp*10);
          const auto digit=static_cast<tmp_type>(c-'0');
          if(limit-digit<tmp)
          {
            failure=true;
            continue;
          }
          tmp=static_cast<tmp_type>(tmp+digit);
        }
        if(remaining==start)
        {
          failure=true;
        }
        return tmp;
      };
    if constexpr(std::is_unsigned_v<T>)
    {
      const auto limit=std::numeric_limits<T>::max();
      const auto tmp=consume_digits(limit);
      if(failure)
      {
        return;
      }
      value=tmp;
      ++count;
    }
    else
    {
      using unsigned_type = std::make_unsigned_t<T>;
      constexpr auto one=unsigned_type{1};
      const auto cast_limit=static_cast<unsigned_type>(
        negative ? std::numeric_limits<T>::min()
                 : std::numeric_limits<T>::max());
      const auto neg_limit=static_cast<unsigned_type>(one+~cast_limit);
      const auto limit=negative ? neg_limit : cast_limit;
      const auto tmp=consume_digits(limit);
      if(failure)
      {
        return;
      }
      value=static_cast<T>(negative ? one+~tmp : tmp);
      ++count;
    }
  }
  else if constexpr(std::is_floating_point_v<T>)
  {
    // FIXME: no overflow detected
    const auto pos_m_sign=input;
    if(remaining!=0)
    {
      const auto c=*input;
      if((c=='-')||(c=='+'))
      {
        ++input;
        --remaining;
      }
    }
    const auto pos_m_int=input;
    for(; remaining!=0; ++input, --remaining)
    {
      if(!std::isdigit(*input))
      {
        break;
      }
    }
    const auto pos_m_sep=input;
    if(remaining!=0)
    {
      if(*input=='.')
      {
        ++input;
        --remaining;
      }
    }
    const auto pos_m_frac=input;
    for(; remaining!=0; ++input, --remaining)
    {
      if(!std::isdigit(*input))
      {
        break;
      }
    }
    const auto pos_pow=input;
    if(remaining!=0)
    {
      const auto c=*input;
      if((c=='e')||(c=='E'))
      {
        ++input;
        --remaining;
      }
    }
    const auto pos_e_sign=input;
    if(remaining!=0)
    {
      const auto c=*input;
      if((c=='-')||(c=='+'))
      {
        ++input;
        --remaining;
      }
    }
    const auto pos_e_int=input;
    for(; remaining!=0; ++input, --remaining)
    {
      if(!std::isdigit(*input))
      {
        break;
      }
    }
    const auto pos_end=input;
    const auto has_m_sign=pos_m_int>pos_m_sign;
    const auto has_m_int=pos_m_sep>pos_m_int;
    const auto has_m_sep=pos_m_frac>pos_m_sep;
    const auto has_m_frac=pos_pow>pos_m_frac;
    const auto has_pow=pos_e_sign>pos_pow;
    const auto has_e_sign=pos_e_int>pos_e_sign;
    const auto has_e_int=pos_end>pos_e_int;
    if(!(has_m_int||(has_m_sep&&has_m_frac)))
    {
      failure=true;
      return;
    }
    constexpr auto ten=T{10};
    auto mantissa=T{};
    for(auto p=pos_m_int; p<pos_m_sep; ++p)
    {
      const auto digit=static_cast<T>(*p-'0');
      mantissa=ten*mantissa+digit;
    }
    if(has_m_sep&&has_m_frac)
    {
      auto decimal=T{1/ten};
      for(auto p=pos_m_frac; p<pos_pow; ++p)
      {
        const auto digit=static_cast<T>(*p-'0');
        mantissa+=decimal*digit;
        decimal/=ten;
      }
    }
    if(has_m_sign&&(*pos_m_sign=='-'))
    {
      mantissa=-mantissa;
    }
    auto exponent=T{};
    if(has_pow&&has_e_int)
    {
      for(auto p=pos_e_int; p<pos_end; ++p)
      {
        const auto digit=static_cast<T>(*p-'0');
        exponent=ten*exponent+digit;
      }
      if(has_e_sign&&(*pos_e_sign=='-'))
      {
        exponent=-exponent;
      }
    }
    value=mantissa*std::pow(ten, exponent);
    ++count;
  }
}

struct variadic_pass_
{
  template<typename ...T>
  variadic_pass_(T...)
  {
  }
};

} // namespace impl_

template<typename ...Args>
inline
int // extraction count
extract(const std::string &input,
        Args &&...args)
{
  auto input_ptr=data(input);
  auto remaining=int(size(input));
  auto failure=false;
  auto count=0;
  impl_::variadic_pass_{
    (impl_::extract_arg_(input_ptr, remaining, failure, count,
                         std::forward<Args>(args)), 1)...};
  return count;
}

template<typename ...Args>
inline
int // extraction count
extract(std::string_view input,
        Args &&...args)
{
  auto input_ptr=data(input);
  auto remaining=int(size(input));
  auto failure=false;
  auto count=0;
  impl_::variadic_pass_{
    (impl_::extract_arg_(input_ptr, remaining, failure, count,
                         std::forward<Args>(args)), 1)...};
  return count;
}

template<typename ...Args>
inline
int // extraction count
extract(const char *input,
        Args &&...args)
{
  auto remaining=int(std::strlen(input));
  auto failure=false;
  auto count=0;
  impl_::variadic_pass_{
    (impl_::extract_arg_(input, remaining, failure, count,
                         std::forward<Args>(args)), 1)...};
  return count;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inline
std::string
read(int capacity=0x100)
{
  auto result=std::string{};
  result.resize(capacity);
  result.resize(::read(STDIN_FILENO, data(result), capacity));
  return result;
}

inline
std::string
read_all(int capacity)
{
  auto result=std::string{};
  result.resize(capacity);
  auto ptr=data(result);
  auto remaining=capacity;
  while(remaining)
  {
    const auto r=int(::read(STDIN_FILENO, ptr, remaining));
    if(r<=0)
    {
      break; // EOF
    }
    ptr+=r;
    remaining-=r;
  }
  result.resize(capacity-remaining);
  return result;
}

inline
std::string
read_line()
{
  auto result=std::string{};
  auto c=char{};
  while(::read(STDIN_FILENO, &c, 1)==1)
  {
    result+=c;
    if(c=='\n')
    {
      break; // end of line
    }
  }
  return result;
}

} // namespace dim::txt

#endif // DIM_TXT_HPP

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
