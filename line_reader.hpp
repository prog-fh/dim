//----------------------------------------------------------------------------

#ifndef DIM_LINE_READER_HPP
#define DIM_LINE_READER_HPP

#include <string>
#include <string_view>
#include <vector>
#include <fstream>
#include <cctype>
#include <stdexcept>
#include <iostream>

namespace dim {

class LineReader
{
public:

  explicit
  LineReader(std::string path,
             std::string comment_symbol="#",
             std::string include_keyword="include")
  : comment_symbol_{std::move(comment_symbol)}
  , include_keyword_{std::move(include_keyword)}
  , input_{}
  , line_{}
  , words_{}
  {
    open_(std::move(path));
    next();
  }

  const std::string &
  include_keyword() const
  {
    return include_keyword_;
  }

  const std::string &
  comment_symbol() const
  {
    return comment_symbol_;
  }

  const std::string &
  current_line() const
  {
    return line_;
  }

  const std::vector<std::string_view> &
  words() const
  {
    return words_;
  }

  bool // next non-empty line available
  next()
  {
    for(;;)
    {
      line_.clear();
      words_.clear();
      if(empty(input_))
      {
        return false;
      }
      auto &file=input_.back();
      if(!getline(file.stream, line_))
      {
        input_.pop_back();
        continue;
      }
      ++file.count;
      if(!empty(comment_symbol_))
      {
        if(auto pos=line_.find(comment_symbol_); pos!=line_.npos)
        {
          line_.resize(pos);
        }
      }
      const auto *str=data(line_);
      const auto len=int(size(line_));
      auto offset=0;
      auto last=0;
      for(;;)
      {
        auto count=0;
        for(auto i=offset; i<len; ++i)
        {
          if(std::isspace(str[i]))
          {
            if(i==offset)
            {
              ++offset;
            }
            else
            {
              break;
            }
          }
          else
          {
            last=i;
            ++count;
          }
        }
        if(!count)
        {
          break;
        }
        words_.emplace_back(str+offset, count);
        offset+=count;
      }
      if(!empty(words_))
      {
        line_.resize(last+1); // ignore remaining spaces
        if((size(words_)>=2)&&(words_[0]==include_keyword_))
        {
          open_(data(words_[1]));
        }
        else
        {
          return true;
        }
      }
    }
  }

  std::string
  where() const
  {
    auto result=std::string{};
    if(!empty(input_))
    {
      const auto &file=input_.back();
      result=file.path;
      result+=':';
      result+=std::to_string(file.count);
      result+=": ";
    }
    return result;
  }

private:

  void
  open_(std::string path)
  {
#if 0 // defined _WIN32 // file opening uses /
    constexpr auto sep='\\';
#else
    constexpr auto sep='/';
#endif
    auto stream=std::ifstream{path};
    if(!stream&&!empty(path)&&(path.front()!=sep)&&!empty(input_))
    {
      const auto &p=input_.back().path;
      if(const auto pos=p.find_last_of(sep); pos!=p.npos)
      {
        auto actual_path=p.substr(0, pos);
        actual_path+=sep;
        actual_path+=path;
        stream=std::ifstream{actual_path};
        if(stream)
        {
          path=std::move(actual_path);
        }
      }
    }
    if(!stream)
    {
      throw std::runtime_error{where()+
                               "cannot read from `"+path+"'"};
    }
    for(const auto &file: input_)
    {
      if(file.path==path)
      {
        throw std::runtime_error{where()+
                                 "recursive inclusion of `"+path+"'"};
      }
    }
    input_.emplace_back(File{std::move(path), std::move(stream), 0});
  }

  struct File
  {
    std::string path;
    std::ifstream stream;
    int count;
  };

  std::string comment_symbol_;
  std::string include_keyword_;
  std::vector<File> input_;
  std::string line_;
  std::vector<std::string_view> words_;
};

} // namespace dim

#endif // DIM_LINE_READER_HPP

//----------------------------------------------------------------------------
