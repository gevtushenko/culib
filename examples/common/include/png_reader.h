//
// Created by egi on 6/29/20.
//

#ifndef CULIB_PNG_READER_H
#define CULIB_PNG_READER_H

#include <memory>

class img_class
{
public:
  const unsigned int width {};
  const unsigned int height {};
  const unsigned int pixels_count {};
  const bool is_gray {};
  const std::size_t row_size {};
  const std::unique_ptr<const unsigned char[]> data;

public:
  img_class (
    int width_arg,
    int height_arg,
    bool is_gray_arg,
    size_t row_size_arg,
    std::unique_ptr<unsigned char[]> data_arg)
    : width (width_arg)
    , height (height_arg)
    , pixels_count (width * height)
    , is_gray (is_gray_arg)
    , row_size (row_size_arg)
    , data (std::move (data_arg))
  { }

};

std::unique_ptr<img_class> read_png_file (char *filename);
void write_png_file (unsigned char *data, unsigned int width, unsigned int height, const char *filename);

#endif //CULIB_PNG_READER_H
