#include <png.h>
#include <iostream>
#include <fstream>
#include <memory>

#include "culib/device/histogram.cuh"
#include "culib/device/memory/resizable_array.h"
#include "culib/device/memory/api.h"

class img_class
{
public:
  const unsigned int width {};
  const unsigned int height {};
  const unsigned int pixels_count {};
  const bool is_gray {};
  const std::size_t row_size {};
  const std::unique_ptr<const png_byte[]> data;

public:
  img_class (
    int width_arg,
    int height_arg,
    bool is_gray_arg,
    size_t row_size_arg,
    std::unique_ptr<png_byte[]> data_arg)
    : width (width_arg)
    , height (height_arg)
    , pixels_count (width * height)
    , is_gray (is_gray_arg)
    , row_size (row_size_arg)
    , data (std::move (data_arg))
  { }

};

std::unique_ptr<img_class> read_png_file (char *filename)
{
  FILE *fp = fopen (filename, "rb");

  if (!fp)
    return {};

  png_structp png = png_create_read_struct (PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

  if (!png)
    return {};

  png_infop info = png_create_info_struct(png);
  if(!info)
    return {};

  if (setjmp(png_jmpbuf(png)))
    return {};

  png_init_io (png, fp);

  png_read_info (png, info);

  int width = png_get_image_width (png, info);
  int height = png_get_image_height (png, info);
  png_byte color_type = png_get_color_type (png, info);
  png_byte bit_depth = png_get_bit_depth (png, info);

  if(bit_depth == 16)
    png_set_strip_16 (png);

  if(color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_palette_to_rgb (png);

  // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
  if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
    png_set_expand_gray_1_2_4_to_8(png);

  if(png_get_valid(png, info, PNG_INFO_tRNS))
    png_set_tRNS_to_alpha(png);

  png_read_update_info(png, info);

  const std::size_t row_size = png_get_rowbytes (png, info);
  std::unique_ptr<png_byte[]> raw_data (new png_byte[height * row_size]);
  std::unique_ptr<png_bytep[]> row_pointers_helper (new png_bytep[height]);

  {
    png_byte *raw_ptr = raw_data.get ();
    for(int y = 0; y < height; y++) {
        row_pointers_helper[y] = raw_ptr;
        raw_ptr += row_size;
      }
  }

  png_read_image(png, row_pointers_helper.get ());

  fclose(fp);

  png_destroy_read_struct(&png, &info, NULL);

  const bool is_gray = color_type == PNG_COLOR_TYPE_GRAY
                    || color_type == PNG_COLOR_TYPE_GRAY_ALPHA;

  return std::unique_ptr<img_class> {new img_class (width, height, is_gray, row_size, std::move (raw_data))};
}

int main (int argc, char *argv[])
{
  if (argc != 2)
    {
      std::cout << "Usage: " << argv[0] << " [bins count] [path to png file]";
      return 1;
    }

  auto img = read_png_file (argv[1]);
  if (!img)
    {
      std::cerr << "Can't read " << argv[1] << "\n";
      return 1;
    }

  if (!img->is_gray || img->width != img->row_size)
    {
      std::cerr << "Only grayscale png without alpha channel is supported now\n";
      return 1;
    }


  const unsigned int bins_count = 256;
  const unsigned int img_elements = img->row_size * img->height;
  culib::device::resizable_array<unsigned int> result (bins_count);
  culib::device::resizable_array<unsigned int> workspace (culib::device::histogram::get_gpu_workspace_size (bins_count, img->pixels_count));
  culib::device::resizable_array<png_byte> device_img (img_elements);
  culib::device::send_n (img->data.get (), img_elements, device_img.get ());
  culib::device::histogram hist (workspace.get ());
  hist (bins_count, img_elements, device_img.get (), result.get ());

  std::unique_ptr<unsigned int[]> host_result (new unsigned int[bins_count]);
  culib::device::recv_n (result.get (), bins_count, host_result.get ());

  std::ofstream os ("result.csv");

  for (unsigned int bin = 1; bin < bins_count; bin++)
    os << host_result[bin] << "\n";

  return 0;
}