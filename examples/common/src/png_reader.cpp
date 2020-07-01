//
// Created by egi on 6/29/20.
//

#include "png_reader.h"

#include <png.h>

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

void write_png_file (unsigned char *data, unsigned int width, unsigned int height, const char *filename)
{
  FILE *fp = fopen (filename, "wb");
  if(!fp)
    return;

  png_structp png = png_create_write_struct (PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png)
    return;

  png_infop info = png_create_info_struct (png);
  if (!info)
    return;

  if (setjmp(png_jmpbuf (png)))
    return;

  png_init_io (png, fp);

  // Output is 8bit depth, RGBA format.
  png_set_IHDR (
    png,
    info,
    width, height,
    8,
    PNG_COLOR_TYPE_GRAY,
    PNG_INTERLACE_NONE,
    PNG_COMPRESSION_TYPE_DEFAULT,
    PNG_FILTER_TYPE_DEFAULT
  );
  png_write_info (png, info);

  // To remove the alpha channel for PNG_COLOR_TYPE_RGB format,
  // Use png_set_filler().
  //png_set_filler(png, 0, PNG_FILLER_AFTER);

  if (!data)
    return;

  std::unique_ptr<png_bytep[]> row_pointers_helper (new png_bytep[height]);

  {
    png_byte *raw_ptr = data;
    for(int y = 0; y < height; y++) {
        row_pointers_helper[y] = raw_ptr;
        raw_ptr += width;
      }
  }

  png_write_image (png, row_pointers_helper.get ());
  png_write_end (png, NULL);

  fclose(fp);

  png_destroy_write_struct(&png, &info);
}
