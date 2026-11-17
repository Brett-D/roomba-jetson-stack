/* Image encoders for the Rust web server (roomba_rs::web::image), written like OpenCV 4.6's
 * imgcodecs (grfmt_jpeg.cpp / grfmt_png.cpp) drives the same system libjpeg-turbo and libpng, so
 * cv2.imencode('.jpg', gray, [IMWRITE_JPEG_QUALITY, q]) and cv2.imencode('.png', bgra) produce the
 * same bytes. Errors longjmp back here (libjpeg's default error handler would exit the process). */
#include <setjmp.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <jpeglib.h>
#include <png.h>
#include <zlib.h>

struct rs_jpeg_error {
  struct jpeg_error_mgr pub;
  jmp_buf jump;
};

static void rs_jpeg_error_exit(j_common_ptr cinfo) {
  struct rs_jpeg_error *err = (struct rs_jpeg_error *)cinfo->err;
  longjmp(err->jump, 1);
}

struct rs_jpeg_out {
  unsigned char *data;
  unsigned long len;
};

/* JpegEncoder::write for a single-channel 8-bit image: grayscale, jpeg_set_defaults, no restart
 * interval, jpeg_set_quality(quality, TRUE), baseline (no progressive / optimized Huffman).
 * Returns 0 and a malloc'ed buffer (free with rs_codec_free), or -1. */
int rs_jpeg_encode_gray(const unsigned char *pixels, int width, int height, size_t stride, int quality,
                        unsigned char **out, size_t *out_len) {
  struct jpeg_compress_struct cinfo;
  struct rs_jpeg_error jerr;
  const int q = quality < 0 ? 0 : (quality > 100 ? 100 : quality);
  struct rs_jpeg_out *res = calloc(1, sizeof(*res));
  if (!res) return -1;
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = rs_jpeg_error_exit;
  if (setjmp(jerr.jump)) {
    jpeg_destroy_compress(&cinfo);
    free(res->data);
    free(res);
    return -1;
  }
  jpeg_create_compress(&cinfo);
  jpeg_mem_dest(&cinfo, &res->data, &res->len);
  cinfo.image_width = (JDIMENSION)width;
  cinfo.image_height = (JDIMENSION)height;
  cinfo.input_components = 1;
  cinfo.in_color_space = JCS_GRAYSCALE;
  jpeg_set_defaults(&cinfo);
  cinfo.restart_interval = 0;
  jpeg_set_quality(&cinfo, q, TRUE);
  jpeg_start_compress(&cinfo, TRUE);
  for (int y = 0; y < height; y++) {
    JSAMPROW row = (JSAMPROW)(pixels + (size_t)y * stride);
    jpeg_write_scanlines(&cinfo, &row, 1);
  }
  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);
  *out = res->data;
  *out_len = res->len;
  free(res);
  return 0;
}

/* Like rs_jpeg_encode_gray for 8-bit RGB rows (the depth view, which has no OpenCV twin). */
int rs_jpeg_encode_rgb(const unsigned char *pixels, int width, int height, size_t stride, int quality,
                       unsigned char **out, size_t *out_len) {
  struct jpeg_compress_struct cinfo;
  struct rs_jpeg_error jerr;
  const int q = quality < 0 ? 0 : (quality > 100 ? 100 : quality);
  struct rs_jpeg_out *res = calloc(1, sizeof(*res));
  if (!res) return -1;
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = rs_jpeg_error_exit;
  if (setjmp(jerr.jump)) {
    jpeg_destroy_compress(&cinfo);
    free(res->data);
    free(res);
    return -1;
  }
  jpeg_create_compress(&cinfo);
  jpeg_mem_dest(&cinfo, &res->data, &res->len);
  cinfo.image_width = (JDIMENSION)width;
  cinfo.image_height = (JDIMENSION)height;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, q, TRUE);
  jpeg_start_compress(&cinfo, TRUE);
  for (int y = 0; y < height; y++) {
    JSAMPROW row = (JSAMPROW)(pixels + (size_t)y * stride);
    jpeg_write_scanlines(&cinfo, &row, 1);
  }
  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);
  *out = res->data;
  *out_len = res->len;
  free(res);
  return 0;
}
struct rs_png_out {
  unsigned char *data;
  size_t len, cap;
};

static void rs_png_write(png_structp png, png_bytep data, png_size_t n) {
  struct rs_png_out *o = (struct rs_png_out *)png_get_io_ptr(png);
  if (o->len + n > o->cap) {
    size_t cap = o->cap ? o->cap : 4096;
    while (cap < o->len + n) cap *= 2;
    unsigned char *grown = realloc(o->data, cap);
    if (!grown) png_error(png, "out of memory");
    o->data = grown;
    o->cap = cap;
  }
  memcpy(o->data + o->len, data, n);
  o->len += n;
}

static void rs_png_flush(png_structp png) { (void)png; }

/* PngEncoder::write for an 8-bit BGRA image with default parameters: SUB filter, zlib level 1
 * (Z_BEST_SPEED), Z_RLE strategy, RGBA colour type with png_set_bgr.
 * Returns 0 and a malloc'ed buffer (free with rs_codec_free), or -1. */
int rs_png_encode_bgra(const unsigned char *pixels, int width, int height, size_t stride, unsigned char **out,
                       size_t *out_len) {
  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png) return -1;
  png_infop info = png_create_info_struct(png);
  struct rs_png_out *res = calloc(1, sizeof(*res));
  png_bytep *rows = malloc(sizeof(png_bytep) * (size_t)(height > 0 ? height : 1));
  if (!info || !res || !rows) {
    png_destroy_write_struct(&png, info ? &info : NULL);
    free(res);
    free(rows);
    return -1;
  }
  if (setjmp(png_jmpbuf(png))) {
    png_destroy_write_struct(&png, &info);
    free(res->data);
    free(res);
    free(rows);
    return -1;
  }
  png_set_write_fn(png, res, rs_png_write, rs_png_flush);
  png_set_filter(png, PNG_FILTER_TYPE_BASE, PNG_FILTER_SUB);
  png_set_compression_level(png, Z_BEST_SPEED);
  png_set_compression_strategy(png, Z_RLE);
  png_set_IHDR(png, info, (png_uint_32)width, (png_uint_32)height, 8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png, info);
  png_set_bgr(png);
  png_set_swap(png);
  for (int y = 0; y < height; y++) rows[y] = (png_bytep)(pixels + (size_t)y * stride);
  png_write_image(png, rows);
  png_write_end(png, info);
  png_destroy_write_struct(&png, &info);
  *out = res->data;
  *out_len = res->len;
  free(res);
  free(rows);
  return 0;
}

void rs_codec_free(void *p) { free(p); }
