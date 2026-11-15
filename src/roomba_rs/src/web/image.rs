//! Camera preview and map images for the web server, matching roomba_web/server.py's OpenCV 4.6
//! calls: cv2.resize(..., INTER_AREA) reproduced operation by operation (the same float32
//! weights, accumulation order and fused multiply-adds as libopencv_imgproc on aarch64, so the
//! pixels are identical), and cv2.imencode('.jpg' / '.png') through codecs.c, which drives the
//! system libjpeg-turbo and libpng with OpenCV's settings (identical bytes).

use std::os::raw::{c_int, c_uchar};

extern "C" {
    fn rs_jpeg_encode_gray(px: *const c_uchar, w: c_int, h: c_int, stride: usize, q: c_int, out: *mut *mut c_uchar, len: *mut usize) -> c_int;
    fn rs_png_encode_bgra(px: *const c_uchar, w: c_int, h: c_int, stride: usize, out: *mut *mut c_uchar, len: *mut usize) -> c_int;
    fn rs_codec_free(p: *mut std::ffi::c_void);
}

fn take_buffer(rc: c_int, out: *mut c_uchar, len: usize) -> Option<Vec<u8>> {
    if rc != 0 || out.is_null() {
        return None;
    }
    // SAFETY: on success the encoder returned a malloc'ed buffer of `len` bytes; copy, then free it.
    unsafe {
        let v = std::slice::from_raw_parts(out, len).to_vec();
        rs_codec_free(out.cast());
        Some(v)
    }
}

/// cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality]) for a single-channel image.
pub fn encode_jpeg_gray(pixels: &[u8], width: usize, height: usize, stride: usize, quality: i32) -> Option<Vec<u8>> {
    if width == 0 || height == 0 || stride < width || pixels.len() < stride * (height - 1) + width {
        return None;
    }
    let (mut out, mut len) = (std::ptr::null_mut(), 0usize);
    // SAFETY: the buffer holds `height` rows of `width` bytes at `stride` (checked above).
    let rc = unsafe { rs_jpeg_encode_gray(pixels.as_ptr(), width as c_int, height as c_int, stride, quality, &mut out, &mut len) };
    take_buffer(rc, out, len)
}

/// cv2.imencode('.png', img) for a 4-channel (BGRA) image of `width * 4`-byte rows.
pub fn encode_png_bgra(pixels: &[u8], width: usize, height: usize) -> Option<Vec<u8>> {
    if width == 0 || height == 0 || pixels.len() < width * height * 4 {
        return None;
    }
    let (mut out, mut len) = (std::ptr::null_mut(), 0usize);
    // SAFETY: the buffer holds `height` rows of `width * 4` bytes (checked above).
    let rc = unsafe { rs_png_encode_bgra(pixels.as_ptr(), width as c_int, height as c_int, width * 4, &mut out, &mut len) };
    take_buffer(rc, out, len)
}

/// saturate_cast<uchar>(float): round half to even, clamp to 0..255.
fn sat_u8(v: f32) -> u8 {
    v.round_ties_even().clamp(0.0, 255.0) as u8
}

struct Alpha {
    di: usize,
    si: usize,
    alpha: f32,
}

/// computeResizeAreaTab (imgproc/src/resize.cpp) for one channel.
fn area_tab(ssize: usize, dsize: usize, scale: f64) -> Vec<Alpha> {
    let mut tab = Vec::with_capacity(ssize * 2);
    for dx in 0..dsize {
        let fsx1 = dx as f64 * scale;
        let fsx2 = fsx1 + scale;
        let cell_width = scale.min(ssize as f64 - fsx1);
        let mut sx1 = fsx1.ceil() as i64;
        let mut sx2 = fsx2.floor() as i64;
        sx2 = sx2.min(ssize as i64 - 1);
        sx1 = sx1.min(sx2);
        if sx1 as f64 - fsx1 > 1e-3 {
            tab.push(Alpha { di: dx, si: (sx1 - 1) as usize, alpha: ((sx1 as f64 - fsx1) / cell_width) as f32 });
        }
        for sx in sx1..sx2 {
            tab.push(Alpha { di: dx, si: sx as usize, alpha: (1.0 / cell_width) as f32 });
        }
        if fsx2 - sx2 as f64 > 1e-3 {
            let w = (fsx2 - sx2 as f64).min(1.0).min(cell_width);
            tab.push(Alpha { di: dx, si: sx2 as usize, alpha: (w / cell_width) as f32 });
        }
    }
    tab
}

/// cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA) for a single-channel 8-bit image that
/// shrinks (dw <= w, dh <= h). Returns `dw * dh` bytes.
pub fn resize_area(src: &[u8], w: usize, h: usize, stride: usize, dw: usize, dh: usize) -> Vec<u8> {
    let inv_x = dw as f64 / w as f64;
    let inv_y = dh as f64 / h as f64;
    let (scale_x, scale_y) = (1.0 / inv_x, 1.0 / inv_y);
    // saturate_cast<int>(double) rounds
    let (iscale_x, iscale_y) = (scale_x.round_ties_even() as i64, scale_y.round_ties_even() as i64);
    let is_area_fast = (scale_x - iscale_x as f64).abs() < f64::EPSILON && (scale_y - iscale_y as f64).abs() < f64::EPSILON;
    if is_area_fast {
        return resize_area_fast(src, w, h, stride, dw, dh, iscale_x as usize, iscale_y as usize);
    }
    let xtab = area_tab(w, dw, scale_x);
    let ytab = area_tab(h, dh, scale_y);
    let mut dst = vec![0u8; dw * dh];
    let mut buf = vec![0f32; dw];
    let mut sum = vec![0f32; dw];
    let mut prev_dy = ytab.first().map_or(0, |t| t.di);
    for t in &ytab {
        let (beta, dy) = (t.alpha, t.di);
        let row = &src[t.si * stride..];
        buf.iter_mut().for_each(|b| *b = 0.0);
        for x in &xtab {
            // buf[dxn] += S[si]*alpha, contracted to a fused multiply-add by GCC
            buf[x.di] = (row[x.si] as f32).mul_add(x.alpha, buf[x.di]);
        }
        if dy != prev_dy {
            let d = &mut dst[prev_dy * dw..(prev_dy + 1) * dw];
            for dx in 0..dw {
                d[dx] = sat_u8(sum[dx]);
                sum[dx] = beta * buf[dx];
            }
            prev_dy = dy;
        } else {
            for dx in 0..dw {
                sum[dx] = beta.mul_add(buf[dx], sum[dx]);
            }
        }
    }
    let d = &mut dst[prev_dy * dw..(prev_dy + 1) * dw];
    for dx in 0..dw {
        d[dx] = sat_u8(sum[dx]);
    }
    dst
}

/// resizeAreaFast_ for integer scale factors (2x2 blocks round as (a+b+c+d+2)>>2).
#[allow(clippy::too_many_arguments)]
fn resize_area_fast(src: &[u8], w: usize, h: usize, stride: usize, dw: usize, dh: usize, sx: usize, sy: usize) -> Vec<u8> {
    let mut dst = vec![0u8; dw * dh];
    let area = sx * sy;
    let scale = 1.0f32 / area as f32;
    let dwidth1 = w / sx;
    for dy in 0..dh {
        let sy0 = dy * sy;
        let d = &mut dst[dy * dw..(dy + 1) * dw];
        if sy0 >= h {
            continue;
        }
        let full = if sy0 + sy <= h { dwidth1 } else { 0 };
        let mut dx = 0;
        if sx == 2 && sy == 2 {
            let (r0, r1) = (&src[sy0 * stride..], &src[(sy0 + 1) * stride..]);
            while dx < full {
                let i = dx * 2;
                d[dx] = ((r0[i] as u32 + r0[i + 1] as u32 + r1[i] as u32 + r1[i + 1] as u32 + 2) >> 2) as u8;
                dx += 1;
            }
        }
        while dx < full {
            let mut total: i32 = 0;
            for yy in 0..sy {
                let r = &src[(sy0 + yy) * stride + dx * sx..];
                for xx in 0..sx {
                    total += r[xx] as i32;
                }
            }
            d[dx] = sat_u8(total as f32 * scale);
            dx += 1;
        }
        while dx < dw {
            let (mut total, mut count) = (0i32, 0i32);
            let sx0 = dx * sx;
            for yy in 0..sy {
                if sy0 + yy >= h {
                    break;
                }
                for xx in 0..sx {
                    if sx0 + xx >= w {
                        break;
                    }
                    total += src[(sy0 + yy) * stride + sx0 + xx] as i32;
                    count += 1;
                }
            }
            d[dx] = if count > 0 { sat_u8(total as f32 / count as f32) } else { 0 };
            dx += 1;
        }
    }
    dst
}

/// server.py _on_preview: a mono8 frame (rows of `step` bytes) shrunk with INTER_AREA to
/// `preview_width` when wider (height int(h * preview_width / w)), then JPEG quality 70.
pub fn preview_jpeg(data: &[u8], width: usize, height: usize, step: usize, preview_width: usize) -> Option<Vec<u8>> {
    if width == 0 || height == 0 || step < width || data.len() != height * step {
        return None;
    }
    if preview_width > 0 && width > preview_width {
        let h = ((height * preview_width) as f64 / width as f64) as usize;
        if h == 0 {
            return None;
        }
        let small = resize_area(data, width, height, step, preview_width, h);
        encode_jpeg_gray(&small, preview_width, h, preview_width, 70)
    } else {
        encode_jpeg_gray(data, width, height, step, 70)
    }
}

/// server.py _on_map: unknown (< 0) transparent, free (0..64) and occupied (>= 65) colours, rows
/// flipped so the top row is the largest y, encoded as PNG.
pub fn map_png(width: usize, height: usize, data: &[i8]) -> Option<Vec<u8>> {
    if width == 0 || height == 0 || data.len() != width * height {
        return None;
    }
    const FREE: [u8; 4] = [70, 60, 52, 255];
    const OCCUPIED: [u8; 4] = [240, 232, 225, 255];
    let mut img = vec![0u8; width * height * 4];
    for (out_row, grid_row) in img.chunks_exact_mut(width * 4).zip(data.chunks_exact(width).rev()) {
        for (px, &v) in out_row.chunks_exact_mut(4).zip(grid_row) {
            if v >= 65 {
                px.copy_from_slice(&OCCUPIED);
            } else if v >= 0 {
                px.copy_from_slice(&FREE);
            }
        }
    }
    encode_png_bgra(&img, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn area_resize_averages_blocks() {
        // 4x2 -> 2x1 (integer 2x2 path) and 3x3 -> 2x2 (fractional weights)
        let img = [10u8, 20, 30, 40, 50, 60, 70, 80];
        assert_eq!(resize_area(&img, 4, 2, 4, 2, 1), vec![35, 55]);
        let flat = [100u8; 9];
        assert_eq!(resize_area(&flat, 3, 3, 3, 2, 2), vec![100; 4]);
        let ramp: Vec<u8> = (0..9).map(|i| (i * 30) as u8).collect();
        let r = resize_area(&ramp, 3, 3, 3, 2, 2);
        assert!(r[0] < r[1] && r[1] < r[3], "{r:?}");
    }

    #[test]
    fn encoders_produce_valid_files() {
        let gray: Vec<u8> = (0..64 * 48).map(|i| (i % 251) as u8).collect();
        let jpg = encode_jpeg_gray(&gray, 64, 48, 64, 70).unwrap();
        assert_eq!(&jpg[..3], &[0xff, 0xd8, 0xff]);
        assert_eq!(&jpg[jpg.len() - 2..], &[0xff, 0xd9]);
        let png = map_png(3, 2, &[-1, 0, 100, 64, 65, -128]).unwrap();
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        assert!(map_png(0, 2, &[]).is_none());
        assert!(map_png(2, 2, &[0; 3]).is_none());
        assert!(preview_jpeg(&gray, 64, 48, 64, 32).is_some());
    }
}
