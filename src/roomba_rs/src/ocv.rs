//! The few OpenCV functions the person detector needs that are not worth re-implementing: StereoBM,
//! drawing (rectangles, anti-aliased polylines, Hershey text), resize and JPEG encoding, through
//! native/person_shim.cpp (OpenCV 4 has no C API). Images are 8-bit, row-major with a row step.

use std::ffi::{c_char, c_int, c_void, CStr, CString};

extern "C" {
    fn rr_stereo_bm_create(num_disparities: c_int, block_size: c_int) -> *mut c_void;
    fn rr_stereo_bm_free(bm: *mut c_void);
    fn rr_stereo_bm_compute(
        bm: *mut c_void,
        left: *const u8,
        left_step: usize,
        right: *const u8,
        right_step: usize,
        rows: c_int,
        cols: c_int,
        disp: *mut i16,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
    fn rr_resize_linear(
        src: *const u8,
        src_rows: c_int,
        src_cols: c_int,
        src_step: usize,
        channels: c_int,
        dst: *mut u8,
        dst_rows: c_int,
        dst_cols: c_int,
        dst_step: usize,
    ) -> c_int;
    fn rr_rectangle(
        img: *mut u8,
        rows: c_int,
        cols: c_int,
        step: usize,
        x1: c_int,
        y1: c_int,
        x2: c_int,
        y2: c_int,
        b: c_int,
        g: c_int,
        r: c_int,
        thickness: c_int,
        line_type: c_int,
    ) -> c_int;
    fn rr_polyline(
        img: *mut u8,
        rows: c_int,
        cols: c_int,
        step: usize,
        xy: *const i32,
        n: c_int,
        b: c_int,
        g: c_int,
        r: c_int,
        thickness: c_int,
        line_type: c_int,
    ) -> c_int;
    fn rr_text_size(text: *const c_char, font: c_int, scale: f64, thickness: c_int, w: *mut c_int, h: *mut c_int, baseline: *mut c_int) -> c_int;
    fn rr_put_text(
        img: *mut u8,
        rows: c_int,
        cols: c_int,
        step: usize,
        text: *const c_char,
        x: c_int,
        y: c_int,
        font: c_int,
        scale: f64,
        b: c_int,
        g: c_int,
        r: c_int,
        thickness: c_int,
        line_type: c_int,
    ) -> c_int;
    fn rr_imencode_jpeg(img: *const u8, rows: c_int, cols: c_int, step: usize, quality: c_int, out: *mut *mut u8, len: *mut usize) -> c_int;
    fn rr_free(p: *mut c_void);
}

pub const FONT_HERSHEY_SIMPLEX: i32 = 0;
pub const LINE_8: i32 = 8;
pub const LINE_AA: i32 = 16;
/// cv2.FILLED
pub const FILLED: i32 = -1;

/// A contiguous BGR (8UC3) image.
#[derive(Clone, Debug, PartialEq)]
pub struct Bgr {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl Bgr {
    pub fn step(&self) -> usize {
        self.width * 3
    }

    fn args(&mut self) -> (*mut u8, c_int, c_int, usize) {
        let step = self.step();
        (self.data.as_mut_ptr(), self.height as c_int, self.width as c_int, step)
    }
}

/// A C string for OpenCV's text functions (an interior NUL ends the text, as in C).
fn c_text(text: &str) -> CString {
    CString::new(text.split('\0').next().unwrap_or("")).unwrap_or_default()
}

/// cv2.rectangle(img, p1, p2, color (B, G, R), thickness, line_type)
pub fn rectangle(img: &mut Bgr, p1: (i32, i32), p2: (i32, i32), color: [i32; 3], thickness: i32, line_type: i32) {
    let (data, rows, cols, step) = img.args();
    // SAFETY: data is rows x step bytes.
    unsafe { rr_rectangle(data, rows, cols, step, p1.0, p1.1, p2.0, p2.1, color[0], color[1], color[2], thickness, line_type) };
}

/// cv2.polylines(img, [pts], False, color, thickness, line_type)
pub fn polyline(img: &mut Bgr, pts: &[(i32, i32)], color: [i32; 3], thickness: i32, line_type: i32) {
    let xy: Vec<i32> = pts.iter().flat_map(|&(x, y)| [x, y]).collect();
    let (data, rows, cols, step) = img.args();
    unsafe { rr_polyline(data, rows, cols, step, xy.as_ptr(), pts.len() as c_int, color[0], color[1], color[2], thickness, line_type) };
}

/// cv2.getTextSize: ((width, height), baseline)
pub fn text_size(text: &str, font: i32, scale: f64, thickness: i32) -> ((i32, i32), i32) {
    let t = c_text(text);
    let (mut w, mut h, mut base) = (0, 0, 0);
    unsafe { rr_text_size(t.as_ptr(), font, scale, thickness, &mut w, &mut h, &mut base) };
    ((w, h), base)
}

/// cv2.putText(img, text, org, font, scale, color, thickness, line_type)
pub fn put_text(img: &mut Bgr, text: &str, org: (i32, i32), font: i32, scale: f64, color: [i32; 3], thickness: i32, line_type: i32) {
    let t = c_text(text);
    let (data, rows, cols, step) = img.args();
    unsafe { rr_put_text(data, rows, cols, step, t.as_ptr(), org.0, org.1, font, scale, color[0], color[1], color[2], thickness, line_type) };
}

/// cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
pub fn imencode_jpeg(img: &Bgr, quality: i32) -> Option<Vec<u8>> {
    let (mut out, mut len) = (std::ptr::null_mut(), 0usize);
    // SAFETY: the image is height x step bytes; on success out is a malloc'ed buffer of len bytes.
    let rc = unsafe { rr_imencode_jpeg(img.data.as_ptr(), img.height as c_int, img.width as c_int, img.step(), quality, &mut out, &mut len) };
    if rc != 0 || out.is_null() {
        return None;
    }
    let bytes = unsafe { std::slice::from_raw_parts(out, len) }.to_vec();
    unsafe { rr_free(out.cast()) };
    Some(bytes)
}

/// cv2.resize(src, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR) of an 8-bit image with
/// `channels` channels and row step `src_step`; returns a contiguous image.
pub fn resize_linear(src: &[u8], src_w: usize, src_h: usize, src_step: usize, channels: usize, dst_w: usize, dst_h: usize) -> Option<Vec<u8>> {
    if src.len() < src_h.saturating_sub(1) * src_step + src_w * channels || !(1..=4).contains(&channels) {
        return None;
    }
    let mut dst = vec![0u8; dst_w * dst_h * channels];
    let rc = unsafe {
        rr_resize_linear(
            src.as_ptr(),
            src_h as c_int,
            src_w as c_int,
            src_step,
            channels as c_int,
            dst.as_mut_ptr(),
            dst_h as c_int,
            dst_w as c_int,
            dst_w * channels,
        )
    };
    (rc == 0).then_some(dst)
}

/// cv2.StereoBM_create(numDisparities, blockSize)
pub struct StereoBm {
    raw: *mut c_void,
}

// SAFETY: used by one owner at a time.
unsafe impl Send for StereoBm {}

impl StereoBm {
    pub fn new(num_disparities: i32, block_size: i32) -> Option<Self> {
        let raw = unsafe { rr_stereo_bm_create(num_disparities, block_size) };
        (!raw.is_null()).then_some(StereoBm { raw })
    }

    /// stereo.compute(left, right): CV_16S disparities (x16) of two mono8 images of rows x cols.
    pub fn compute(&mut self, left: &[u8], left_step: usize, right: &[u8], right_step: usize, rows: usize, cols: usize) -> Result<Vec<i16>, String> {
        let need = |step: usize| rows.saturating_sub(1) * step + cols;
        if rows == 0 || cols == 0 || left.len() < need(left_step) || right.len() < need(right_step) {
            return Err("image too small".into());
        }
        let mut disp = vec![0i16; rows * cols];
        let mut err = vec![0u8; 512];
        let rc = unsafe {
            rr_stereo_bm_compute(
                self.raw,
                left.as_ptr(),
                left_step,
                right.as_ptr(),
                right_step,
                rows as c_int,
                cols as c_int,
                disp.as_mut_ptr(),
                err.as_mut_ptr().cast(),
                err.len(),
            )
        };
        if rc != 0 {
            return Err(CStr::from_bytes_until_nul(&err).map(|s| s.to_string_lossy().into_owned()).unwrap_or_default());
        }
        Ok(disp)
    }
}

impl Drop for StereoBm {
    fn drop(&mut self) {
        unsafe { rr_stereo_bm_free(self.raw) }
    }
}
