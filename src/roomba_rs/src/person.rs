//! person_detector's processing - a port of roomba_perception/person_detector.py (and floor_grid.py's
//! overlay): YOLO (TensorRT, [`crate::trt`]) on the rectified left color image, distance from the
//! registered depth inside the box with a StereoBM fallback, ignore filtering (exclusion zones, line
//! of sight on the map; [`crate::scene_filter`]) and the annotated JPEG. ROS-free so the replay
//! example can drive it with recorded messages; the node is src/bin/person_detector.rs.
//!
//! Numerics follow the Python operation for operation (float32 where numpy computes in float32,
//! numpy's percentile interpolation, OpenCV's NMSBoxes / BGR2GRAY / addWeighted arithmetic, numpy's
//! fused matmul for the grid), so detections, distances and ignore decisions are identical
//! (tools/ab_person_detector_offline.py). OpenCV itself ([`crate::ocv`]) draws and encodes.

use std::collections::BTreeMap;
use std::time::Instant;

use crate::cv::round_half_even;
use crate::ocv::{self, Bgr, StereoBm, FILLED, FONT_HERSHEY_SIMPLEX, LINE_8, LINE_AA};
use crate::scene_filter::{GridLineOfSight, ZoneSet};
use crate::tf_buffer::{TfBuffer, TfError, Transform};
use crate::trt::TrtEngine;
use crate::zones::Level;

pub const COCO_NAMES: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
];

/// Python's name lookup: COCO name, or the id as text past the list.
pub fn class_name(cls: i64) -> String {
    usize::try_from(cls).ok().and_then(|c| COCO_NAMES.get(c)).map_or_else(|| cls.to_string(), |n| n.to_string())
}

/// Node parameters (names and defaults of the Python node).
#[derive(Clone, Debug)]
pub struct Params {
    pub engine_path: String,
    pub conf_threshold: f64,
    pub nms_threshold: f64,
    pub classes: Vec<i64>,
    pub jpeg_quality: i64,
    pub min_depth_pixels: i64,
    pub map_frame: String,
    pub ignore_zones: bool,
    pub line_of_sight: bool,
    pub occupied_threshold: i64,
    pub min_blocking_cells: i64,
    pub target_margin_m: f64,
    pub map_topic: String,
    pub require_localization: bool,
    pub floor_grid: bool,
    pub floor_grid_fallback_frame: String,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            engine_path: crate::zones::expand_user("~/models/yolo11n_384x640_fp16.engine").to_string_lossy().into_owned(),
            conf_threshold: 0.45,
            nms_threshold: 0.5,
            classes: vec![0],
            jpeg_quality: 80,
            min_depth_pixels: 15,
            map_frame: "map".into(),
            ignore_zones: true,
            line_of_sight: true,
            occupied_threshold: 65,
            min_blocking_cells: 2,
            target_margin_m: 0.35,
            map_topic: "/rtabmap/map".into(),
            require_localization: false,
            floor_grid: true,
            floor_grid_fallback_frame: "base_link".into(),
        }
    }
}

/// message_filters.TimeSynchronizer (Jazzy's Python implementation) for three inputs: messages are
/// keyed by stamp (a repeated stamp replaces the message), each queue keeps the `queue_size` newest
/// stamps, a set is emitted as soon as one stamp is in all queues (the oldest such stamp), and
/// emitting drops every older message.
pub struct TimeSync3<T> {
    queues: [BTreeMap<i64, T>; 3],
    queue_size: usize,
}

impl<T> TimeSync3<T> {
    pub fn new(queue_size: usize) -> Self {
        TimeSync3 { queues: [BTreeMap::new(), BTreeMap::new(), BTreeMap::new()], queue_size }
    }

    pub fn add(&mut self, input: usize, stamp_ns: i64, msg: T) -> Option<[T; 3]> {
        let q = &mut self.queues[input];
        q.insert(stamp_ns, msg);
        while q.len() > self.queue_size {
            q.pop_first();
        }
        let [a, b, c] = &self.queues;
        let t = *a.keys().find(|t| b.contains_key(t) && c.contains_key(t))?;
        let [a, b, c] = &mut self.queues;
        let msgs = [a.remove(&t)?, b.remove(&t)?, c.remove(&t)?];
        for q in &mut self.queues {
            q.retain(|s, _| *s >= t);
        }
        Some(msgs)
    }
}

/// A received image: `step` bytes per row (sensor_msgs/Image).
#[derive(Clone, Copy, Debug)]
pub struct ImageView<'a> {
    pub data: &'a [u8],
    pub width: usize,
    pub height: usize,
    pub step: usize,
}

impl ImageView<'_> {
    /// Whether the data holds height rows of width pixels of `bpp` bytes.
    pub fn valid(&self, bpp: usize) -> bool {
        self.height > 0 && self.step >= self.width * bpp && self.data.len() >= self.height * self.step
    }

    fn u16_at(&self, x: usize, y: usize) -> u16 {
        let i = y * self.step + 2 * x;
        u16::from_le_bytes([self.data[i], self.data[i + 1]])
    }
}

/// One synchronized set (left color bgr8, registered depth 16UC1 in mm, right mono8).
pub struct Frame<'a> {
    pub stamp_ns: i64,
    pub frame_id: &'a str,
    pub color: ImageView<'a>,
    pub depth: ImageView<'a>,
    pub right: ImageView<'a>,
}

/// tf2_ros Buffer.lookup_transform(target, source, time, timeout): waits up to `timeout_s` for the
/// transform (time 0 = latest).
pub trait TfLookup {
    fn lookup(&self, target: &str, source: &str, time_ns: i64, timeout_s: f64) -> Result<Transform, TfError>;
}

/// No waiting: nothing changes the buffer while the caller blocks (offline replay).
impl TfLookup for TfBuffer {
    fn lookup(&self, target: &str, source: &str, time_ns: i64, _timeout_s: f64) -> Result<Transform, TfError> {
        self.lookup_transform(target, source, time_ns).map(|t| t.transform)
    }
}

/// A kept detection: clipped pixel box, score and class id.
#[derive(Clone, Debug, PartialEq)]
pub struct Det {
    pub x1: i64,
    pub y1: i64,
    pub x2: i64,
    pub y2: i64,
    pub score: f32,
    pub cls: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Depth,
    Stereo,
    None,
}

/// vision_msgs/Detection2D content.
#[derive(Clone, Debug, PartialEq)]
pub struct Det2d {
    pub center: [f64; 2],
    pub size: [f64; 2],
    pub class_id: String,
    pub score: f64,
}

/// vision_msgs/Detection3D content (left optical frame, meters); `id` is the ignore reason.
#[derive(Clone, Debug, PartialEq)]
pub struct Det3d {
    pub position: [f64; 3],
    pub size: [f64; 3],
    pub class_id: String,
    pub score: f64,
    pub id: String,
}

pub const MARKER_SPHERE: i32 = 2;
pub const MARKER_TEXT_VIEW_FACING: i32 = 9;

/// visualization_msgs/Marker content (action ADD, orientation identity, lifetime 0.4 s).
#[derive(Clone, Debug, PartialEq)]
pub struct MarkerOut {
    pub ns: &'static str,
    pub id: i32,
    pub kind: i32,
    pub position: [f64; 3],
    pub scale: [f64; 3],
    pub color: [f32; 4],
    pub text: String,
}

pub const MARKER_LIFETIME_NS: u32 = 400_000_000;

/// Everything one frame publishes (header = the color image's).
#[derive(Clone, Debug, Default)]
pub struct FrameOutput {
    pub det2d: Vec<Det2d>,
    pub det3d: Vec<Det3d>,
    pub ignored: Vec<Det3d>,
    pub markers: Vec<MarkerOut>,
    /// Where each detection's distance came from (not published; for tests and the A/B).
    pub sources: Vec<Source>,
    /// The annotated image (before encoding) and its JPEG, when someone subscribes.
    pub annotated: Option<Bgr>,
    pub jpeg: Option<Vec<u8>>,
}

pub type Log = (Level, String);

// ---- detection ----

/// cv::Rect2d intersection area (OpenCV 4.6 `operator&=`, then area()).
fn intersection_area(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let empty = |r: &[f64; 4]| r[2] <= 0.0 || r[3] <= 0.0;
    if empty(a) || empty(b) {
        return 0.0;
    }
    let (xmin, xmax) = if a[0] < b[0] { (a, b) } else { (b, a) };
    let (ymin, ymax) = if a[1] < b[1] { (a, b) } else { (b, a) };
    if (xmin[0] < 0.0 && xmin[0] + xmin[2] < xmax[0]) || (ymin[1] < 0.0 && ymin[1] + ymin[3] < ymax[1]) {
        return 0.0;
    }
    let cv_min = |p: f64, q: f64| if q < p { q } else { p };
    let w = cv_min(xmin[2] - (xmax[0] - xmin[0]), xmax[2]);
    let h = cv_min(ymin[3] - (ymax[1] - ymin[1]), ymax[3]);
    if w <= 0.0 || h <= 0.0 {
        0.0
    } else {
        w * h
    }
}

/// dnn rectOverlap: 1 - (float)jaccardDistance(a, b) for cv::Rect2d (x, y, w, h).
fn rect_overlap(a: &[f64; 4], b: &[f64; 4]) -> f32 {
    let (aa, ab) = (a[2] * a[3], b[2] * b[3]);
    let jaccard = if aa + ab <= f64::EPSILON {
        0.0
    } else {
        let aab = intersection_area(a, b);
        1.0 - aab / (aa + ab - aab)
    };
    1.0 - jaccard as f32
}

/// cv2.dnn.NMSBoxes(boxes (x, y, w, h), scores, score_threshold, nms_threshold) with eta 1, top_k 0:
/// indices of the kept boxes, highest score first.
pub fn nms_boxes(boxes: &[[f64; 4]], scores: &[f32], score_threshold: f32, nms_threshold: f32) -> Vec<usize> {
    let mut order: Vec<(f32, usize)> = scores.iter().enumerate().filter(|(_, &s)| s > score_threshold).map(|(i, &s)| (s, i)).collect();
    // std::stable_sort with pair1.first > pair2.first
    order.sort_by(|p, q| if p.0 > q.0 { std::cmp::Ordering::Less } else if q.0 > p.0 { std::cmp::Ordering::Greater } else { std::cmp::Ordering::Equal });
    let mut kept: Vec<usize> = Vec::new();
    for (_, idx) in order {
        if kept.iter().all(|&k| rect_overlap(&boxes[idx], &boxes[k]) <= nms_threshold) {
            kept.push(idx);
        }
    }
    kept
}

/// _detect after inference: YOLO output (rows = 4 + classes, n anchors) to clipped pixel boxes of a
/// w x h image letterboxed with scale r and padding (px, py).
#[allow(clippy::too_many_arguments)]
pub fn postprocess(out: &[f32], rows: usize, n: usize, class_ids: &[i64], conf: f64, nms: f64, r: f64, px: i64, py: i64, w: usize, h: usize) -> Vec<Det> {
    let conf32 = conf as f32;
    let at = |row: usize, j: usize| out[row * n + j];
    let class_rows: Vec<(usize, i64)> = class_ids.iter().map(|&c| (4 + c as usize, c)).filter(|&(row, _)| row < rows).collect();
    if class_rows.is_empty() {
        return Vec::new();
    }
    let (r32, px32, py32) = (r as f32, px as f32, py as f32);
    let (mut boxes, mut scores, mut classes) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..n {
        // np.max / np.argmax (first maximum); a NaN score makes np.max NaN, which is never kept
        let mut best = at(class_rows[0].0, j);
        let mut arg = 0;
        let mut nan = best.is_nan();
        for (k, &(row, _)) in class_rows.iter().enumerate().skip(1) {
            let s = at(row, j);
            nan |= s.is_nan();
            if s > best {
                best = s;
                arg = k;
            }
        }
        if nan || !(best > conf32) {
            continue;
        }
        let (cx, cy, bw, bh) = (at(0, j), at(1, j), at(2, j), at(3, j));
        let x1 = (cx - bw / 2.0 - px32) / r32;
        let y1 = (cy - bh / 2.0 - py32) / r32;
        boxes.push([x1, y1, bw / r32, bh / r32]);
        scores.push(best);
        classes.push(class_rows[arg].1);
    }
    if boxes.is_empty() {
        return Vec::new();
    }
    let boxes64: Vec<[f64; 4]> = boxes.iter().map(|b| b.map(f64::from)).collect();
    let clip = |v: f32, hi: usize| v.clamp(0.0, (hi - 1) as f32) as i64;
    let mut dets = Vec::new();
    for i in nms_boxes(&boxes64, &scores, conf32, nms as f32) {
        let b = boxes[i];
        let (bx1, by1) = (clip(b[0], w), clip(b[1], h));
        let (bx2, by2) = (clip(b[0] + b[2], w), clip(b[1] + b[3], h));
        if bx2 - bx1 >= 4 && by2 - by1 >= 4 {
            dets.push(Det { x1: bx1, y1: by1, x2: bx2, y2: by2, score: scores[i], cls: classes[i] });
        }
    }
    dets
}

// ---- distance ----

/// numpy.percentile(values, q) (method 'linear') of a non-empty sample; `diff` is numpy's b - a in
/// the sample's dtype.
fn lerp_percentile<T: Copy + PartialOrd>(sorted: &[T], q: f64, as_f64: impl Fn(T) -> f64, diff: impl Fn(T, T) -> f64) -> f64 {
    let n = sorted.len();
    let virt = (n - 1) as f64 * (q / 100.0);
    let (prev, next) = if virt >= (n - 1) as f64 {
        (n - 1, n - 1)
    } else {
        let p = virt.floor() as usize;
        (p, p + 1)
    };
    let prev_index = if virt >= (n - 1) as f64 { -1.0 } else { prev as f64 };
    let gamma = virt - prev_index;
    let (a, b) = (sorted[prev], sorted[next]);
    let d = diff(b, a);
    if gamma >= 0.5 {
        as_f64(b) - d * (1.0 - gamma)
    } else {
        as_f64(a) + d * gamma
    }
}

pub fn percentile_u16(mut v: Vec<u16>, q: f64) -> f64 {
    v.sort_unstable();
    lerp_percentile(&v, q, f64::from, |b, a| f64::from(b.wrapping_sub(a)))
}

pub fn percentile_f32(mut v: Vec<f32>, q: f64) -> f64 {
    v.sort_unstable_by(|a, b| a.total_cmp(b));
    lerp_percentile(&v, q, f64::from, |b, a| f64::from(b - a))
}

/// Python slice start:stop of a sequence of length len (non-negative bounds).
fn py_range(start: i64, stop: i64, len: usize) -> std::ops::Range<usize> {
    let clamp = |i: i64| if i < 0 { (i + len as i64).max(0) as usize } else { (i as usize).min(len) };
    let (a, b) = (clamp(start), clamp(stop));
    a..b.max(a)
}

/// cv2.cvtColor(BGR2GRAY) of rows [r0, r1) of a bgr8 image (OpenCV's 15-bit fixed point).
fn gray_rows(color: &ImageView, r0: usize, r1: usize) -> Vec<u8> {
    let w = color.width;
    let mut out = Vec::with_capacity((r1 - r0) * w);
    for y in r0..r1 {
        let row = &color.data[y * color.step..y * color.step + 3 * w];
        out.extend(row.chunks_exact(3).map(|p| ((p[0] as u32 * 3735 + p[1] as u32 * 19235 + p[2] as u32 * 9798 + 16384) >> 15) as u8));
    }
    out
}

// ---- floor grid (floor_grid.py) ----

/// floor_grid.quat_matrix
pub fn quat_matrix(q: [f64; 4]) -> [[f64; 3]; 3] {
    let [x, y, z, w] = q;
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

/// numpy.arange(start, stop, step) for floats.
fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let len = ((stop - start) / step).ceil().max(0.0) as usize;
    let mut v = Vec::with_capacity(len);
    if len > 0 {
        v.push(start);
    }
    if len > 1 {
        v.push(start + step);
        let delta = v[1] - v[0];
        v.extend((2..len).map(|i| start + i as f64 * delta));
    }
    v
}

/// floor_grid.grid_lines: lines (each a list of floor points) and major-line flags.
pub fn grid_lines(center: [f64; 2], spacing: f64, extent: f64, step: f64) -> (Vec<Vec<[f64; 3]>>, Vec<bool>) {
    let n = (extent / spacing) as i64;
    let x0 = (center[0] / spacing).floor() * spacing;
    let y0 = (center[1] / spacing).floor() * spacing;
    let s = arange(-extent, extent + step / 2.0, step);
    let major = |v: f64| (v - round_half_even(v)).abs() < 1e-3;
    let (mut lines, mut flags) = (Vec::new(), Vec::new());
    for i in -n..=n {
        let o = i as f64 * spacing;
        let x = x0 + o;
        lines.push(s.iter().map(|&si| [x, y0 + si, 0.0]).collect());
        flags.push(major(x));
        let y = y0 + o;
        lines.push(s.iter().map(|&si| [x0 + si, y, 0.0]).collect());
        flags.push(major(y));
    }
    (lines, flags)
}

/// numpy's matmul accumulation (reference BLAS / its own loop, compiled with fused multiply-adds).
fn dot3_fused(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[2].mul_add(b[2], a[1].mul_add(b[1], a[0] * b[0]))
}

/// cv2.addWeighted(src1, alpha, src2, beta, 0.0, dst=src2) for 8-bit data (OpenCV's float32 fused
/// multiply-adds and round-half-even).
pub fn add_weighted(src1: &[u8], alpha: f64, dst: &mut [u8], beta: f64) {
    let (a, b) = (alpha as f32, beta as f32);
    let blend = |s: u8, d: u8| (s as f32).mul_add(a, (d as f32).mul_add(b, 0.0)).round_ties_even().clamp(0.0, 255.0) as u8;
    // Most pixels are not under a grid line (s == d): look those up.
    let same: [u8; 256] = std::array::from_fn(|p| blend(p as u8, p as u8));
    for (d, &s) in dst.iter_mut().zip(src1) {
        *d = if s == *d { same[s as usize] } else { blend(s, *d) };
    }
}

pub struct GridStyle {
    pub spacing: f64,
    pub max_range: f64,
    pub min_depth: f64,
    pub color: [i32; 3],
    pub alpha: f64,
}

impl Default for GridStyle {
    fn default() -> Self {
        GridStyle { spacing: 0.25, max_range: 2.5, min_depth: 0.15, color: [255, 220, 60], alpha: 0.55 }
    }
}

/// floor_grid.draw_floor_grid: the z = 0 plane of a frame whose points map into the camera optical
/// frame as rot @ p + trans. Returns the number of drawn segments.
#[allow(clippy::too_many_arguments)]
pub fn draw_floor_grid(img: &mut Bgr, rot: [[f64; 3]; 3], trans: [f64; 3], fx: f64, fy: f64, cx: f64, cy: f64, style: &GridStyle) -> usize {
    let neg_col = |i: usize| [-rot[0][i], -rot[1][i], -rot[2][i]];
    let cam = [dot3_fused(neg_col(0), trans), dot3_fused(neg_col(1), trans), dot3_fused(neg_col(2), trans)];
    let (lines, major) = grid_lines([cam[0], cam[1]], style.spacing, style.max_range, 0.1);
    let (w, h) = (img.width as f64, img.height as f64);
    let mut overlay = img.clone();
    let mut drawn = 0;
    for (line, &is_major) in lines.iter().zip(&major) {
        let uv_ok: Vec<(f64, f64, bool)> = line
            .iter()
            .map(|&p| {
                let q = [dot3_fused(p, rot[0]) + trans[0], dot3_fused(p, rot[1]) + trans[1], dot3_fused(p, rot[2]) + trans[2]];
                let (u, v) = (fx * q[0] / q[2] + cx, fy * q[1] / q[2] + cy);
                let (dx, dy) = (p[0] - cam[0], p[1] - cam[1]);
                let dist = (dx * dx + dy * dy).sqrt();
                let ok = q[2] > style.min_depth && dist < style.max_range && u.abs() < 4.0 * w && v.abs() < 4.0 * h;
                (u, v, ok)
            })
            .collect();
        let thick = if is_major { 2 } else { 1 };
        // runs of consecutive valid samples, each drawn as a polyline
        let mut i = 0;
        while i < uv_ok.len() {
            if !uv_ok[i].2 {
                i += 1;
                continue;
            }
            let start = i;
            while i < uv_ok.len() && uv_ok[i].2 {
                i += 1;
            }
            if i - start >= 2 {
                let poly: Vec<(i32, i32)> = uv_ok[start..i].iter().map(|&(u, v, _)| (round_half_even(u) as i32, round_half_even(v) as i32)).collect();
                ocv::polyline(&mut overlay, &poly, style.color, thick, LINE_AA);
                drawn += i - start - 1;
            }
        }
    }
    add_weighted(&overlay.data, style.alpha, &mut img.data, 1.0 - style.alpha);
    drawn
}

// ---- the detector ----

/// Python's f'{x:.2f}' / f'{x:.1f}' (Rust rounds the exact binary value half to even, like Python).
fn fixed(x: f64, places: usize) -> String {
    format!("{x:.places$}")
}

pub struct Detector {
    pub params: Params,
    engine: TrtEngine,
    in_w: usize,
    in_h: usize,
    out_rows: usize,
    out_n: usize,
    stereo: StereoBm,
    left_p: Option<[f64; 12]>,
    baseline: Option<f64>,
    zones: ZoneSet,
    los: GridLineOfSight,
    localized: bool,
    grid_on: bool,
    tf_warned: bool,
    tf_retry_after: f64,
    n_frames: u64,
    t_infer: f64,
    t_total: f64,
}

impl Detector {
    /// Loads the engine (TrtYolo). Returns the detector and the startup log line.
    pub fn new(params: Params) -> Result<(Self, Log), String> {
        if let Some(c) = params.classes.iter().find(|&&c| !(0..80).contains(&c)) {
            return Err(format!("class id {c} is not a COCO class (0..79)"));
        }
        let engine = TrtEngine::load(&params.engine_path)?;
        let (inp, out) = (engine.input_info().dims.clone(), engine.output_info().dims.clone());
        if inp.len() != 4 || inp[1] != 3 || out.len() != 3 || out[1] < 5 || params.classes.iter().any(|&c| c + 4 >= out[1]) {
            return Err(format!("{}: expected input (1, 3, H, W) and output (1, 4+C, N) with the configured classes, got {inp:?} and {out:?}", params.engine_path));
        }
        let (in_h, in_w) = (inp[2] as usize, inp[3] as usize);
        let names: Vec<String> = params.classes.iter().map(|&c| format!("'{}'", COCO_NAMES[c as usize])).collect();
        let log = (Level::Info, format!("Loaded {} (input {in_w}x{in_h}); classes [{}]", params.engine_path, names.join(", ")));
        let stereo = StereoBm::new(160, 15).ok_or("StereoBM_create failed")?;
        let los = GridLineOfSight::new(params.occupied_threshold, params.min_blocking_cells);
        let d = Detector {
            localized: !params.require_localization,
            grid_on: params.floor_grid,
            engine,
            in_w,
            in_h,
            out_rows: out[1] as usize,
            out_n: out[2] as usize,
            stereo,
            left_p: None,
            baseline: None,
            zones: ZoneSet::default(),
            los,
            tf_warned: false,
            tf_retry_after: 0.0,
            n_frames: 0,
            t_infer: 0.0,
            t_total: 0.0,
            params,
        };
        Ok((d, log))
    }

    pub fn engine(&self) -> &TrtEngine {
        &self.engine
    }

    pub fn on_left_info(&mut self, p: [f64; 12]) {
        self.left_p = Some(p);
    }

    pub fn on_right_info(&mut self, p: [f64; 12]) {
        let fx = p[0];
        if fx > 0.0 && p[3] != 0.0 {
            self.baseline = Some(-p[3] / fx);
        }
    }

    pub fn on_grid_toggle(&mut self, on: bool) -> Log {
        self.grid_on = on;
        (Level::Info, format!("Floor grid {}", if on { "on" } else { "off" }))
    }

    pub fn on_zones(&mut self, json: &str) -> Log {
        match self.zones.load_json(json) {
            Ok(()) => (Level::Info, format!("{} ignore zone(s) active", self.zones.zones.len())),
            Err(e) => (Level::Error, format!("Bad ignore zone message: {e}")),
        }
    }

    /// A new occupancy grid (nav_msgs/OccupancyGrid fields).
    #[allow(clippy::too_many_arguments)]
    pub fn on_map(&mut self, width: usize, height: usize, resolution: f32, origin: [f64; 2], orientation: [f64; 4], frame: &str, data: &[i8]) {
        if data.len() == width * height {
            self.los.update(width, height, resolution, origin, orientation, frame, data);
        }
    }

    pub fn on_localized(&mut self) -> Option<Log> {
        if self.localized {
            return None;
        }
        self.localized = true;
        Some((Level::Info, "Localized in saved map: exclusion zones and line of sight active".into()))
    }

    /// The 30 s statistics line (None without frames); resets the counters.
    pub fn stats(&mut self) -> Option<Log> {
        let n = self.n_frames;
        let log = (n > 0).then(|| {
            let per = |t: f64| fixed(1000.0 * t / n as f64, 1);
            (Level::Info, format!("{} Hz, inference {} ms, total {} ms/frame", fixed(n as f64 / 30.0, 1), per(self.t_infer), per(self.t_total)))
        });
        (self.n_frames, self.t_infer, self.t_total) = (0, 0.0, 0.0);
        log
    }

    /// TrtYolo.preprocess: letterbox into the input tensor; returns (r, px, py).
    fn preprocess(&mut self, bgr: &ImageView) -> Result<(f64, i64, i64), String> {
        let (w, h) = (bgr.width, bgr.height);
        let (in_w, in_h) = (self.in_w, self.in_h);
        let r = (in_w as f64 / w as f64).min(in_h as f64 / h as f64);
        let (nw, nh) = (round_half_even(w as f64 * r) as usize, round_half_even(h as f64 * r) as usize);
        let resized = if (nw, nh) != (w, h) { Some(ocv::resize_linear(bgr.data, w, h, bgr.step, 3, nw, nh).ok_or("cv2.resize failed")?) } else { None };
        let (src, step) = match &resized {
            Some(v) => (&v[..], nw * 3),
            None => (bgr.data, bgr.step),
        };
        let (px, py) = ((in_w as i64 - nw as i64).div_euclid(2), (in_h as i64 - nh as i64).div_euclid(2));
        // canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0), gray 114 padding
        let scale = (1.0f64 / 255.0) as f32;
        let lut: [f32; 256] = std::array::from_fn(|v| v as f32 * scale);
        let pad = lut[114];
        let plane = in_w * in_h;
        let (x0, x1) = (px as usize, px as usize + nw); // r <= in_w / w, so 0 <= px and x1 <= in_w
        let inp = self.engine.input_mut();
        let (red, rest) = inp.split_at_mut(plane);
        let (green, blue) = rest.split_at_mut(plane);
        for y in 0..in_h {
            let span = y * in_w..(y + 1) * in_w;
            let (ro, go, bo) = (&mut red[span.clone()], &mut green[span.clone()], &mut blue[span]);
            let sy = y as i64 - py;
            if !(0..nh as i64).contains(&sy) {
                for o in [ro, go, bo] {
                    o.fill(pad);
                }
                continue;
            }
            let row = &src[sy as usize * step..sy as usize * step + nw * 3];
            for o in [&mut *ro, &mut *go, &mut *bo] {
                o[..x0].fill(pad);
                o[x1..].fill(pad);
            }
            for (((p, r), g), b) in row.chunks_exact(3).zip(&mut ro[x0..x1]).zip(&mut go[x0..x1]).zip(&mut bo[x0..x1]) {
                *r = lut[p[2] as usize];
                *g = lut[p[1] as usize];
                *b = lut[p[0] as usize];
            }
        }
        Ok((r, px, py))
    }

    fn detect(&mut self, bgr: &ImageView) -> Result<Vec<Det>, String> {
        let (r, px, py) = self.preprocess(bgr)?;
        let t0 = Instant::now();
        self.engine.infer()?;
        self.t_infer += t0.elapsed().as_secs_f64();
        let p = &self.params;
        Ok(postprocess(self.engine.output(), self.out_rows, self.out_n, &p.classes, p.conf_threshold, p.nms_threshold, r, px, py, bgr.width, bgr.height))
    }

    /// _distance: (z in m, source, (u, v)) from the torso region of the box.
    fn distance(&mut self, f: &Frame, det: &Det, fx: f64) -> (Option<f64>, Source, (f64, f64)) {
        let (x1, y1, x2, y2) = (det.x1, det.y1, det.x2, det.y2);
        let (w, h) = ((x2 - x1) as f64, (y2 - y1) as f64);
        let (rx1, mut rx2) = ((x1 as f64 + 0.25 * w) as i64, (x2 as f64 - 0.25 * w) as i64);
        let (ry1, mut ry2) = ((y1 as f64 + 0.2 * h) as i64, (y2 as f64 - 0.35 * h) as i64);
        ry2 = ry2.max(ry1 + 2);
        rx2 = rx2.max(rx1 + 2);
        let uv = ((rx1 + rx2) as f64 / 2.0, (ry1 + ry2) as f64 / 2.0);
        let min_px = self.params.min_depth_pixels;

        let d = &f.depth;
        let (rows, cols) = (py_range(ry1, ry2, d.height), py_range(rx1, rx2, d.width));
        let mut valid = Vec::new();
        for y in rows {
            for x in cols.clone() {
                let v = d.u16_at(x, y);
                if v > 0 {
                    valid.push(v);
                }
            }
        }
        if !valid.is_empty() && valid.len() as i64 >= min_px {
            return (Some(percentile_u16(valid, 30.0) / 1000.0), Source::Depth, uv);
        }

        if let Some(baseline) = self.baseline.filter(|&b| b != 0.0 && fx > 0.0) {
            // Stereo fallback: block-match the box's row band (StereoBM needs full-width rows).
            let height = f.color.height as i64;
            let (b1, b2) = (y1.max(0), height.min(y2));
            if b2 - b1 >= 16 && f.right.width == f.color.width && f.right.height == f.color.height {
                let (b1u, b2u) = (b1 as usize, b2 as usize);
                let gray = gray_rows(&f.color, b1u, b2u);
                let right = &f.right.data[b1u * f.right.step..];
                let w = f.color.width;
                if let Ok(disp) = self.stereo.compute(&gray, w, right, f.right.step, b2u - b1u, w) {
                    let mut ds = Vec::new();
                    for y in py_range(ry1 - b1, ry2 - b1, b2u - b1u) {
                        for x in py_range(rx1, rx2, w) {
                            let v = disp[y * w + x] as f32 / 16.0;
                            if v > 1.0 {
                                ds.push(v);
                            }
                        }
                    }
                    if !ds.is_empty() && ds.len() as i64 >= min_px {
                        return (Some(fx * baseline / percentile_f32(ds, 70.0)), Source::Stereo, uv);
                    }
                }
            }
        }
        (None, Source::None, uv)
    }

    /// _map_transform: map <- camera optical frame at the image stamp (latest as fallback).
    fn map_transform(&mut self, f: &Frame, tf: &dyn TfLookup, clock: &dyn Fn() -> f64, logs: &mut Vec<Log>) -> Option<Transform> {
        if !(self.params.ignore_zones || self.params.line_of_sight) || !self.localized {
            return None;
        }
        if clock() < self.tf_retry_after {
            return None;
        }
        let mut err = None;
        for (stamp, wait) in [(f.stamp_ns, 0.03), (0, 0.0)] {
            match tf.lookup(&self.params.map_frame, f.frame_id, stamp, wait) {
                Ok(t) => return Some(t),
                Err(e) => err = Some(e),
            }
        }
        // Back off so a missing map doesn't add lookup latency to every frame.
        self.tf_retry_after = clock() + 1.0;
        if !self.tf_warned {
            let e = err.map(|e| e.message).unwrap_or_default();
            logs.push((Level::Warn, format!("No {} transform yet, not filtering: {e}", self.params.map_frame)));
            self.tf_warned = true;
        }
        None
    }

    /// _ignore_reason for a detection at `p` (camera optical frame).
    fn ignore_reason(&self, tf_map: Option<&Transform>, p: [f64; 3]) -> Option<String> {
        let tf = tf_map?;
        let [px, py, _] = tf.apply(p);
        if self.params.ignore_zones {
            if let Some(zone) = self.zones.containing(px, py).filter(|z| !z.is_empty()) {
                return Some(format!("in {zone}"));
            }
        }
        if self.params.line_of_sight && self.los.ready() && self.los.frame.as_deref() == Some(self.params.map_frame.as_str()) {
            let [cx, cy, _] = tf.apply([0.0, 0.0, 0.0]);
            if self.los.blocked([cx, cy], [px, py], 0.15, self.params.target_margin_m) as i64 >= self.los.min_blocking_cells {
                return Some("behind obstacle".into());
            }
        }
        None
    }

    /// _draw_grid: the floor grid of the first available ground frame, at the image stamp.
    #[allow(clippy::too_many_arguments)]
    fn draw_grid(&self, img: &mut Bgr, f: &Frame, fx: f64, fy: f64, cx: f64, cy: f64, tf: &dyn TfLookup) {
        let frames = [&self.params.map_frame, &self.params.floor_grid_fallback_frame];
        for frame in frames.into_iter().filter(|s| !s.is_empty()) {
            for (stamp, wait) in [(f.stamp_ns, 0.02), (0, 0.0)] {
                if let Ok(t) = tf.lookup(f.frame_id, frame, stamp, wait) {
                    draw_floor_grid(img, quat_matrix(t.rotation), t.translation, fx, fy, cx, cy, &GridStyle::default());
                    return;
                }
            }
        }
    }

    /// _on_frames. None until the left camera_info arrived (the frame is dropped). `annotate`: someone
    /// subscribes to the annotated image; `clock`: time.monotonic(). Err: inference failed.
    pub fn process(&mut self, f: &Frame, annotate: bool, tf: &dyn TfLookup, clock: &dyn Fn() -> f64, logs: &mut Vec<Log>) -> Result<Option<FrameOutput>, String> {
        let Some(p) = self.left_p else { return Ok(None) };
        if !f.color.valid(3) || !f.depth.valid(2) || !f.right.valid(1) {
            logs.push((Level::Warn, "Skipping a frame with inconsistent image sizes".into()));
            return Ok(None);
        }
        let t_start = Instant::now();
        let (fx, fy, cx, cy) = (p[0], p[5], p[2], p[6]);
        let dets = self.detect(&f.color)?;
        let mut out = FrameOutput::default();
        let mut annotated = annotate.then(|| {
            let c = &f.color;
            let mut data = Vec::with_capacity(c.width * c.height * 3);
            for y in 0..c.height {
                data.extend_from_slice(&c.data[y * c.step..y * c.step + c.width * 3]);
            }
            Bgr { width: c.width, height: c.height, data }
        });
        if let Some(img) = annotated.as_mut().filter(|_| self.grid_on) {
            self.draw_grid(img, f, fx, fy, cx, cy, tf);
        }
        let tf_map = if dets.is_empty() { None } else { self.map_transform(f, tf, clock, logs) };

        for (k, det) in dets.iter().enumerate() {
            let name = class_name(det.cls);
            let score = det.score as f64;
            let (z, source, (u, v)) = self.distance(f, det, fx);
            out.sources.push(source);
            let (x1, y1, x2, y2) = (det.x1, det.y1, det.x2, det.y2);
            out.det2d.push(Det2d {
                center: [(x1 + x2) as f64 / 2.0, (y1 + y2) as f64 / 2.0],
                size: [(x2 - x1) as f64, (y2 - y1) as f64],
                class_id: name.clone(),
                score,
            });
            let mut reason = None;
            if let Some(z) = z {
                let (x, y) = ((u - cx) * z / fx, (v - cy) * z / fy);
                reason = self.ignore_reason(tf_map.as_ref(), [x, y, z]);
                let d3 = Det3d {
                    position: [x, y, z],
                    size: [(x2 - x1) as f64 * z / fx, (y2 - y1) as f64 * z / fy, 0.3],
                    class_id: name.clone(),
                    score,
                    id: reason.clone().unwrap_or_default(),
                };
                if reason.is_some() { out.ignored.push(d3) } else { out.det3d.push(d3) }
                let mut text = format!("{name} {} m", fixed(z, 2));
                if let Some(r) = &reason {
                    text += &format!(" ({r})");
                }
                let ignored = reason.is_some();
                out.markers.push(MarkerOut {
                    ns: "people",
                    id: 2 * k as i32,
                    kind: MARKER_SPHERE,
                    position: [x, y, z],
                    scale: [0.25; 3],
                    color: if ignored { [0.5, 0.5, 0.5, 0.6] } else { [0.1, 0.9, 0.2, 0.8] },
                    text: String::new(),
                });
                out.markers.push(MarkerOut {
                    ns: "people_labels",
                    id: 2 * k as i32 + 1,
                    kind: MARKER_TEXT_VIEW_FACING,
                    position: [x, y - 0.35, z],
                    scale: [0.0, 0.0, 0.18],
                    color: [1.0; 4],
                    text,
                });
            }

            if let Some(img) = annotated.as_mut() {
                let color = match (&reason, source) {
                    (Some(_), _) => [110, 110, 110],
                    (None, Source::Depth) => [0, 220, 0],
                    (None, Source::Stereo) => [0, 200, 255],
                    (None, Source::None) => [200, 200, 200],
                };
                let mut label = match z {
                    Some(z) => format!("{name} {} m", fixed(z, 2)),
                    None => format!("{name} ? m"),
                };
                if let Some(r) = &reason {
                    label += &format!(" - ignored: {r}");
                }
                let (x1, y1, x2, y2) = (x1 as i32, y1 as i32, x2 as i32, y2 as i32);
                ocv::rectangle(img, (x1, y1), (x2, y2), color, 2, LINE_8);
                let ((tw, th), _) = ocv::text_size(&label, FONT_HERSHEY_SIMPLEX, 0.6, 2);
                let ty = y1.max(th + 8);
                ocv::rectangle(img, (x1, ty - th - 8), (x1 + tw + 8, ty), color, FILLED, LINE_8);
                let text_color = if reason.is_some() { [255, 255, 255] } else { [0, 0, 0] };
                ocv::put_text(img, &label, (x1 + 4, ty - 5), FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, LINE_AA);
            }
        }

        if let Some(img) = annotated {
            out.jpeg = ocv::imencode_jpeg(&img, self.params.jpeg_quality as i32);
            out.annotated = Some(img);
        }
        self.n_frames += 1;
        self.t_total += t_start.elapsed().as_secs_f64();
        Ok(Some(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_sync_matches_message_filters() {
        let mut s = TimeSync3::new(10);
        assert!(s.add(0, 10, "c10").is_none());
        assert!(s.add(1, 10, "d10").is_none());
        assert!(s.add(0, 20, "c20").is_none());
        assert!(s.add(1, 20, "d20").is_none());
        assert!(s.add(2, 5, "r5").is_none());
        // 20 completes before 10: emitting 20 drops everything older, including the unmatched 10s
        assert_eq!(s.add(2, 20, "r20"), Some(["c20", "d20", "r20"]));
        assert!(s.add(2, 10, "r10").is_none());
        assert!(s.queues.iter().all(|q| q.keys().all(|&k| k == 10)));
        // queue size: the oldest stamp is evicted
        let mut s = TimeSync3::new(2);
        for t in 1..=3 {
            s.add(0, t, t);
        }
        assert_eq!(s.queues[0].keys().copied().collect::<Vec<_>>(), vec![2, 3]);
    }

    #[test]
    fn nms_keeps_overlapping_fractional_boxes_like_rect2d() {
        // cv2.dnn.NMSBoxes([[0, 0, 1.9, 1.9], [0.9, 0.9, 1.9, 1.9]], [0.9, 0.8], 0.5, 0.5) == [0, 1]
        assert_eq!(nms_boxes(&[[0.0, 0.0, 1.9, 1.9], [0.9, 0.9, 1.9, 1.9]], &[0.9, 0.8], 0.5, 0.5), vec![0, 1]);
        // ... and [[0, 0, 10, 10], [0.4, 0.4, 10, 10]] -> [0]; ties keep index order
        assert_eq!(nms_boxes(&[[0.0, 0.0, 10.0, 10.0], [0.4, 0.4, 10.0, 10.0]], &[0.9, 0.8], 0.5, 0.5), vec![0]);
        assert_eq!(nms_boxes(&[[0.0, 0.0, 1.0, 1.0], [5.0, 5.0, 1.0, 1.0], [9.0, 9.0, 1.0, 1.0]], &[0.6, 0.7, 0.6], 0.5, 0.5), vec![1, 0, 2]);
    }

    #[test]
    fn percentiles_match_numpy() {
        // repr() of numpy 1.26's results
        assert_eq!(percentile_u16(vec![5, 1, 3, 2, 4], 30.0), 2.2);
        assert_eq!(percentile_u16(vec![1000, 1001, 1003, 1010], 30.0), 1000.9);
        assert_eq!(percentile_u16(vec![7], 30.0), 7.0);
        assert_eq!(percentile_f32(vec![9.0, 1.5, 3.25], 70.0), 5.549999999999999);
        assert_eq!(percentile_f32(vec![2.1, 1.7], 70.0), 1.9799999475479126);
    }

    #[test]
    fn arange_like_numpy() {
        let s = arange(-2.5, 2.5 + 0.05, 0.1);
        assert_eq!(s.len(), 51);
        assert_eq!(s[0], -2.5);
        assert!((s[50] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn postprocess_letterbox_and_classes() {
        // two anchors, classes 0 (person) and 1; anchor 0 is a person at (100, 62) in the letterbox
        let (rows, n) = (6, 2);
        let mut out = vec![0f32; rows * n];
        let set = |o: &mut Vec<f32>, r: usize, j: usize, v: f32| o[r * n + j] = v;
        set(&mut out, 0, 0, 100.0);
        set(&mut out, 1, 0, 62.0);
        set(&mut out, 2, 0, 20.0);
        set(&mut out, 3, 0, 40.0);
        set(&mut out, 4, 0, 0.9);
        set(&mut out, 5, 1, 0.8);
        let d = postprocess(&out, rows, n, &[0], 0.45, 0.5, 1.0, 0, 12, 640, 360);
        assert_eq!(d, vec![Det { x1: 90, y1: 30, x2: 110, y2: 70, score: 0.9, cls: 0 }]);
    }

    #[test]
    fn add_weighted_rounds_like_opencv() {
        let mut dst = vec![0u8, 255, 100, 7];
        add_weighted(&[255, 0, 100, 200], 0.55, &mut dst, 0.45);
        assert_eq!(dst, vec![140, 115, 100, 113]);
    }

    #[test]
    fn python_float_format() {
        assert_eq!(fixed(1.125, 2), "1.12");
        assert_eq!(fixed(1.375, 2), "1.38");
        assert_eq!(fixed(0.25, 1), "0.2");
        assert_eq!(fixed(2.675, 2), "2.67");
    }
}
