//! tf2-equivalent transform buffer for the Rust nodes (rclrs has no tf2).
//!
//! A port of tf2::BufferCore as shipped with ROS 2 Jazzy (tf2 0.36): every child frame keeps a
//! time-sorted history of its transform to its parent, and entries older than the cache time (tf2's
//! default 10 s) behind that frame's newest are dropped; static transforms (/tf_static) never
//! expire. `lookup_transform` walks from both frames up to their common ancestor and evaluates each
//! link at the requested time - translation interpolated linearly, rotation by slerp - and never
//! extrapolates. Time 0 means "latest": the newest time every link of the chain has, as in tf2.
//! Errors carry tf2's exception type and message text.
//!
//! The arithmetic follows tf2's LinearMath operation by operation - including the quaternion ->
//! rotation matrix -> quaternion round trip tf2 applies on insert and on output, and the fused
//! multiply-adds GCC compiled into libtf2 - so results are bit-identical to Python/C++ tf2
//! (tools/ab_tf_buffer.py compares them on recorded TF).
//!
//! The ROS side (subscriptions to /tf and /tf_static) is in [`crate::tf_listener`]. Times are
//! nanoseconds since the epoch (`sec * 1e9 + nanosec` of a header stamp).

use std::collections::{HashMap, VecDeque};
use std::fmt;

pub type Vec3 = [f64; 3];
pub type Quat = [f64; 4];

/// tf2::BUFFER_CORE_DEFAULT_CACHE_TIME (10 s).
pub const DEFAULT_CACHE_TIME_NS: i64 = 10_000_000_000;
const MAX_GRAPH_DEPTH: u32 = 1000;
const QUATERNION_NORMALIZATION_TOLERANCE: f64 = 10e-3;

/// Rigid transform: `rotation` (x, y, z, w) then `translation`, mapping child-frame coordinates
/// into the parent frame (geometry_msgs/Transform).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
}

impl Transform {
    pub const IDENTITY: Transform = Transform { translation: [0.0; 3], rotation: [0.0, 0.0, 0.0, 1.0] };

    /// Applies the transform to a point (scene_filter.apply_transform).
    pub fn apply(&self, p: Vec3) -> Vec3 {
        let r = crate::geometry::quat_rotate(self.rotation, p);
        [r[0] + self.translation[0], r[1] + self.translation[1], r[2] + self.translation[2]]
    }
}

/// Result of [`TfBuffer::lookup_transform`] (geometry_msgs/TransformStamped): maps `child_frame_id`
/// (the source frame) coordinates into `frame_id` (the target frame) at `stamp_ns`.
#[derive(Clone, Debug, PartialEq)]
pub struct StampedTransform {
    pub stamp_ns: i64,
    pub frame_id: String,
    pub child_frame_id: String,
    pub transform: Transform,
}

/// tf2's exception types. Python's tf2 reports all extrapolation kinds as ExtrapolationException.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    InvalidArgument,
    Lookup,
    Connectivity,
    Extrapolation,
    BackwardExtrapolation,
    ForwardExtrapolation,
    NoDataForExtrapolation,
}

impl ErrorKind {
    /// The tf2_py / tf2_ros exception class raised for this error.
    pub fn python_name(self) -> &'static str {
        match self {
            ErrorKind::InvalidArgument => "InvalidArgumentException",
            ErrorKind::Lookup => "LookupException",
            ErrorKind::Connectivity => "ConnectivityException",
            _ => "ExtrapolationException",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TfError {
    pub kind: ErrorKind,
    pub message: String,
}

impl TfError {
    fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        TfError { kind, message: message.into() }
    }
}

impl fmt::Display for TfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for TfError {}

// ---- tf2 LinearMath, operation for operation ----
//
// ROS's libtf2 is built by GCC with floating-point contraction (its default): in a sum of products
// `p1 + p2 + p3 ...` the first product is fused with the second (rounded) one into a fused
// multiply-add, and each further product is fused into the running sum. The same fusions are
// written out here (checked against the disassembly of libtf2.so 0.36.23 on aarch64), which makes
// the results bit-identical to tf2's instead of differing in the last bits - differences that
// acos() near 1 in slerp can amplify to 1e-7.

fn fma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

/// tf2 `Quaternion * Quaternion` (and `*=`).
fn qmul(a: Quat, b: Quat) -> Quat {
    [
        fma(-a[2], b[1], fma(a[1], b[2], fma(a[3], b[0], a[0] * b[3]))),
        fma(-a[0], b[2], fma(a[2], b[0], fma(a[3], b[1], a[1] * b[3]))),
        fma(-a[1], b[0], fma(a[0], b[1], fma(a[3], b[2], a[2] * b[3]))),
        fma(-a[2], b[2], fma(-a[1], b[1], fma(a[3], b[3], -(a[0] * b[0])))),
    ]
}

/// tf2 `Quaternion * Vector3`.
fn qvmul(q: Quat, w: Vec3) -> Quat {
    [
        fma(-q[2], w[1], fma(q[3], w[0], q[1] * w[2])),
        fma(-q[0], w[2], fma(q[3], w[1], q[2] * w[0])),
        fma(-q[1], w[0], fma(q[3], w[2], q[0] * w[1])),
        fma(-q[2], w[2], fma(-q[0], w[0], -(q[1] * w[1]))),
    ]
}

/// tf2 Quaternion::length2.
fn length2(q: Quat) -> f64 {
    fma(q[3], q[3], fma(q[2], q[2], fma(q[0], q[0], q[1] * q[1])))
}

/// tf2 Quaternion::inverse (the conjugate, not normalized).
fn inverse(q: Quat) -> Quat {
    [-q[0], -q[1], -q[2], q[3]]
}

/// tf2::quatRotate.
fn quat_rotate(q: Quat, v: Vec3) -> Vec3 {
    let r = qmul(qvmul(q, v), inverse(q));
    [r[0], r[1], r[2]]
}

fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn neg(a: Vec3) -> Vec3 {
    [-a[0], -a[1], -a[2]]
}

fn qdot(a: Quat, b: Quat) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

/// tf2Acos: acos with the argument clamped to [-1, 1].
fn acos_clamped(x: f64) -> f64 {
    let x = if x < -1.0 {
        -1.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    };
    x.acos()
}

fn angle_shortest_path(a: Quat, b: Quat) -> f64 {
    let s = (length2(a) * length2(b)).sqrt();
    if qdot(a, b) < 0.0 {
        acos_clamped(qdot(a, [-b[0], -b[1], -b[2], -b[3]]) / s) * 2.0
    } else {
        acos_clamped(qdot(a, b) / s) * 2.0
    }
}

/// tf2 Quaternion::slerp.
fn slerp(a: Quat, b: Quat, t: f64) -> Quat {
    let theta = angle_shortest_path(a, b) / 2.0;
    if theta == 0.0 {
        return a;
    }
    let d = 1.0 / theta.sin();
    let s0 = ((1.0 - t) * theta).sin();
    let s1 = (t * theta).sin();
    if qdot(a, b) < 0.0 {
        [
            fma(a[0], s0, -(b[0] * s1)) * d,
            fma(a[1], s0, -(b[1] * s1)) * d,
            fma(a[2], s0, -(b[2] * s1)) * d,
            fma(a[3], s0, -(b[3] * s1)) * d,
        ]
    } else {
        [
            fma(a[0], s0, b[0] * s1) * d,
            fma(a[1], s0, b[1] * s1) * d,
            fma(a[2], s0, b[2] * s1) * d,
            fma(a[3], s0, b[3] * s1) * d,
        ]
    }
}

/// Matrix3x3::setRotation then getRotation: what a quaternion becomes when stored in a
/// tf2::Transform (normalized; sign chosen by the matrix's trace).
pub fn quat_through_matrix(q: Quat) -> Quat {
    let [x, y, z, w] = q;
    let s = 2.0 / length2(q);
    let (xs, ys, zs) = (x * s, y * s, z * s);
    let (xy, xz, yz) = (x * ys, x * zs, y * zs);
    let (yy, zz) = (y * ys, z * zs);
    // wx = w * xs etc. and xx = x * xs are fused into the sums that use them.
    let m = [
        [1.0 - (yy + zz), fma(-w, zs, xy), fma(w, ys, xz)],
        [fma(w, zs, xy), 1.0 - fma(x, xs, zz), fma(-w, xs, yz)],
        [fma(-w, ys, xz), fma(w, xs, yz), 1.0 - fma(x, xs, yy)],
    ];
    let trace = m[0][0] + m[1][1] + m[2][2];
    let mut t = [0.0; 4];
    if trace > 0.0 {
        let mut s = (trace + 1.0).sqrt();
        t[3] = s * 0.5;
        s = 0.5 / s;
        t[0] = (m[2][1] - m[1][2]) * s;
        t[1] = (m[0][2] - m[2][0]) * s;
        t[2] = (m[1][0] - m[0][1]) * s;
    } else {
        let i = if m[0][0] < m[1][1] {
            if m[1][1] < m[2][2] {
                2
            } else {
                1
            }
        } else if m[0][0] < m[2][2] {
            2
        } else {
            0
        };
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        let mut s = (m[i][i] - m[j][j] - m[k][k] + 1.0).sqrt();
        t[i] = s * 0.5;
        s = 0.5 / s;
        t[3] = (m[k][j] - m[j][k]) * s;
        t[j] = (m[j][i] + m[i][j]) * s;
        t[k] = (m[k][i] + m[i][k]) * s;
    }
    t
}

/// tf2::displayTimePoint: seconds with 6 decimals.
fn display_time(ns: i64) -> String {
    let nsec = ns % 1_000_000_000;
    let sec = (ns - nsec) / 1_000_000_000;
    format!("{:.6}", sec as i32 as f64 + 1e-9 * (nsec as i32 as f64))
}

/// printf("%f").
fn c_f(x: f64) -> String {
    if x.is_nan() {
        if x.is_sign_negative() { "-nan" } else { "nan" }.into()
    } else if x.is_infinite() {
        if x < 0.0 { "-inf" } else { "inf" }.into()
    } else {
        format!("{x:.6}")
    }
}

// ---- caches ----

#[derive(Clone, Copy, Debug, PartialEq)]
struct Storage {
    rotation: Quat,
    translation: Vec3,
    stamp: i64,
    parent: u32,
    child: u32,
}

enum Cache {
    Static(Storage),
    /// Newest first, like tf2's TimeCache list.
    Dynamic(VecDeque<Storage>),
}

type CacheError = (ErrorKind, String);

impl Cache {
    fn latest(&self) -> i64 {
        match self {
            Cache::Static(_) => 0,
            Cache::Dynamic(s) => s.front().map_or(0, |e| e.stamp),
        }
    }

    fn latest_time_and_parent(&self) -> (i64, u32) {
        match self {
            Cache::Static(s) => (0, s.parent),
            Cache::Dynamic(s) => s.front().map_or((0, 0), |e| (e.stamp, e.parent)),
        }
    }

    fn insert(&mut self, d: Storage, max_age: i64) -> bool {
        let s = match self {
            Cache::Static(s) => {
                *s = d;
                return true;
            }
            Cache::Dynamic(s) => s,
        };
        let latest = s.front().map_or(0, |e| e.stamp);
        // TimeCache::insertData prunes first, against the newest entry before this one.
        while s.back().is_some_and(|b| b.stamp.saturating_add(max_age) < latest) {
            s.pop_back();
        }
        if !s.is_empty() && d.stamp < latest.saturating_sub(max_age) {
            return false;
        }
        let pos = s.iter().position(|e| e.stamp <= d.stamp).unwrap_or(s.len());
        let duplicate = s.iter().skip(pos).take_while(|e| e.stamp == d.stamp).any(|e| *e == d);
        if !duplicate {
            s.insert(pos, d);
        }
        true
    }

    /// TimeCache::getData / StaticCache::getData.
    fn get_data(&self, time: i64) -> Result<Storage, CacheError> {
        let s = match self {
            Cache::Static(s) => return Ok(Storage { stamp: time, ..*s }),
            Cache::Dynamic(s) => s,
        };
        let Some(front) = s.front() else {
            return Err((ErrorKind::NoDataForExtrapolation, String::new()));
        };
        if time == 0 {
            return Ok(*front);
        }
        if s.len() == 1 {
            return if front.stamp == time {
                Ok(*front)
            } else {
                Err((
                    ErrorKind::NoDataForExtrapolation,
                    format!(
                        "Lookup would require extrapolation at time {}, but only time {} is in the buffer",
                        display_time(time),
                        display_time(front.stamp)
                    ),
                ))
            };
        }
        let back = s[s.len() - 1];
        if time == front.stamp {
            return Ok(*front);
        } else if time == back.stamp {
            return Ok(back);
        } else if time > front.stamp {
            return Err((
                ErrorKind::ForwardExtrapolation,
                format!(
                    "Lookup would require extrapolation into the future.  Requested time {} but the latest data is at time {}",
                    display_time(time),
                    display_time(front.stamp)
                ),
            ));
        } else if time < back.stamp {
            return Err((
                ErrorKind::BackwardExtrapolation,
                format!(
                    "Lookup would require extrapolation into the past.  Requested time {} but the earliest data is at time {}",
                    display_time(time),
                    display_time(back.stamp)
                ),
            ));
        }
        let i = s.partition_point(|e| e.stamp > time);
        let (one, two) = (s[i], s[i - 1]); // older, newer
        if one.parent != two.parent {
            return Ok(one);
        }
        if two.stamp == one.stamp {
            return Ok(two);
        }
        let ratio = (time - one.stamp) as f64 / (two.stamp - one.stamp) as f64;
        let k = 1.0 - ratio;
        let (a, b) = (one.translation, two.translation);
        Ok(Storage {
            translation: [fma(k, a[0], ratio * b[0]), fma(k, a[1], ratio * b[1]), fma(k, a[2], ratio * b[2])],
            rotation: slerp(one.rotation, two.rotation, ratio),
            ..one
        })
    }
}

/// tf2's TransformAccum.
struct Accum {
    source_vec: Vec3,
    source_quat: Quat,
    target_vec: Vec3,
    target_quat: Quat,
}

enum WalkEnd {
    Identity,
    TargetParentOfSource,
    SourceParentOfTarget,
    FullPath,
}

impl Accum {
    fn new() -> Self {
        Accum { source_vec: [0.0; 3], source_quat: [0.0, 0.0, 0.0, 1.0], target_vec: [0.0; 3], target_quat: [0.0, 0.0, 0.0, 1.0] }
    }

    fn accum(&mut self, st: &Storage, source: bool) {
        let (v, q) = if source {
            (&mut self.source_vec, &mut self.source_quat)
        } else {
            (&mut self.target_vec, &mut self.target_quat)
        };
        *v = add(quat_rotate(st.rotation, *v), st.translation);
        *q = qmul(st.rotation, *q);
    }

    fn finalize(&self, end: WalkEnd) -> (Vec3, Quat) {
        match end {
            WalkEnd::Identity => ([0.0; 3], [0.0, 0.0, 0.0, 1.0]),
            WalkEnd::TargetParentOfSource => (self.source_vec, self.source_quat),
            WalkEnd::SourceParentOfTarget => {
                let inv = inverse(self.target_quat);
                (quat_rotate(inv, neg(self.target_vec)), inv)
            }
            WalkEnd::FullPath => {
                let inv = inverse(self.target_quat);
                let inv_vec = quat_rotate(inv, neg(self.target_vec));
                (add(quat_rotate(inv, self.source_vec), inv_vec), qmul(inv, self.source_quat))
            }
        }
    }
}

fn strip_slash(frame: &str) -> &str {
    frame.strip_prefix('/').unwrap_or(frame)
}

/// The transform tree: feed it every transform received on /tf (`is_static` false) and
/// /tf_static (`is_static` true), then look transforms up.
pub struct TfBuffer {
    cache_time_ns: i64,
    ids: HashMap<String, u32>,
    names: Vec<String>,
    frames: Vec<Option<Cache>>,
}

impl Default for TfBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl TfBuffer {
    /// A buffer keeping 10 s of history per frame (tf2's default).
    pub fn new() -> Self {
        Self::with_cache_time_ns(DEFAULT_CACHE_TIME_NS)
    }

    pub fn with_cache_time_ns(cache_time_ns: i64) -> Self {
        TfBuffer {
            cache_time_ns,
            ids: HashMap::from([("NO_PARENT".to_string(), 0)]),
            names: vec!["NO_PARENT".into()],
            frames: vec![None],
        }
    }

    pub fn cache_time_ns(&self) -> i64 {
        self.cache_time_ns
    }

    /// Drops all dynamic history (tf2 BufferCore::clear; static transforms stay).
    pub fn clear(&mut self) {
        for c in self.frames.iter_mut().skip(1).flatten() {
            if let Cache::Dynamic(s) = c {
                s.clear();
            }
        }
    }

    fn id(&self, frame: &str) -> u32 {
        self.ids.get(frame).copied().unwrap_or(0)
    }

    fn id_or_insert(&mut self, frame: &str) -> u32 {
        if let Some(&id) = self.ids.get(frame) {
            return id;
        }
        let id = self.frames.len() as u32;
        self.frames.push(None);
        self.ids.insert(frame.to_string(), id);
        self.names.push(frame.to_string());
        id
    }

    fn cache(&self, id: u32) -> Option<&Cache> {
        self.frames.get(id as usize).and_then(|c| c.as_ref())
    }

    fn name(&self, id: u32) -> &str {
        &self.names[id as usize]
    }

    /// Adds a transform `parent` <- `child` at `stamp_ns` (BufferCore::setTransform). Returns tf2's
    /// error/warning text when it is ignored (self transform, missing frame id, NaN, data older
    /// than the cache window), which tf2 would print to the console.
    pub fn set_transform(
        &mut self,
        parent: &str,
        child: &str,
        stamp_ns: i64,
        transform: &Transform,
        authority: &str,
        is_static: bool,
    ) -> Result<(), String> {
        let (parent, child) = (strip_slash(parent), strip_slash(child));
        let t = transform.translation;
        let q = quat_through_matrix(transform.rotation);
        let mut errors = Vec::new();
        if child == parent {
            errors.push(format!(
                "TF_SELF_TRANSFORM: Ignoring transform from authority \"{authority}\" with frame_id and  child_frame_id \"{child}\" because they are the same"
            ));
        }
        if child.is_empty() {
            errors.push(format!(
                "TF_NO_CHILD_FRAME_ID: Ignoring transform from authority \"{authority}\" because child_frame_id not set "
            ));
        }
        if parent.is_empty() {
            errors.push(format!(
                "TF_NO_FRAME_ID: Ignoring transform with child_frame_id \"{child}\"  from authority \"{authority}\" because frame_id not set"
            ));
        }
        if t.iter().chain(q.iter()).any(|v| v.is_nan()) {
            errors.push(format!(
                "TF_NAN_INPUT: Ignoring transform for child_frame_id \"{child}\" from authority \"{authority}\" because of a nan value in the transform ({} {} {}) ({} {} {} {})",
                c_f(t[0]), c_f(t[1]), c_f(t[2]), c_f(q[0]), c_f(q[1]), c_f(q[2]), c_f(q[3])
            ));
        }
        let norm2 = q[3] * q[3] + q[0] * q[0] + q[1] * q[1] + q[2] * q[2];
        if !((norm2 - 1.0).abs() < QUATERNION_NORMALIZATION_TOLERANCE) {
            errors.push(format!(
                "TF_DENORMALIZED_QUATERNION: Ignoring transform for child_frame_id \"{child}\" from authority \"{authority}\" because of an invalid quaternion in the transform ({} {} {} {})",
                c_f(q[0]), c_f(q[1]), c_f(q[2]), c_f(q[3])
            ));
        }
        if !errors.is_empty() {
            return Err(errors.join("\n"));
        }
        let child_id = self.id_or_insert(child);
        let replace = match &self.frames[child_id as usize] {
            None => true,
            Some(Cache::Static(_)) => !is_static,
            Some(Cache::Dynamic(_)) => is_static,
        };
        if replace {
            self.frames[child_id as usize] = Some(if is_static {
                Cache::Static(Storage { rotation: [0.0, 0.0, 0.0, 1.0], translation: [0.0; 3], stamp: 0, parent: 0, child: 0 })
            } else {
                Cache::Dynamic(VecDeque::new())
            });
        }
        let parent_id = self.id_or_insert(parent);
        let data = Storage { rotation: q, translation: t, stamp: stamp_ns, parent: parent_id, child: child_id };
        let max_age = self.cache_time_ns;
        if self.frames[child_id as usize].as_mut().is_some_and(|c| c.insert(data, max_age)) {
            Ok(())
        } else {
            Err(format!(
                "TF_OLD_DATA ignoring data from the past for frame {child} at time {} according to authority {authority}\nPossible reasons are listed at http://wiki.ros.org/tf/Errors%20explained",
                display_time(stamp_ns)
            ))
        }
    }

    fn validate(&self, what: &str, frame: &str) -> Result<u32, TfError> {
        if frame.is_empty() {
            return Err(TfError::new(
                ErrorKind::InvalidArgument,
                format!("Invalid argument \"{frame}\" passed to {what} - in tf2 frame_ids cannot be empty"),
            ));
        }
        if frame.starts_with('/') {
            return Err(TfError::new(
                ErrorKind::InvalidArgument,
                format!("Invalid argument \"{frame}\" passed to {what} - in tf2 frame_ids cannot start with a '/'"),
            ));
        }
        match self.id(frame) {
            0 => Err(TfError::new(ErrorKind::Lookup, format!("\"{frame}\" passed to {what} does not exist. "))),
            id => Ok(id),
        }
    }

    /// Transform mapping `source` coordinates into `target` at `time_ns` (0 = latest available),
    /// like tf2 `Buffer.lookup_transform(target, source, time)`. The stamp is the time used.
    pub fn lookup_transform(&self, target: &str, source: &str, time_ns: i64) -> Result<StampedTransform, TfError> {
        let (transform, stamp_ns) = if target == source {
            let stamp = match (time_ns, self.cache(self.id(target))) {
                (0, Some(c)) => c.latest(),
                _ => time_ns,
            };
            (Transform::IDENTITY, stamp)
        } else {
            let target_id = self.validate("lookupTransform argument target_frame", target)?;
            let source_id = self.validate("lookupTransform argument source_frame", source)?;
            let (accum, end, time) = self.walk(time_ns, target_id, source_id)?;
            let (translation, rotation) = accum.finalize(end);
            (Transform { translation, rotation: quat_through_matrix(rotation) }, time)
        };
        Ok(StampedTransform { stamp_ns, frame_id: target.into(), child_frame_id: source.into(), transform })
    }

    /// Whether `lookup_transform(target, source, time_ns)` would succeed (tf2 canTransform), with
    /// tf2's reason when not.
    pub fn can_transform(&self, target: &str, source: &str, time_ns: i64) -> Result<(), String> {
        if target == source {
            return Ok(());
        }
        let check = |what: &str, frame: &str| -> Result<u32, String> {
            let why = if frame.is_empty() {
                "in tf2 frame_ids cannot be empty"
            } else if frame.starts_with('/') {
                "in tf2 frame_ids cannot start with a '/'"
            } else {
                match self.id(frame) {
                    0 => "frame does not exist",
                    id => return Ok(id),
                }
            };
            Err(format!("Invalid frame ID \"{frame}\" passed to {what} - {why}"))
        };
        let target_id = check("canTransform argument target_frame", target)?;
        let source_id = check("canTransform argument source_frame", source)?;
        self.walk(time_ns, target_id, source_id).map(|_| ()).map_err(|e| e.message)
    }

    /// Whether a frame has been seen (as parent or child).
    pub fn frame_exists(&self, frame: &str) -> bool {
        self.ids.contains_key(frame)
    }

    /// Names of all frames that have a parent (tf2 getAllFrameNames).
    pub fn frame_names(&self) -> Vec<String> {
        (1..self.frames.len()).filter(|&i| self.frames[i].is_some()).map(|i| self.names[i].clone()).collect()
    }

    /// The frame's parent at the newest data, if it has one.
    pub fn parent(&self, frame: &str) -> Option<&str> {
        let (_, parent) = self.cache(self.id(frame))?.latest_time_and_parent();
        (parent != 0).then(|| self.name(parent))
    }

    fn all_frames_as_string(&self) -> String {
        let mut out = String::new();
        for id in 1..self.frames.len() as u32 {
            if let Some(c) = self.cache(id) {
                let parent = c.get_data(0).map_or(0, |s| s.parent);
                out += &format!("Frame {} exists with parent {}.\n", self.name(id), self.name(parent));
            }
        }
        out
    }

    fn loop_error(&self) -> TfError {
        TfError::new(
            ErrorKind::Lookup,
            format!("The tf tree is invalid because it contains a loop.\n{}\n", self.all_frames_as_string()),
        )
    }

    fn connectivity_error(&self, source_id: u32, target_id: u32) -> TfError {
        TfError::new(
            ErrorKind::Connectivity,
            format!(
                "Could not find a connection between '{}' and '{}' because they are not part of the same tree.Tf has two or more unconnected trees.",
                self.name(target_id),
                self.name(source_id)
            ),
        )
    }

    /// BufferCore::getLatestCommonTime.
    fn latest_common_time(&self, target_id: u32, source_id: u32) -> Result<i64, TfError> {
        if source_id == 0 || target_id == 0 {
            return Err(TfError::new(ErrorKind::Lookup, ""));
        }
        let or_zero = |t: i64| if t == i64::MAX { 0 } else { t };
        if source_id == target_id {
            return Ok(self.cache(source_id).map_or(0, |c| c.latest()));
        }
        let mut chain: Vec<(i64, u32)> = Vec::new();
        let (mut frame, mut depth, mut common) = (source_id, 0, i64::MAX);
        while frame != 0 {
            let Some(cache) = self.cache(frame) else { break };
            let (t, parent) = cache.latest_time_and_parent();
            if parent == 0 {
                break;
            }
            if t != 0 {
                common = common.min(t);
            }
            chain.push((t, parent));
            frame = parent;
            if frame == target_id {
                return Ok(or_zero(common));
            }
            depth += 1;
            if depth > MAX_GRAPH_DEPTH {
                return Err(self.loop_error());
            }
        }
        let (mut frame, mut depth, mut common, mut common_parent) = (target_id, 0, i64::MAX, 0);
        loop {
            let Some(cache) = self.cache(frame) else { break };
            let (t, parent) = cache.latest_time_and_parent();
            if parent == 0 {
                break;
            }
            if t != 0 {
                common = common.min(t);
            }
            if chain.iter().any(|&(_, p)| p == parent) {
                common_parent = parent;
                break;
            }
            frame = parent;
            if frame == source_id {
                return Ok(or_zero(common));
            }
            depth += 1;
            if depth > MAX_GRAPH_DEPTH {
                return Err(self.loop_error());
            }
        }
        if common_parent == 0 {
            return Err(self.connectivity_error(source_id, target_id));
        }
        for &(t, p) in &chain {
            if t != 0 {
                common = common.min(t);
            }
            if p == common_parent {
                break;
            }
        }
        Ok(or_zero(common))
    }

    /// BufferCore::walkToTopParent with a TransformAccum.
    fn walk(&self, time: i64, target_id: u32, source_id: u32) -> Result<(Accum, WalkEnd, i64), TfError> {
        let mut acc = Accum::new();
        if source_id == target_id {
            return Ok((acc, WalkEnd::Identity, time));
        }
        let time = if time == 0 { self.latest_common_time(target_id, source_id)? } else { time };
        let suffix = || {
            format!(", when looking up transform from frame [{}] to frame [{}]", self.name(source_id), self.name(target_id))
        };

        let (mut frame, mut top_parent, mut depth) = (source_id, source_id, 0);
        // Set when the walk up from the source stopped at missing data: (error, that frame's newest stamp).
        let mut source_gap: Option<(CacheError, i64)> = None;
        while frame != 0 {
            let Some(cache) = self.cache(frame) else {
                top_parent = frame;
                break;
            };
            let st = match cache.get_data(time) {
                Ok(st) if st.parent != 0 => st,
                result => {
                    // "there may still be a path from source -> target"
                    top_parent = frame;
                    let err = result.err().unwrap_or((ErrorKind::Extrapolation, String::new()));
                    source_gap = Some((err, cache.latest()));
                    break;
                }
            };
            if frame == target_id {
                return Ok((acc, WalkEnd::TargetParentOfSource, time));
            }
            acc.accum(&st, true);
            top_parent = frame;
            frame = st.parent;
            depth += 1;
            if depth > MAX_GRAPH_DEPTH {
                return Err(self.loop_error());
            }
        }

        let (mut frame, mut depth) = (target_id, 0);
        while frame != top_parent {
            let Some(cache) = self.cache(frame) else { break };
            let st = match cache.get_data(time) {
                Ok(st) if st.parent != 0 => st,
                result => {
                    let (kind, msg) = result.err().unwrap_or((ErrorKind::Extrapolation, String::new()));
                    // Prefer the source side's error unless its data is older than this frame's.
                    if let Some(((source_kind, source_msg), source_latest)) = &source_gap {
                        if *source_latest >= cache.latest() {
                            return Err(TfError::new(*source_kind, format!("{source_msg}{}", suffix())));
                        }
                    }
                    return Err(TfError::new(kind, format!("{msg}{}", suffix())));
                }
            };
            if frame == source_id {
                return Ok((acc, WalkEnd::SourceParentOfTarget, time));
            }
            acc.accum(&st, false);
            frame = st.parent;
            depth += 1;
            if depth > MAX_GRAPH_DEPTH {
                return Err(self.loop_error());
            }
        }
        if frame != top_parent {
            if let Some(((kind, msg), _)) = source_gap {
                return Err(TfError::new(kind, format!("{msg}{}", suffix())));
            }
            return Err(self.connectivity_error(source_id, target_id));
        }
        Ok((acc, WalkEnd::FullPath, time))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: i64 = 1_000_000_000;

    fn tf(t: Vec3, yaw: f64) -> Transform {
        Transform { translation: t, rotation: [0.0, 0.0, (yaw / 2.0).sin(), (yaw / 2.0).cos()] }
    }

    fn yaw(q: Quat) -> f64 {
        2.0 * q[2].atan2(q[3])
    }

    fn close(a: &[f64], b: &[f64]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-9)
    }

    /// map -> odom (dynamic, 100.05..100.85 s) -> base_link (dynamic, 100..101 s) -> camera (static)
    fn robot() -> TfBuffer {
        let mut b = TfBuffer::new();
        b.set_transform("base_link", "camera", 0, &tf([0.1, 0.0, 0.2], 0.0), "a", true).unwrap();
        for i in 0..=10 {
            let t = 100 * S + i * S / 10;
            b.set_transform("odom", "base_link", t, &tf([i as f64, 0.0, 0.0], 0.0), "a", false).unwrap();
        }
        for i in 0..=4 {
            let t = 100 * S + i * S / 5 + S / 20;
            b.set_transform("map", "odom", t, &tf([0.0, 1.0, 0.0], 0.1 * i as f64), "a", false).unwrap();
        }
        b
    }

    #[test]
    fn interpolates_translation_and_rotation() {
        let b = robot();
        let t = b.lookup_transform("odom", "base_link", 100 * S + S / 4).unwrap();
        assert_eq!(t.stamp_ns, 100 * S + S / 4);
        assert!(close(&t.transform.translation, &[2.5, 0.0, 0.0]), "{t:?}");
        let m = b.lookup_transform("map", "odom", 100 * S + S / 20 + S / 10).unwrap();
        assert!((yaw(m.transform.rotation) - 0.05).abs() < 1e-12, "{m:?}");
    }

    #[test]
    fn chains_through_static_and_inverts() {
        let b = robot();
        let time = 100 * S + S / 2;
        let t = b.lookup_transform("map", "camera", time).unwrap();
        let m = b.lookup_transform("map", "odom", time).unwrap().transform;
        let o = b.lookup_transform("odom", "base_link", time).unwrap().transform;
        let expect = m.apply(o.apply([0.1, 0.0, 0.2]));
        assert!(close(&t.transform.translation, &expect), "{t:?} {expect:?}");
        let inv = b.lookup_transform("camera", "map", time).unwrap().transform;
        assert!(close(&inv.apply(t.transform.translation), &[0.0; 3]));
    }

    #[test]
    fn latest_is_the_common_time_of_the_chain() {
        let b = robot();
        // odom -> base_link ends at 101.0 s, map -> odom at 100.85 s.
        let t = b.lookup_transform("map", "camera", 0).unwrap();
        assert_eq!(t.stamp_ns, 100 * S + 4 * S / 5 + S / 20);
        assert_eq!(b.lookup_transform("odom", "base_link", 0).unwrap().stamp_ns, 101 * S);
        assert_eq!(b.lookup_transform("base_link", "camera", 0).unwrap().stamp_ns, 0);
    }

    #[test]
    fn no_extrapolation() {
        let b = robot();
        let past = b.lookup_transform("odom", "base_link", 99 * S).unwrap_err();
        assert_eq!(past.kind, ErrorKind::BackwardExtrapolation);
        assert!(past.message.starts_with("Lookup would require extrapolation into the past."), "{past}");
        let future = b.lookup_transform("map", "base_link", 101 * S).unwrap_err();
        assert_eq!(future.kind, ErrorKind::ForwardExtrapolation, "{future}");
        assert!(future.message.ends_with("when looking up transform from frame [base_link] to frame [map]"), "{future}");
        assert!(b.can_transform("map", "base_link", 101 * S).is_err());
        assert!(b.can_transform("map", "base_link", 100 * S + S / 2).is_ok());
    }

    #[test]
    fn errors_like_tf2() {
        let mut b = robot();
        b.set_transform("world", "other", 0, &Transform::IDENTITY, "a", true).unwrap();
        assert_eq!(b.lookup_transform("map", "other", 0).unwrap_err().kind, ErrorKind::Connectivity);
        assert_eq!(b.lookup_transform("map", "nowhere", 0).unwrap_err().kind, ErrorKind::Lookup);
        assert_eq!(b.lookup_transform("/map", "odom", 0).unwrap_err().kind, ErrorKind::InvalidArgument);
        assert_eq!(b.lookup_transform("", "odom", 0).unwrap_err().kind, ErrorKind::InvalidArgument);
        // The same frame is always the identity, even unknown ones.
        let same = b.lookup_transform("nowhere", "nowhere", 5).unwrap();
        assert_eq!((same.stamp_ns, same.transform), (5, Transform::IDENTITY));
        assert!(b.set_transform("a", "a", 1, &Transform::IDENTITY, "x", false).is_err());
        let nan = Transform { translation: [f64::NAN, 0.0, 0.0], rotation: [0.0, 0.0, 0.0, 1.0] };
        assert!(b.set_transform("a", "b", 1, &nan, "x", false).is_err());
    }

    #[test]
    fn cache_window_and_static_frames() {
        let mut b = TfBuffer::new();
        b.set_transform("map", "fixed", 1, &Transform::IDENTITY, "a", true).unwrap();
        for i in 0..=30 {
            b.set_transform("map", "robot", i * S, &Transform::IDENTITY, "a", false).unwrap();
        }
        // Pruned on insert to 10 s behind the newest entry before it: 19..30 s remain.
        assert!(b.lookup_transform("map", "robot", 19 * S).is_ok());
        assert!(b.lookup_transform("map", "robot", 18 * S).is_err());
        // Re-sending a stamp within the window is fine; older data is refused.
        assert!(b.set_transform("map", "robot", 20 * S, &Transform::IDENTITY, "a", false).is_ok());
        let old = b.set_transform("map", "robot", 15 * S, &Transform::IDENTITY, "a", false);
        assert!(old.unwrap_err().starts_with("TF_OLD_DATA"));
        // Static transforms answer for any time.
        assert!(b.lookup_transform("fixed", "robot", 25 * S).is_ok());
        assert_eq!(b.lookup_transform("map", "fixed", 0).unwrap().stamp_ns, 0);
    }

    #[test]
    fn slerp_takes_the_short_way() {
        let a = [0.0, 0.0, (1.5f64).sin(), (1.5f64).cos()]; // yaw 3 rad
        let b = [0.0, 0.0, -(1.5f64).sin(), (1.5f64).cos()]; // yaw -3 rad, 0.28 rad further on
        let m = slerp(a, b, 0.5);
        assert!((yaw(m).abs() - std::f64::consts::PI).abs() < 1e-9, "{m:?}");
    }
}
