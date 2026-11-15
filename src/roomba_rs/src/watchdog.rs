//! vslam_watchdog's logic without ROS, ported from roomba_perception/vslam_watchdog.py (see that
//! file's docstring for what it checks and why). The structure, names and arithmetic follow the
//! Python node so both make the same decisions on the same input.
//!
//! Time comes in as numbers: message stamps in seconds and `now`, a CLOCK_MONOTONIC reading in
//! seconds like Python's time.monotonic(). Everything the node does goes out through [`Io`]
//! (TF, odometry, service calls, log lines), implemented by the ROS node (bin/vslam_watchdog.rs)
//! and by the deterministic replay (examples/watchdog_replay.rs).

use std::collections::VecDeque;

use crate::geometry::{cross, norm, quat_conj, quat_mul, quat_rotate, Quat, Vec3, IDENTITY};

pub const VO_OK: u8 = 1;
pub const VO_FAILED: u8 = 2;

pub type Pose = (Vec3, Quat);
const POSE_IDENTITY: Pose = ([0.0; 3], IDENTITY);
/// math.degrees() factor.
const RAD2DEG: f64 = 180.0 / std::f64::consts::PI;

// ---------- small math, same formulas (and operation order) as the Python helpers ----------

fn compose(a: &Pose, b: &Pose) -> Pose {
    let r = quat_rotate(a.1, b.0);
    ([a.0[0] + r[0], a.0[1] + r[1], a.0[2] + r[2]], quat_mul(a.1, b.1))
}

fn invert(a: &Pose) -> Pose {
    let qi = quat_conj(a.1);
    let t = quat_rotate(qi, a.0);
    ([-t[0], -t[1], -t[2]], qi)
}

fn angle_between(q1: Quat, q2: Quat) -> f64 {
    let d = (q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]).abs();
    2.0 * d.min(1.0).acos()
}

fn quat_normalize(q: Quat) -> Quat {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

fn rotvec_to_quat(r: Vec3) -> Quat {
    let a = norm(r);
    let s = if a < 1e-9 { 0.5 } else { (0.5 * a).sin() / a };
    [r[0] * s, r[1] * s, r[2] * s, (0.5 * a).cos()]
}

fn quat_to_rotvec(q: Quat) -> Vec3 {
    let [mut x, mut y, mut z, mut w] = q;
    if w < 0.0 {
        (x, y, z, w) = (-x, -y, -z, -w);
    }
    let n = (x * x + y * y + z * z).sqrt();
    if n < 1e-9 {
        return [2.0 * x, 2.0 * y, 2.0 * z];
    }
    let k = 2.0 * n.atan2(w) / n;
    [x * k, y * k, z * k]
}

pub fn quat_yaw(q: Quat) -> f64 {
    let [x, y, z, w] = q;
    (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z))
}

fn yaw_quat(yaw: f64) -> Quat {
    [0.0, 0.0, (yaw / 2.0).sin(), (yaw / 2.0).cos()]
}

fn wrap_angle(a: f64) -> f64 {
    a.sin().atan2(a.cos())
}

fn norm2(a: Vec3, b: Vec3) -> f64 {
    let (dx, dy) = (a[0] - b[0], a[1] - b[1]);
    (dx * dx + dy * dy).sqrt()
}

/// Python's round(x, places).
pub fn py_round(x: f64, places: usize) -> f64 {
    format!("{x:.places$}").parse().unwrap_or(x)
}

/// A float as Python's json.dumps writes it (repr: "1.0", "0.25").
pub fn json_float(x: f64) -> String {
    if x.is_finite() && x.fract() == 0.0 { format!("{x:.1}") } else { format!("{x}") }
}

// ---------- interface ----------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stamp {
    pub sec: i32,
    pub nanosec: u32,
}

impl Stamp {
    pub fn secs(self) -> f64 {
        self.sec as f64 + self.nanosec as f64 * 1e-9
    }

    /// Python: ns = int(round(t * 1e9)); sec, nanosec = divmod(ns, 1e9).
    pub fn from_secs(t: f64) -> Self {
        let ns = (t * 1e9).round_ties_even() as i64;
        Stamp { sec: ns.div_euclid(1_000_000_000) as i32, nanosec: ns.rem_euclid(1_000_000_000) as u32 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Twist {
    pub linear: Vec3,
    pub angular: Vec3,
    pub covariance: [f64; 36],
}

impl Default for Twist {
    fn default() -> Self {
        Twist { linear: [0.0; 3], angular: [0.0; 3], covariance: [0.0; 36] }
    }
}

/// nav_msgs/Odometry content (frames are odom_frame / base_frame on output).
#[derive(Clone, Debug, PartialEq)]
pub struct Odom {
    pub stamp: Stamp,
    pub pos: Vec3,
    pub quat: Quat,
    pub pose_covariance: [f64; 36],
    pub twist: Twist,
}

fn diag_cov(d: [f64; 6]) -> [f64; 36] {
    let mut c = [0.0; 36];
    for (i, v) in d.into_iter().enumerate() {
        c[i * 7] = v;
    }
    c
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Svc {
    Pause,
    Reset,
    NewMap,
    Resume,
}

impl Svc {
    pub const ALL: [Svc; 4] = [Svc::Pause, Svc::Reset, Svc::NewMap, Svc::Resume];

    pub fn key(self) -> &'static str {
        match self {
            Svc::Pause => "pause",
            Svc::Reset => "reset",
            Svc::NewMap => "new_map",
            Svc::Resume => "resume",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Info,
    Warn,
    Error,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Pausing,
    Resetting,
    Waiting,
    NewMap,
    Resuming,
}

impl Phase {
    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Idle => "idle",
            Phase::Pausing => "pausing",
            Phase::Resetting => "resetting",
            Phase::Waiting => "waiting",
            Phase::NewMap => "new_map",
            Phase::Resuming => "resuming",
        }
    }
}

/// Everything the watchdog does to the outside world.
pub trait Io {
    fn service_ready(&self, svc: Svc) -> bool;
    /// TF odom -> base_frame and odom -> vslam_odom_frame (one TFMessage).
    fn tf(&mut self, stamp: Stamp, base: Pose, anchor: Pose);
    /// /odom (publish_tf) or fused_vio_topic.
    fn odom(&mut self, odom: Odom);
    /// Call the service; report the response with [`Watchdog::service_done`].
    fn call(&mut self, svc: Svc);
    fn log(&mut self, level: Level, msg: String);
}

/// Parameter lookup: returns the value of parameter `name`, declaring it with `default`.
pub trait ParamSource {
    fn float(&mut self, name: &str, default: f64) -> anyhow::Result<f64>;
    fn int(&mut self, name: &str, default: i64) -> anyhow::Result<i64>;
    fn boolean(&mut self, name: &str, default: bool) -> anyhow::Result<bool>;
    fn string(&mut self, name: &str, default: &str) -> anyhow::Result<String>;
}

#[derive(Clone, Debug)]
pub struct Params {
    pub max_speed: f64,
    pub max_vspeed: f64,
    pub max_accel: f64,
    pub max_turn: f64,
    pub margin_pos: f64,
    pub margin_rot: f64,
    pub gyro_tol: f64,
    pub gyro_tol_rel: f64,
    pub level_gain: f64,
    pub burst_count: i64,
    pub burst_window: f64,
    pub max_fail_s: f64,
    pub boundary: f64,
    pub max_height: f64,
    pub cooldown: f64,
    pub recover_timeout: f64,
    pub new_map_always: bool,
    pub odom_frame: String,
    pub vslam_frame: String,
    pub base_frame: String,
    pub imu_topic: String,
    pub rtabmap_ns: String,
    pub wheel_topic: String,
    pub wheel_max_age: f64,
    pub wheel_tol: f64,
    pub wheel_tol_rel: f64,
    pub wheel_window: f64,
    pub wheel_window_tol: f64,
    pub wheel_window_rel: f64,
    pub planar: bool,
    pub publish_tf: bool,
    pub fused_topic: String,
    pub vio_pos_var: f64,
    pub vio_yaw_var: f64,
}

impl Params {
    /// The node's parameters (names, defaults, declaration order) as in the Python node.
    pub fn load(p: &mut impl ParamSource) -> anyhow::Result<Self> {
        Ok(Params {
            max_speed: p.float("max_speed_mps", 1.5)?,
            max_vspeed: p.float("max_vertical_speed_mps", 0.8)?,
            max_accel: p.float("max_accel_mps2", 10.0)?,
            max_turn: p.float("max_turn_rate_rps", 8.0)?,
            margin_pos: p.float("jitter_margin_m", 0.02)?,
            margin_rot: p.float("jitter_margin_rad", 0.03)?,
            gyro_tol: p.float("gyro_tolerance_rad", 0.05)?,
            gyro_tol_rel: p.float("gyro_tolerance_ratio", 0.15)?,
            level_gain: p.float("level_gain", 0.02)?,
            burst_count: p.int("violations_for_reset", 8)?,
            burst_window: p.float("violation_window_s", 2.0)?,
            max_fail_s: p.float("max_tracking_loss_s", 1.0)?,
            boundary: p.float("boundary_radius_m", 20.0)?,
            max_height: p.float("max_height_m", 3.0)?,
            cooldown: p.float("reset_cooldown_s", 5.0)?,
            recover_timeout: p.float("recovery_timeout_s", 6.0)?,
            new_map_always: p.boolean("new_map_on_recovery", false)?,
            odom_frame: p.string("odom_frame", "odom")?,
            vslam_frame: p.string("vslam_odom_frame", "vslam_odom")?,
            base_frame: p.string("base_frame", "camera_link")?,
            imu_topic: p.string("imu_topic", "/camera/imu/data_raw")?,
            rtabmap_ns: p.string("rtabmap_ns", "/rtabmap/rtabmap")?,
            wheel_topic: p.string("wheel_odom_topic", "")?,
            wheel_max_age: p.float("wheel_max_age_s", 0.25)?,
            wheel_tol: p.float("wheel_tolerance_m", 0.03)?,
            wheel_tol_rel: p.float("wheel_tolerance_ratio", 0.5)?,
            wheel_window: p.float("wheel_window_s", 0.5)?,
            wheel_window_tol: p.float("wheel_window_tolerance_m", 0.08)?,
            wheel_window_rel: p.float("wheel_window_tolerance_ratio", 0.6)?,
            planar: p.boolean("planar", false)?,
            publish_tf: p.boolean("publish_tf", true)?,
            fused_topic: p.string("fused_vio_topic", "/vslam_filter/vio")?,
            vio_pos_var: p.float("vio_position_variance", 0.002)?,
            vio_yaw_var: p.float("vio_yaw_variance", 0.0005)?,
        })
    }

    pub fn service_name(&self, svc: Svc) -> String {
        match svc {
            Svc::Reset => "/visual_slam/reset".into(),
            Svc::Pause => format!("{}/pause", self.rtabmap_ns),
            Svc::Resume => format!("{}/resume", self.rtabmap_ns),
            Svc::NewMap => format!("{}/trigger_new_map", self.rtabmap_ns),
        }
    }
}

/// Defaults of every parameter (a ParamSource that only returns defaults).
pub struct Defaults;

impl ParamSource for Defaults {
    fn float(&mut self, _: &str, d: f64) -> anyhow::Result<f64> {
        Ok(d)
    }
    fn int(&mut self, _: &str, d: i64) -> anyhow::Result<i64> {
        Ok(d)
    }
    fn boolean(&mut self, _: &str, d: bool) -> anyhow::Result<bool> {
        Ok(d)
    }
    fn string(&mut self, _: &str, d: &str) -> anyhow::Result<String> {
        Ok(d.to_string())
    }
}

// ---------- the watchdog ----------

#[derive(Clone, Copy, Debug)]
struct Stamped {
    t: f64,
    pos: Vec3,
    quat: Quat,
}

#[derive(Clone, Copy, Debug)]
struct WheelSample {
    t: f64,
    x: f64,
    y: f64,
    yaw: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Pos,
    Rot,
    Lost,
}

fn push_bounded<T>(buf: &mut VecDeque<T>, item: T, maxlen: usize) {
    if buf.len() == maxlen {
        buf.pop_front();
    }
    buf.push_back(item);
}

const HISTORY_LEN: usize = 300;
const IMU_LEN: usize = 4000;
const WHEEL_LEN: usize = 2600;
const RECENT_LEN: usize = 60;

pub struct Watchdog {
    pub p: Params,
    offset: Pose, // odom <- vslam_odom
    last: Option<Stamped>,
    vel: Vec3,
    violations: VecDeque<f64>,
    pub filtered_total: u64,
    history: VecDeque<(f64, Stamped)>,
    pending_anchor: Option<Stamped>,
    imu_buf: VecDeque<(f64, Vec3)>,
    pub imu_rot: Option<Quat>,
    gyro_bias: Vec3,
    acc_lp: Option<Vec3>,
    last_imu: f64,
    pub state: Phase,
    state_since: f64,
    last_recovery: f64,
    pub recoveries: u64,
    fail_since: Option<f64>,
    ok_count: u64,
    last_vo_ok: f64,
    anchor: Option<Stamped>,
    last_log: f64,
    wheel_buf: VecDeque<WheelSample>,
    recent: VecDeque<Stamped>,
    wheel_twist: (f64, f64),
    bridge: Option<Stamped>,
    bridge_t: f64,
    pub bridged: u64,
    pub wheel_followed: u64,
    last_wheel: f64,
    pub forwarded: u64,
}

impl Watchdog {
    pub fn new(p: Params, now: f64) -> Self {
        Watchdog {
            p,
            offset: POSE_IDENTITY,
            last: None,
            vel: [0.0; 3],
            violations: VecDeque::new(),
            filtered_total: 0,
            history: VecDeque::with_capacity(HISTORY_LEN),
            pending_anchor: None,
            imu_buf: VecDeque::with_capacity(IMU_LEN),
            imu_rot: None,
            gyro_bias: [0.0; 3],
            acc_lp: None,
            last_imu: 0.0,
            state: Phase::Idle,
            state_since: now,
            last_recovery: 0.0,
            recoveries: 0,
            fail_since: None,
            ok_count: 0,
            last_vo_ok: 0.0,
            anchor: None,
            last_log: 0.0,
            wheel_buf: VecDeque::with_capacity(WHEEL_LEN),
            recent: VecDeque::with_capacity(RECENT_LEN),
            wheel_twist: (0.0, 0.0),
            bridge: None,
            bridge_t: 0.0,
            bridged: 0,
            wheel_followed: 0,
            last_wheel: 0.0,
            forwarded: 0,
        }
    }

    // ---------- gyro ----------

    /// Raw IMU sample (angular velocity `w`, specific force `a`, IMU frame). `lookup` gives the
    /// rotation base_frame <- IMU frame (static TF); it is only asked until it succeeds.
    pub fn on_imu(&mut self, t: f64, w: Vec3, a: Vec3, now: f64, lookup: impl FnOnce() -> Option<Quat>, io: &mut dyn Io) {
        let rot = match self.imu_rot {
            Some(r) => r,
            None => {
                let Some(r) = lookup() else { return };
                self.imu_rot = Some(r);
                io.log(Level::Info, "Gyro connected: rotations are checked against the IMU".into());
                r
            }
        };
        if self.imu_buf.back().is_some_and(|b| t <= b.0) {
            return;
        }
        push_bounded(&mut self.imu_buf, (t, quat_rotate(rot, w)), IMU_LEN);
        let a_c = quat_rotate(rot, a);
        if norm(a_c) > 1.0 {
            // ~1 s time constant at 200 Hz: averages out steps and hand sway.
            self.acc_lp = Some(match self.acc_lp {
                None => a_c,
                Some(p) => [0.995 * p[0] + 0.005 * a_c[0], 0.995 * p[1] + 0.005 * a_c[1], 0.995 * p[2] + 0.005 * a_c[2]],
            });
        }
        self.last_imu = now;
    }

    /// Body rotation from t0 to t1 integrated from the gyro (q1 = q0 * delta), or None when the
    /// IMU buffer doesn't cover the interval.
    fn gyro_delta(&self, t0: f64, t1: f64) -> Option<Quat> {
        let buf = &self.imu_buf;
        if buf.len() < 2 || t1 <= t0 || buf[buf.len() - 1].0 < t1 - 0.03 {
            return None;
        }
        // Newest sample at or before t0 (or the oldest one).
        let mut k = buf.len() - 1;
        while k > 0 && buf[k].0 > t0 {
            k -= 1;
        }
        if buf[k].0 > t0 + 0.01 {
            return None;
        }
        let b = self.gyro_bias;
        let mut q = IDENTITY;
        for i in k..buf.len() {
            let (t, w) = buf[i];
            let ta = t.max(t0);
            let tb = (if i + 1 < buf.len() { buf[i + 1].0 } else { t1 }).min(t1);
            if tb > ta {
                let dt = tb - ta;
                q = quat_mul(q, rotvec_to_quat([(w[0] - b[0]) * dt, (w[1] - b[1]) * dt, (w[2] - b[2]) * dt]));
            }
        }
        Some(quat_normalize(q))
    }

    /// While tracking is good the VIO rotation is accurate on average: the leftover difference
    /// to the gyro over slow frames is gyro bias (learned slowly, outliers clipped).
    fn learn_gyro_bias(&mut self, q_vo: Quat, q_gyro: Quat, dt: f64) {
        let rv_g = quat_to_rotvec(q_gyro);
        if dt <= 0.0 || dt > 0.2 || norm(rv_g) > 0.1 {
            return;
        }
        let rv_v = quat_to_rotvec(q_vo);
        let mut resid = [(rv_g[0] - rv_v[0]) / dt, (rv_g[1] - rv_v[1]) / dt, (rv_g[2] - rv_v[2]) / dt];
        let n = norm(resid);
        if n > 0.02 {
            let k = 0.02 / n;
            resid = [resid[0] * k, resid[1] * k, resid[2] * k];
        }
        let b = self.gyro_bias;
        self.gyro_bias = [b[0] + 0.002 * resid[0], b[1] + 0.002 * resid[1], b[2] + 0.002 * resid[2]];
        let n = norm(self.gyro_bias);
        if n > 0.05 {
            // ~3 deg/s: anything larger is not bias
            let k = 0.05 / n;
            self.gyro_bias = self.gyro_bias.map(|v| v * k);
        }
    }

    /// Roll/pitch of quat (odom <- base) nudged toward the measured gravity direction, or None
    /// if unchanged.
    fn level(&self, quat: Quat) -> Option<Quat> {
        let acc = self.acc_lp?;
        if self.p.level_gain <= 0.0 {
            return None;
        }
        let n = norm(acc);
        if (n - 9.81).abs() > 0.8 {
            return None; // accelerating hard: the average isn't gravity right now
        }
        let up = quat_rotate(quat, [acc[0] / n, acc[1] / n, acc[2] / n]);
        let axis = cross(up, [0.0, 0.0, 1.0]);
        let s = norm(axis);
        if s < 1e-6 {
            return None;
        }
        let ang = s.atan2(up[2]);
        let g = self.p.level_gain;
        let corr = rotvec_to_quat([axis[0] / s * ang * g, axis[1] / s * ang * g, axis[2] / s * ang * g]);
        Some(quat_normalize(quat_mul(corr, quat)))
    }

    /// Angle between odom's up and the measured gravity (degrees).
    fn tilt_deg(&self, quat: Quat) -> Option<f64> {
        let acc = self.acc_lp?;
        let n = norm(acc);
        let up = quat_rotate(quat, [acc[0] / n, acc[1] / n, acc[2] / n]);
        Some(up[2].min(1.0).max(-1.0).acos() * RAD2DEG)
    }

    // ---------- wheel odometry ----------

    /// Wheel odometry at stamp t: pose (x, y, orientation) and twist (vx, wz).
    pub fn on_wheel(&mut self, t: f64, x: f64, y: f64, q: Quat, vx: f64, wz: f64, now: f64) {
        if self.wheel_buf.back().is_some_and(|b| t <= b.t) {
            return;
        }
        push_bounded(&mut self.wheel_buf, WheelSample { t, x, y, yaw: quat_yaw(q) }, WHEEL_LEN);
        self.wheel_twist = (vx, wz);
        self.last_wheel = now;
    }

    /// Wheel odometry (x, y, yaw) interpolated at t, or None if not covered.
    fn wheel_pose(&self, t: f64) -> Option<(f64, f64, f64)> {
        let buf = &self.wheel_buf;
        let (first, last) = (buf.front()?, buf.back()?);
        if t < first.t || t > last.t + self.p.wheel_max_age {
            return None;
        }
        if t >= last.t {
            return Some((last.x, last.y, last.yaw));
        }
        for i in (1..buf.len()).rev() {
            let a = buf[i - 1];
            if a.t <= t {
                let b = buf[i];
                let f = (t - a.t) / (b.t - a.t);
                return Some((a.x + f * (b.x - a.x), a.y + f * (b.y - a.y), a.yaw + f * wrap_angle(b.yaw - a.yaw)));
            }
        }
        None
    }

    /// Body motion t0 -> t1 measured by the wheels: (translation in the t0 body frame, yaw change).
    fn wheel_delta(&self, t0: f64, t1: f64) -> Option<(Vec3, f64)> {
        let p0 = self.wheel_pose(t0)?;
        let p1 = self.wheel_pose(t1)?;
        let (dx, dy) = (p1.0 - p0.0, p1.1 - p0.1);
        let (c, s) = (p0.2.cos(), p0.2.sin());
        Some(([c * dx + s * dy, -s * dx + c * dy, 0.0], wrap_angle(p1.2 - p0.2)))
    }

    /// Position at t1 starting from (pos0, quat0) at t0 and moving as the wheels did.
    fn wheel_position(&self, t0: f64, pos0: Vec3, quat0: Quat, t1: f64) -> Option<Vec3> {
        let d = self.wheel_delta(t0, t1)?;
        // The wheels move the robot along the floor: rotate by heading only.
        let r = quat_rotate(yaw_quat(quat_yaw(quat0)), d.0);
        Some([pos0[0] + r[0], pos0[1] + r[1], pos0[2] + r[2]])
    }

    /// Pose at t from `base`: wheels for position, gyro (or wheels) for heading.
    fn dead_reckon(&self, base: &Stamped, t: f64) -> Option<Pose> {
        let d = self.wheel_delta(base.t, t)?;
        let rot = self.gyro_delta(base.t, t).unwrap_or_else(|| yaw_quat(d.1));
        let r = quat_rotate(yaw_quat(quat_yaw(base.quat)), d.0);
        Some(([base.pos[0] + r[0], base.pos[1] + r[1], base.pos[2] + r[2]], quat_normalize(quat_mul(base.quat, rot))))
    }

    /// While the VIO is reset, keep odom -> base moving with the wheels.
    fn publish_bridge(&mut self, io: &mut dyn Io) {
        let Some(bridge) = self.bridge else { return };
        let Some(t) = self.wheel_buf.back().map(|w| w.t) else { return };
        if t <= self.bridge_t {
            return;
        }
        let Some((pos, quat)) = self.dead_reckon(&bridge, t) else { return };
        self.bridge_t = t;
        self.bridged += 1;
        let msg = Odom {
            stamp: Stamp::from_secs(t),
            pos: [0.0; 3],
            quat: [0.0, 0.0, 0.0, 1.0],
            pose_covariance: diag_cov([0.05, 0.05, 0.05, 0.05, 0.05, 0.1]),
            twist: Twist { linear: [self.wheel_twist.0, 0.0, 0.0], angular: [0.0, 0.0, self.wheel_twist.1], covariance: [0.0; 36] },
        };
        self.publish(&msg, pos, quat, false, io);
    }

    // ---------- tracking status ----------

    pub fn on_status(&mut self, vo_state: u8, now: f64, io: &mut dyn Io) {
        if vo_state == VO_OK {
            self.ok_count += 1;
            self.last_vo_ok = now;
        } else {
            self.ok_count = 0;
        }
        if self.state != Phase::Idle {
            return;
        }
        if vo_state == VO_FAILED {
            match self.fail_since {
                None => self.fail_since = Some(now),
                Some(since) if now - since > self.p.max_fail_s => {
                    self.recover(format!("visual tracking lost for {:.1} s", now - since), now, io)
                }
                Some(_) => {}
            }
        } else {
            self.fail_since = None;
        }
    }

    // ---------- motion filter ----------

    /// (reason and kind if moving from the last pose to (pos, quat) is impossible, gyro-measured
    /// rotation over the interval).
    fn check_motion(&self, t: f64, pos: Vec3, quat: Quat) -> (Option<(String, Kind)>, Option<Quat>) {
        let last = self.last.expect("check_motion needs a last pose");
        let (lt, lpos, lquat) = (last.t, last.pos, last.quat);
        let dt = t - lt;
        if dt <= 0.0 {
            return (Some(("non-increasing stamp".into(), Kind::Pos)), None);
        }
        let q_g = self.gyro_delta(lt, t);
        let p = &self.p;
        if let Some(g) = q_g {
            let gyro_angle = angle_between(g, IDENTITY);
            let err = angle_between(quat_mul(quat_conj(lquat), quat), g);
            if err > p.gyro_tol + p.gyro_tol_rel * gyro_angle {
                let reason = format!(
                    "rotation off by {:.0} deg from gyro ({:.0} deg/s measured)",
                    err * RAD2DEG,
                    gyro_angle / dt * RAD2DEG
                );
                return (Some((reason, Kind::Rot)), q_g);
            }
        } else {
            let turn = angle_between(lquat, quat) / dt;
            if turn > p.max_turn + p.margin_rot / dt {
                return (Some((format!("turn rate {:.0} deg/s", turn * RAD2DEG), Kind::Rot)), None);
            }
        }
        let v = [(pos[0] - lpos[0]) / dt, (pos[1] - lpos[1]) / dt, (pos[2] - lpos[2]) / dt];
        let speed = norm(v);
        // Ground robot: the wheels measure how far it really moved.
        let wheel_pos = if self.wheel_buf.is_empty() { None } else { self.wheel_position(lt, lpos, lquat, t) };
        if let Some(wp) = wheel_pos {
            if self.fail_since.is_some() {
                return (Some(("visual tracking lost, following the wheels".into(), Kind::Lost)), q_g);
            }
            let moved = norm2(wp, lpos);
            let off = norm2(pos, wp);
            if off > p.wheel_tol + p.wheel_tol_rel * moved {
                return (Some((format!("{:.0} cm off the wheel odometry", off * 100.0), Kind::Pos)), q_g);
            }
            // Slow drift passes the per-frame test, so also compare over a longer window.
            let reference = self.recent.iter().rev().find(|r| t - r.t >= p.wheel_window).copied();
            if let Some(r) = reference {
                if let Some(rp) = self.wheel_position(r.t, r.pos, r.quat, t) {
                    let moved = norm2(rp, r.pos);
                    let off = norm2(pos, rp);
                    if off > p.wheel_window_tol + p.wheel_window_rel * moved {
                        let reason =
                            format!("drifted {:.0} cm from the wheel odometry in {:.1} s", off * 100.0, t - r.t);
                        return (Some((reason, Kind::Pos)), q_g);
                    }
                }
            }
        }
        if speed > p.max_speed + p.margin_pos / dt {
            return (Some((format!("speed {speed:.1} m/s"), Kind::Pos)), q_g);
        }
        if v[2].abs() > p.max_vspeed + p.margin_pos / dt {
            return (Some((format!("vertical speed {:+.1} m/s", v[2]), Kind::Pos)), q_g);
        }
        let dv = norm([v[0] - self.vel[0], v[1] - self.vel[1], v[2] - self.vel[2]]);
        if dv > p.max_accel * dt + 2.0 * p.margin_pos / dt {
            return (Some((format!("acceleration {:.0} m/s^2", dv / dt), Kind::Pos)), q_g);
        }
        (None, q_g)
    }

    /// One VIO pose (in vslam_odom).
    pub fn on_odom(&mut self, msg: &Odom, now: f64, io: &mut dyn Io) {
        if self.state != Phase::Idle {
            return; // hold the last pose while the VIO is being reset
        }
        let t = msg.stamp.secs();
        if self.last.is_some_and(|l| t <= l.t) {
            return;
        }
        let raw: Pose = (msg.pos, msg.quat);
        if let Some(anchor) = self.pending_anchor.take() {
            // First pose after a reset: continue from the last good position, carried forward
            // by the wheels (position) and the gyro (heading).
            let reckoned = match self.dead_reckon(&anchor, t) {
                Some(r) => r,
                None => {
                    let q_g = self.gyro_delta(anchor.t, t).unwrap_or_else(|| {
                        io.log(Level::Warn, "No gyro data across the recovery; heading may be off".into());
                        IDENTITY
                    });
                    (anchor.pos, quat_normalize(quat_mul(anchor.quat, q_g)))
                }
            };
            self.offset = compose(&reckoned, &invert(&raw));
        }
        let (mut pos, mut quat) = compose(&self.offset, &raw);

        let (reason, q_g) = if self.last.is_none() { (None, None) } else { self.check_motion(t, pos, quat) };
        if let Some((reason, kind)) = &reason {
            // Keep the body where it plausibly is and continue the VIO's motion from there.
            let last = self.last.expect("a violation needs a last pose");
            let new_q = if *kind == Kind::Rot {
                q_g.map_or(last.quat, |g| quat_normalize(quat_mul(last.quat, g)))
            } else {
                quat
            };
            let wheel_pos = self.wheel_position(last.t, last.pos, last.quat, t);
            let dt = t - last.t;
            let pred_pos = wheel_pos.unwrap_or([
                last.pos[0] + self.vel[0] * dt,
                last.pos[1] + self.vel[1] * dt,
                last.pos[2] + self.vel[2] * dt,
            ]);
            self.vel = self.vel.map(|v| 0.8 * v); // don't coast on a stale velocity
            self.offset = compose(&(pred_pos, new_q), &invert(&raw));
            (pos, quat) = (pred_pos, new_q);
            if *kind == Kind::Lost {
                self.wheel_followed += 1;
                if now - self.last_log > 5.0 {
                    self.last_log = now;
                    io.log(Level::Info, "Visual tracking lost: following the wheel odometry".into());
                }
            } else {
                self.filtered_total += 1;
                self.violations.push_back(t);
                while self.violations.front().is_some_and(|&v| t - v > self.p.burst_window) {
                    self.violations.pop_front();
                }
                if now - self.last_log > 1.0 {
                    self.last_log = now;
                    io.log(
                        Level::Warn,
                        format!(
                            "Filtered impossible motion ({reason}); holding position [{} filtered so far]",
                            self.filtered_total
                        ),
                    );
                }
            }
            if self.violations.len() as i64 >= self.p.burst_count {
                self.violations.clear();
                let why = format!("{} impossible motions within {:.0} s", self.p.burst_count, self.p.burst_window);
                self.recover(why, now, io);
                if self.state != Phase::Idle {
                    return;
                }
            }
        }
        let accepted = reason.is_none();
        if accepted {
            if let Some(last) = self.last {
                let dt = t - last.t;
                for i in 0..3 {
                    self.vel[i] = 0.7 * self.vel[i] + 0.3 * (pos[i] - last.pos[i]) / dt;
                }
                if let Some(g) = q_g {
                    self.learn_gyro_bias(quat_mul(quat_conj(last.quat), quat), g, dt);
                }
            }
        }

        if let Some(leveled) = self.level(quat) {
            quat = leveled;
            self.offset = compose(&(pos, quat), &invert(&raw));
        }

        if pos[0].hypot(pos[1]) > self.p.boundary || pos[2].abs() > self.p.max_height {
            let why = format!("position ({:.1}, {:.1}, {:.1}) m outside boundary", pos[0], pos[1], pos[2]);
            self.recover(why, now, io);
            return;
        }
        // Only in-bounds, accepted poses from while visual tracking was OK can become a recovery
        // anchor (IMU-only poses after tracking fails drift).
        let current = Stamped { t, pos, quat };
        if accepted && self.fail_since.is_none() && now - self.last_vo_ok < 0.5 {
            push_bounded(&mut self.history, (now, current), HISTORY_LEN);
        }
        self.last = Some(current);
        push_bounded(&mut self.recent, current, RECENT_LEN);
        let vio_ok = accepted && self.fail_since.is_none();
        self.publish(msg, pos, quat, vio_ok, io);
    }

    fn publish(&mut self, msg: &Odom, pos: Vec3, quat: Quat, vio_ok: bool, io: &mut dyn Io) {
        let (pos, quat) = if self.p.planar {
            // Ground robot: it can't leave the floor, so publish x, y and heading only.
            ([pos[0], pos[1], 0.0], yaw_quat(quat_yaw(quat)))
        } else {
            (pos, quat)
        };
        if !self.p.publish_tf {
            // EKF mode: forward only VIO poses that passed every check with tracking OK.
            if vio_ok {
                self.forward(msg.stamp, pos, quat, io);
            }
            return;
        }
        io.tf(msg.stamp, (pos, quat), self.offset);
        io.odom(Odom { stamp: msg.stamp, pos, quat, pose_covariance: msg.pose_covariance, twist: msg.twist.clone() });
    }

    fn forward(&mut self, stamp: Stamp, pos: Vec3, quat: Quat, io: &mut dyn Io) {
        let (pv, yv) = (self.p.vio_pos_var, self.p.vio_yaw_var);
        io.odom(Odom { stamp, pos, quat, pose_covariance: diag_cov([pv, pv, pv, yv, yv, yv]), twist: Twist::default() });
        self.forwarded += 1;
    }

    /// /vslam_filter/status JSON (same keys, order and number format as json.dumps in Python).
    pub fn status(&self, now: f64) -> String {
        let tilt = match (self.last, self.acc_lp) {
            (Some(l), Some(_)) => self.tilt_deg(l.quat).map_or("null".into(), |d| json_float(py_round(d, 1))),
            _ => "null".into(),
        };
        let wheels = self.wheel_buf.back().is_some_and(|w| self.wheel_pose(w.t).is_some()) && now - self.last_wheel < 1.0;
        format!(
            "{{\"state\": \"{}\", \"filtered\": {}, \"recoveries\": {}, \"speed_mps\": {}, \"gyro\": {}, \
             \"gyro_bias_dps\": {}, \"tilt_deg\": {}, \"wheels\": {}, \"bridged\": {}, \"wheel_followed\": {}, \
             \"output\": \"{}\", \"forwarded\": {}}}",
            self.state.as_str(),
            self.filtered_total,
            self.recoveries,
            json_float(py_round(norm(self.vel), 2)),
            now - self.last_imu < 1.0,
            json_float(py_round(norm(self.gyro_bias) * RAD2DEG, 2)),
            tilt,
            wheels,
            self.bridged,
            self.wheel_followed,
            if self.p.publish_tf { "tf" } else { "ekf" },
            self.forwarded,
        )
    }

    /// Most recent trustworthy pose from before the failure started.
    fn last_good_pose(&self, now: f64) -> Option<Stamped> {
        let cutoff = now - 0.3;
        self.history.iter().rev().find(|(mono, _)| *mono <= cutoff).map(|(_, s)| *s)
    }

    // ---------- recovery sequence ----------

    fn recover(&mut self, reason: String, now: f64, io: &mut dyn Io) {
        if self.state != Phase::Idle || now - self.last_recovery < self.p.cooldown {
            return;
        }
        if !io.service_ready(Svc::Reset) {
            io.log(Level::Error, format!("Cannot recover ({reason}): /visual_slam/reset unavailable"));
            self.last_recovery = now;
            return;
        }
        self.recoveries += 1;
        self.last_recovery = now;
        self.anchor = if self.p.new_map_always { None } else { self.last_good_pose(now) };
        // With wheels, keep the robot's pose moving from the anchor until the VIO is back.
        self.bridge = if self.wheel_buf.is_empty() { None } else { self.anchor };
        self.bridge_t = 0.0;
        let how = if self.anchor.is_some() { "continuing from last good pose" } else { "starting a new map session" };
        io.log(Level::Warn, format!("Recovery #{}: {reason} ({how})", self.recoveries));
        self.enter(Phase::Pausing, now);
        self.call(Svc::Pause, now, io);
    }

    /// The response of a service called through [`Io::call`] arrived (`error`: the call failed).
    pub fn service_done(&mut self, svc: Svc, error: Option<String>, now: f64, io: &mut dyn Io) {
        if let Some(e) = error {
            io.log(Level::Error, format!("{} failed: {e}", self.p.service_name(svc)));
        }
        match svc {
            Svc::Pause => self.after_pause(now, io),
            Svc::Reset => self.after_reset(now),
            Svc::NewMap => self.resume(now, io),
            Svc::Resume => self.done(now, io),
        }
    }

    fn after_pause(&mut self, now: f64, io: &mut dyn Io) {
        self.enter(Phase::Resetting, now);
        self.call(Svc::Reset, now, io);
    }

    fn after_reset(&mut self, now: f64) {
        // The VIO restarts in a fresh vslam_odom: its first pose after the reset is placed at
        // the anchor (heading advanced by the gyro), see on_odom.
        self.offset = POSE_IDENTITY;
        self.pending_anchor = self.anchor;
        self.last = None;
        self.recent.clear();
        self.vel = [0.0; 3];
        self.enter(Phase::Waiting, now);
    }

    /// 10 Hz timer.
    pub fn tick(&mut self, now: f64, io: &mut dyn Io) {
        if self.state == Phase::Idle {
            return;
        }
        self.publish_bridge(io);
        let elapsed = now - self.state_since;
        if self.state == Phase::Waiting {
            // Healthy again: several consecutive OK statuses after the reset.
            if self.ok_count >= 5 || elapsed > self.p.recover_timeout {
                if self.ok_count < 5 {
                    io.log(Level::Warn, "Tracking not healthy yet; resuming anyway".into());
                }
                if self.anchor.is_none() {
                    self.enter(Phase::NewMap, now);
                    self.call(Svc::NewMap, now, io);
                } else {
                    self.resume(now, io);
                }
            }
        } else if elapsed > self.p.recover_timeout {
            io.log(Level::Error, format!("Recovery stuck in \"{}\"; resuming mapping", self.state.as_str()));
            self.resume(now, io);
        }
    }

    fn resume(&mut self, now: f64, io: &mut dyn Io) {
        self.enter(Phase::Resuming, now);
        self.call(Svc::Resume, now, io);
    }

    fn done(&mut self, now: f64, io: &mut dyn Io) {
        self.history.clear();
        self.violations.clear();
        self.fail_since = None;
        self.bridge = None;
        let how = if self.anchor.is_some() { "same map, continued from last good pose" } else { "new map session started" };
        self.enter(Phase::Idle, now);
        io.log(Level::Info, format!("Recovery #{} complete: {how}", self.recoveries));
    }

    fn enter(&mut self, state: Phase, now: f64) {
        self.state = state;
        self.state_since = now;
        if state == Phase::Waiting {
            self.ok_count = 0;
        }
    }

    /// Calls a service; continues right away (skipping it) if it is unavailable.
    fn call(&mut self, svc: Svc, now: f64, io: &mut dyn Io) {
        if !io.service_ready(svc) {
            io.log(Level::Warn, format!("{} unavailable, skipping", self.p.service_name(svc)));
            self.service_done(svc, None, now, io);
            return;
        }
        io.call(svc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Rec {
        unavailable: Vec<Svc>,
        tfs: Vec<(Stamp, Pose, Pose)>,
        odoms: Vec<Odom>,
        calls: Vec<Svc>,
        logs: Vec<(Level, String)>,
    }

    impl Io for Rec {
        fn service_ready(&self, svc: Svc) -> bool {
            !self.unavailable.contains(&svc)
        }
        fn tf(&mut self, stamp: Stamp, base: Pose, anchor: Pose) {
            self.tfs.push((stamp, base, anchor));
        }
        fn odom(&mut self, odom: Odom) {
            self.odoms.push(odom);
        }
        fn call(&mut self, svc: Svc) {
            self.calls.push(svc);
        }
        fn log(&mut self, level: Level, msg: String) {
            self.logs.push((level, msg));
        }
    }

    fn vio(t: f64, x: f64) -> Odom {
        Odom { stamp: Stamp::from_secs(t), pos: [x, 0.0, 0.0], quat: IDENTITY, pose_covariance: [0.0; 36], twist: Twist::default() }
    }

    fn robot() -> Params {
        let mut p = Params::load(&mut Defaults).unwrap();
        p.max_speed = 0.6;
        p.max_accel = 1.5;
        p.planar = true;
        p
    }

    #[test]
    fn rounding_and_json_floats_match_python() {
        // round(x, 2) / json.dumps in Python 3.12
        assert_eq!(json_float(py_round(0.125, 2)), "0.12");
        assert_eq!(json_float(py_round(0.375, 2)), "0.38");
        assert_eq!(json_float(py_round(2.675, 2)), "2.67");
        assert_eq!(json_float(py_round(1.005, 2)), "1.0");
        assert_eq!(json_float(py_round(-0.001, 2)), "-0.0");
        assert_eq!(Stamp::from_secs(1.5), Stamp { sec: 1, nanosec: 500_000_000 });
    }

    #[test]
    fn fresh_status_json() {
        let w = Watchdog::new(Params::load(&mut Defaults).unwrap(), 100.0);
        assert_eq!(
            w.status(100.0),
            "{\"state\": \"idle\", \"filtered\": 0, \"recoveries\": 0, \"speed_mps\": 0.0, \"gyro\": false, \
             \"gyro_bias_dps\": 0.0, \"tilt_deg\": null, \"wheels\": false, \"bridged\": 0, \"wheel_followed\": 0, \
             \"output\": \"tf\", \"forwarded\": 0}"
        );
    }

    #[test]
    fn gyro_delta_integrates_the_rate() {
        let mut w = Watchdog::new(robot(), 0.0);
        let mut io = Rec::default();
        let z90: Quat = [0.0, 0.0, std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2];
        for i in 0..=400 {
            // IMU frame rotated 90 deg about z from base: imu -y is base +x.
            w.on_imu(i as f64 * 0.005, [0.0, -1.0, 0.0], [0.0, 0.0, 9.81], 0.0, || Some(z90), &mut io);
        }
        assert_eq!(io.logs.len(), 1);
        let q = w.gyro_delta(0.5, 1.5).unwrap();
        let rv = quat_to_rotvec(q);
        assert!((rv[0] - 1.0).abs() < 1e-9 && rv[1].abs() < 1e-9 && rv[2].abs() < 1e-9, "{rv:?}");
        assert!(w.gyro_delta(1.5, 2.5).is_none(), "not covered by the buffer");
    }

    #[test]
    fn jump_is_filtered_and_motion_continues_from_the_held_pose() {
        let mut w = Watchdog::new(robot(), 0.0);
        let mut io = Rec::default();
        for i in 0..10 {
            w.on_odom(&vio(i as f64 * 0.1, i as f64 * 0.01), 0.0, &mut io);
        }
        // 2 m jump, then normal motion continues in the jumped frame.
        w.on_odom(&vio(1.0, 2.1), 10.0, &mut io);
        w.on_odom(&vio(1.1, 2.11), 10.1, &mut io);
        assert_eq!(w.filtered_total, 1);
        assert!(io.logs[0].1.starts_with("Filtered impossible motion (speed 20.1 m/s)"), "{:?}", io.logs);
        let held = io.odoms[10].pos[0];
        assert!((held - 0.1).abs() < 1e-3, "held at the predicted position: {held}");
        assert!((io.odoms[11].pos[0] - held - 0.01).abs() < 1e-9);
        // odom -> vslam_odom moved back by the jump.
        assert!((io.tfs[11].2 .0[0] + 2.0).abs() < 1e-3);
    }

    #[test]
    fn ekf_mode_forwards_only_accepted_poses_while_tracking() {
        let mut p = robot();
        p.publish_tf = false;
        let mut w = Watchdog::new(p, 0.0);
        let mut io = Rec::default();
        for i in 0..5 {
            w.on_odom(&vio(i as f64 * 0.1, i as f64 * 0.01), 0.0, &mut io);
        }
        w.on_odom(&vio(0.5, 3.0), 0.0, &mut io);
        assert_eq!((io.odoms.len(), w.forwarded, io.tfs.len()), (5, 5, 0));
        assert_eq!(io.odoms[0].pose_covariance[0], 0.002);
        assert_eq!(io.odoms[0].pose_covariance[35], 0.0005);
        assert!(w.status(0.0).contains("\"output\": \"ekf\", \"forwarded\": 5"));
    }

    #[test]
    fn tracking_loss_runs_the_recovery_sequence() {
        let mut w = Watchdog::new(robot(), 1000.0);
        let mut io = Rec { unavailable: vec![Svc::Pause, Svc::Resume], ..Default::default() };
        for i in 0..20 {
            let now = 1000.0 + i as f64 * 0.1;
            w.on_status(VO_OK, now, &mut io);
            w.on_odom(&vio(i as f64 * 0.1, i as f64 * 0.01), now, &mut io);
        }
        for i in 0..12 {
            w.on_status(VO_FAILED, 1002.0 + i as f64 * 0.1, &mut io);
        }
        // Pause unavailable (skipped), reset called.
        assert_eq!(io.calls, vec![Svc::Reset]);
        assert_eq!(w.state, Phase::Resetting);
        assert!(io.logs.iter().any(|l| l.1 == "Recovery #1: visual tracking lost for 1.1 s (continuing from last good pose)"));
        assert!(io.logs.iter().any(|l| l.1 == "/rtabmap/rtabmap/pause unavailable, skipping"));
        w.service_done(Svc::Reset, None, 1003.3, &mut io);
        assert_eq!(w.state, Phase::Waiting);
        for i in 0..5 {
            w.on_status(VO_OK, 1003.4 + i as f64 * 0.1, &mut io);
        }
        w.tick(1004.0, &mut io);
        assert_eq!(w.state, Phase::Idle);
        assert_eq!(io.logs.last().unwrap().1, "Recovery #1 complete: same map, continued from last good pose");
        // The first pose after the reset continues from the anchor, not from the new VIO origin.
        w.on_odom(&vio(3.5, 0.0), 1004.1, &mut io);
        let p = io.odoms.last().unwrap().pos;
        assert!(p[0] > 0.1 && p[0] < 0.2, "{p:?}");
        assert!(w.status(1004.1).contains("\"recoveries\": 1"));
    }
}
