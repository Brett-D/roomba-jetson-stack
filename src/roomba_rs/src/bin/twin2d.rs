//! twin2d: the 2D digital twin of the house and the Roomba, for running the real navigation stack
//! (roomba_driver, floor_obstacles, imu_filter_madgwick, ignore_zones, map_gap_filler, Nav2,
//! explorer, nav_coach, web_server) on a simulated robot. Started by
//! roomba_bringup/launch/twin2d.launch.py (scripts/run_twin2d.sh); the logic lives in roomba_rs::twin.
//!
//! In place of the hardware and the localisation it provides:
//!   - the Roomba: its Open Interface on a pty (`pty_link`, the driver's port): modes, sensor
//!     streams in both layouts, encoders, bumpers, light bumpers, cliffs, battery;
//!   - the camera: /camera/depth/image_rect_raw + camera_info (15 Hz), /camera/imu/data_raw
//!     (200 Hz), the camera driver's static TF, a synthetic /camera/left/image_rect_mono view;
//!   - localisation: map -> odom -> base_link on /tf and /odom (ground truth, optionally drifting),
//!     /vslam_filter/status and /visual_slam/status like vslam_watchdog and cuVSLAM;
//!   - RTAB-Map: /rtabmap/map (latched, revealed from what the camera saw) and /rtabmap/mapPath.
//! It also drives the scenario (doors, people, battery and tracking events), starts the explorer
//! and plays the operator (re-enables driving after a cliff stop), and measures the outcome:
//! result.json, timeline.csv, events.jsonl and trajectory.png in `out_dir` when the explorer is done,
//! the time is up, or on SIGINT/SIGTERM. Topics of its own: /twin/state (JSON, 2 Hz),
//! /twin/events (JSON per event), /twin/ground_truth (Odometry, map frame), /twin/world (the true
//! floor plan with closed doors and obstacles, latched).
//!
//! Real time only: the driver runs on wall-clock timers (see README, "Digital twin (2D)").

use std::collections::HashMap;
use std::ffi::CString;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{bail, Context as _, Result};
use isaac_ros_visual_slam_interfaces::msg::VisualSlamStatus;
use rclrs::*;
use ros_env::{action_msgs, builtin_interfaces, geometry_msgs, nav2_msgs, nav_msgs, rcl_interfaces, roomba_msgs, sensor_msgs, std_msgs, tf2_msgs, vision_msgs};
use serde_json::{json, Value};

use roomba_rs::node_params::{block_signals, mono, on_signal, NodeParams};
use roomba_rs::oi;
use roomba_rs::twin::depth::{self, Camera, LensPose, RenderOpts, D1000};
use roomba_rs::twin::mapper::{Coverage, Mapper};
use roomba_rs::twin::metrics::{self, driving_state, round, CostGrid, Metrics, Sample};
use roomba_rs::twin::oi_sim::OiSim;
use roomba_rs::twin::physics::{wrap, Pack, Robot, StepEvent};
use roomba_rs::twin::rng::Rng;
use roomba_rs::twin::scenario::{person_position, Scenario, WorldOverlay};
use roomba_rs::twin::world::{Body, Grid, Kind, World, P2};
use roomba_rs::watchdog::ParamSource;

const DT: f64 = 0.005;

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn time_msg(ns: i64) -> builtin_interfaces::msg::Time {
    builtin_interfaces::msg::Time { sec: ns.div_euclid(1_000_000_000) as i32, nanosec: ns.rem_euclid(1_000_000_000) as u32 }
}

fn header(ns: i64, frame: &str) -> std_msgs::msg::Header {
    std_msgs::msg::Header { stamp: time_msg(ns), frame_id: frame.into() }
}

fn quat(q: [f64; 4]) -> geometry_msgs::msg::Quaternion {
    geometry_msgs::msg::Quaternion { x: q[0], y: q[1], z: q[2], w: q[3] }
}

fn yaw_quat(yaw: f64) -> geometry_msgs::msg::Quaternion {
    quat([0.0, 0.0, (yaw / 2.0).sin(), (yaw / 2.0).cos()])
}

fn transform(ns: i64, parent: &str, child: &str, t: [f64; 3], q: geometry_msgs::msg::Quaternion) -> geometry_msgs::msg::TransformStamped {
    geometry_msgs::msg::TransformStamped {
        header: header(ns, parent),
        child_frame_id: child.into(),
        transform: geometry_msgs::msg::Transform { translation: geometry_msgs::msg::Vector3 { x: t[0], y: t[1], z: t[2] }, rotation: q },
    }
}

/// a * b for planar poses (x, y, yaw).
fn compose(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    let (c, s) = (a[2].cos(), a[2].sin());
    [a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2])]
}

fn inverse(a: [f64; 3]) -> [f64; 3] {
    let (c, s) = (a[2].cos(), a[2].sin());
    [-(c * a[0] + s * a[1]), s * a[0] - c * a[1], -a[2]]
}

// ---------------------------------------------------------------- pty

/// The Roomba's serial port: a pty whose slave end is symlinked at `link` for the driver.
struct Pty {
    master: OwnedFd,
    _slave: OwnedFd,
    name: String,
}

impl Pty {
    fn open(link: &str) -> Result<Pty> {
        // SAFETY: plain libc calls on fds this function owns; buffers are sized and NUL-terminated.
        unsafe {
            let m = libc::posix_openpt(libc::O_RDWR | libc::O_NOCTTY);
            if m < 0 {
                bail!("posix_openpt: {}", std::io::Error::last_os_error());
            }
            let master = OwnedFd::from_raw_fd(m);
            if libc::grantpt(m) != 0 || libc::unlockpt(m) != 0 {
                bail!("grantpt/unlockpt: {}", std::io::Error::last_os_error());
            }
            let mut buf = [0 as libc::c_char; 128];
            if libc::ptsname_r(m, buf.as_mut_ptr(), buf.len()) != 0 {
                bail!("ptsname_r: {}", std::io::Error::last_os_error());
            }
            let name = std::ffi::CStr::from_ptr(buf.as_ptr()).to_string_lossy().to_string();
            let cname = CString::new(name.clone())?;
            let s = libc::open(cname.as_ptr(), libc::O_RDWR | libc::O_NOCTTY);
            if s < 0 {
                bail!("open {name}: {}", std::io::Error::last_os_error());
            }
            // Kept open: the pty survives the driver closing and reopening it.
            let slave = OwnedFd::from_raw_fd(s);
            let mut t: libc::termios = std::mem::zeroed();
            libc::tcgetattr(s, &mut t);
            libc::cfmakeraw(&mut t);
            libc::tcsetattr(s, libc::TCSANOW, &t);
            let flags = libc::fcntl(m, libc::F_GETFL);
            libc::fcntl(m, libc::F_SETFL, flags | libc::O_NONBLOCK);
            let link_path = std::path::Path::new(link);
            if let Ok(meta) = std::fs::symlink_metadata(link_path) {
                if !meta.file_type().is_symlink() {
                    bail!("{link} exists and is not a symlink; not replacing it");
                }
            }
            let tmp = format!("{link}.new");
            let _ = std::fs::remove_file(&tmp);
            std::os::unix::fs::symlink(&name, &tmp).with_context(|| format!("symlink {tmp}"))?;
            std::fs::rename(&tmp, link_path).with_context(|| format!("rename to {link}"))?;
            Ok(Pty { master, _slave: slave, name })
        }
    }

    fn read(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let mut buf = [0u8; 512];
        loop {
            // SAFETY: reads into a local buffer of the given size.
            let n = unsafe { libc::read(self.master.as_raw_fd(), buf.as_mut_ptr().cast(), buf.len()) };
            if n <= 0 {
                break;
            }
            out.extend_from_slice(&buf[..n as usize]);
        }
        out
    }

    /// False if the pty buffer is full (the driver isn't reading).
    fn write(&self, data: &[u8]) -> bool {
        // SAFETY: writes from a slice of the given length.
        let n = unsafe { libc::write(self.master.as_raw_fd(), data.as_ptr().cast(), data.len()) };
        n == data.len() as isize
    }
}

// ---------------------------------------------------------------- shared state

/// Filled by the ROS callbacks (executor thread), read by the simulation thread.
#[derive(Default)]
struct Shared {
    explorer: Option<Value>,
    robot_mode: Option<String>,
    battery_pct: Option<f64>,
    /// navigate_to_pose goals: uuid -> last status; and the changes for the sim thread.
    goals: HashMap<[u8; 16], i8>,
    goal_updates: Vec<([u8; 16], i8)>,
    /// (behavior, uuid) seen
    behaviors: HashMap<(String, [u8; 16]), i8>,
    behavior_updates: Vec<String>,
    logs: Vec<(String, u8, String)>,
    cm_state: u8,
    cm_stops: u64,
    coverage: Coverage,
    map_nodes: usize,
    set_mode_requests: Vec<u8>,
    set_mode_results: Vec<String>,
    depth_frames: u64,
    depth_skipped: u64,
    render_ms: f64,
    costmap: Option<CostGrid>,
}

impl Shared {
    /// A navigate_to_pose goal is accepted or executing.
    fn goal_active(&self) -> bool {
        self.goals.values().any(|&s| s == 1 || s == 2)
    }
}

struct Snapshot {
    stamp: i64,
    t: f64,
    lens_true: LensPose,
    lens_est: LensPose,
    base_est: [f64; 3],
    bodies: Vec<Body>,
    tracking_ok: bool,
    preview: bool,
}

struct Pubs {
    tf: Publisher<tf2_msgs::msg::TFMessage>,
    odom: Publisher<nav_msgs::msg::Odometry>,
    truth: Publisher<nav_msgs::msg::Odometry>,
    imu: Publisher<sensor_msgs::msg::Imu>,
    filter_status: Publisher<std_msgs::msg::String>,
    vslam_status: Publisher<VisualSlamStatus>,
    state: Publisher<std_msgs::msg::String>,
    events: Publisher<std_msgs::msg::String>,
    explorer_cmd: Publisher<std_msgs::msg::String>,
    world: Publisher<nav_msgs::msg::OccupancyGrid>,
    persons: Publisher<vision_msgs::msg::Detection3DArray>,
}

struct RenderPubs {
    depth: Publisher<sensor_msgs::msg::Image>,
    depth_info: Publisher<sensor_msgs::msg::CameraInfo>,
    left: Publisher<sensor_msgs::msg::Image>,
    left_info: Publisher<sensor_msgs::msg::CameraInfo>,
    map: Publisher<nav_msgs::msg::OccupancyGrid>,
    path: Publisher<nav_msgs::msg::Path>,
}

fn camera_info(stamp: i64) -> sensor_msgs::msg::CameraInfo {
    let k = D1000;
    sensor_msgs::msg::CameraInfo {
        header: header(stamp, "camera_left_optical_frame"),
        height: k.height as u32,
        width: k.width as u32,
        distortion_model: "plumb_bob".into(),
        d: vec![0.0; 5],
        k: [k.fx, 0.0, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0],
        r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        p: [k.fx, 0.0, k.cx, 0.0, 0.0, k.fy, k.cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        ..Default::default()
    }
}

fn grid_msg(stamp: i64, g: &Grid, data: Vec<i8>) -> nav_msgs::msg::OccupancyGrid {
    let mut m = nav_msgs::msg::OccupancyGrid::default();
    m.header = header(stamp, "map");
    m.info.map_load_time = time_msg(stamp);
    m.info.resolution = g.res as f32;
    m.info.width = g.width as u32;
    m.info.height = g.height as u32;
    m.info.origin.position.x = g.origin[0];
    m.info.origin.position.y = g.origin[1];
    m.info.origin.orientation.w = 1.0;
    m.data = data;
    m
}

/// The true floor plan with what is in the way now (closed doors, obstacles), for /twin/world.
fn world_occupancy(world: &World) -> Vec<i8> {
    let g = &world.grid;
    let mut data: Vec<i8> = g.solid.iter().map(|&s| if s { 100 } else { 0 }).collect();
    for b in &world.bodies {
        if !b.active || !matches!(b.kind, Kind::Door | Kind::Obstacle | Kind::Railing | Kind::Dock) {
            continue;
        }
        let bb = b.shape.bbox();
        let (i0, j0) = g.cell([bb[0], bb[1]]);
        let (i1, j1) = g.cell([bb[2], bb[3]]);
        for j in j0..=j1 {
            for i in i0..=i1 {
                if let Some(k) = g.idx(i, j) {
                    if b.shape.nearest(g.cell_center(i, j)).0 <= g.res * 0.5 {
                        data[k] = if b.kind == Kind::Door { 90 } else { 70 };
                    }
                }
            }
        }
    }
    data
}

// ---------------------------------------------------------------- odometry estimate

/// odom -> base_link as the odometry believes (truth plus drift) and map -> odom (RTAB-Map's
/// correction, snapped to the truth every correction period).
struct Localization {
    odom: [f64; 3],
    map_odom: [f64; 3],
    last_truth: [f64; 3],
    last_fix: f64,
    v: f64,
    w: f64,
}

impl Localization {
    fn new(start: [f64; 3]) -> Self {
        Localization { odom: [0.0; 3], map_odom: start, last_truth: start, last_fix: 0.0, v: 0.0, w: 0.0 }
    }

    fn update(&mut self, truth: [f64; 3], t: f64, s: &Scenario) {
        let rel = compose(inverse(self.last_truth), truth);
        self.last_truth = truth;
        let ds = rel[0].hypot(rel[1]);
        let o = &s.odometry;
        if ds < 0.2 {
            let ds_e = ds * (1.0 + o.scale_error);
            let dth = rel[2] + o.yaw_drift_deg_per_m.to_radians() * ds + o.yaw_bias_deg_s.to_radians() * DT;
            let dir = if ds > 1e-12 { rel[1].atan2(rel[0]) } else { 0.0 };
            let step = [ds_e * dir.cos(), ds_e * dir.sin(), dth];
            self.odom = compose(self.odom, step);
            self.v = ds_e * rel[0].signum() / DT;
            self.w = dth / DT;
        }
        let period = o.correction_period_s;
        if period > 0.0 && t - self.last_fix >= period {
            self.map_odom = compose(truth, inverse(self.odom));
            self.last_fix = t;
        }
    }

    fn map_base(&self) -> [f64; 3] {
        compose(self.map_odom, self.odom)
    }
}

// ---------------------------------------------------------------- the simulation

struct Sim {
    node: Node,
    scenario: Scenario,
    world: World,
    door_ids: Vec<usize>,
    people_ids: Vec<usize>,
    robot: Robot,
    pack: Pack,
    oi: OiSim,
    pty: Pty,
    loc: Localization,
    rng: Rng,
    /// Person detections draw from their own stream (the depth noise stays as it was).
    person_rng: Rng,
    mount: [f64; 3],
    q_bi: [f64; 4],
    pubs: Pubs,
    shared: Arc<Mutex<Shared>>,
    mapper: Arc<Mutex<Mapper>>,
    render_tx: mpsc::SyncSender<Snapshot>,
    metrics: Metrics,
    out_dir: String,
    world_path: String,
    /// Explorer parameter overrides of an experiment (twin2d.launch.py explorer_params).
    explorer_params: serde_json::Map<String, Value>,
    clock: Clock,
    // timeline
    event_done: Vec<bool>,
    tracking_lost_until: f64,
    explorer_started: bool,
    start_attempts: u32,
    next_start_try: f64,
    explorer_state: String,
    explorer_message: String,
    explorer_since: f64,
    interventions: u32,
    intervene_at: Option<f64>,
    restart_explorer_at: Option<f64>,
    finished: Option<(String, String)>,
    finished_pending: Option<(String, String)>,
    finish_at: Option<f64>,
    last_explorer_log: Option<String>,
    goal_start: HashMap<[u8; 16], (f64, P2, Option<P2>)>,
    goal_counts: [u64; 4],
    false_successes: Vec<Value>,
    battery_first: Option<f64>,
    battery_min: Option<f64>,
    last_bump: (bool, bool),
    /// A pressed bumper with nothing there (scenario bump events) until this time.
    forced_bump: (bool, bool),
    forced_bump_until: f64,
    world_dirty: bool,
    wall_start: Instant,
}

impl Sim {
    fn log(&self, msg: &str) {
        log!(self.node.info(), "{msg}");
    }

    fn event(&mut self, t: f64, kind: &str, pos: P2, detail: String) {
        let data = json!({"t": round(t, 2), "kind": kind, "x": round(pos[0], 3), "y": round(pos[1], 3), "detail": detail}).to_string();
        self.pubs.events.publish(std_msgs::msg::String { data }).ok();
        self.metrics.event(t, kind, pos, detail);
    }

    fn pos(&self) -> P2 {
        [self.robot.x, self.robot.y]
    }

    /// Raw values of every OI packet (7..=58) now.
    fn raw_packets(&mut self) -> [i32; 59] {
        let r = &self.robot;
        let mut v = [0i32; 59];
        let bump_right = r.bump.1 || r.cfg.stuck_bump_right || self.forced_bump.1;
        let bump_left = r.bump.0 || self.forced_bump.0;
        let drop = r.fallen as i32;
        v[7] = bump_right as i32 | (bump_left as i32) << 1 | drop << 2 | drop << 3;
        v[8] = (r.wall_signal > 40) as i32;
        for k in 0..4 {
            v[9 + k] = r.cliff[k] as i32;
        }
        let covers = |id: u8| self.oi.layout.iter().any(|&p| oi::packet_range(p).is_ok_and(|rg| rg.contains(&id)));
        if covers(19) {
            let d = self.robot.odo_mm.round();
            self.robot.odo_mm -= d;
            v[19] = d.clamp(-32768.0, 32767.0) as i32;
        }
        if covers(20) {
            let a = self.robot.odo_deg.round();
            self.robot.odo_deg -= a;
            v[20] = a.clamp(-32768.0, 32767.0) as i32;
        }
        let r = &self.robot;
        let p = &self.pack;
        v[21] = 0;
        v[22] = (p.voltage() * 1000.0).round() as i32;
        v[23] = -(p.current * 1000.0).round() as i32;
        v[24] = 25;
        v[25] = (p.soc * p.cfg.capacity_ah * 1000.0).round() as i32;
        v[26] = (p.cfg.capacity_ah * 1000.0).round() as i32;
        v[27] = r.wall_signal as i32;
        for k in 0..4 {
            v[28 + k] = if r.cliff[k] { 300 } else { 2850 } + (self.rng.gauss(15.0)).round() as i32;
        }
        v[35] = self.oi.mode as i32;
        v[38] = self.oi.layout.len() as i32;
        v[39..43].copy_from_slice(&self.oi.requested);
        v[43] = (r.enc.0.round() as i64).rem_euclid(65536) as i32;
        v[44] = (r.enc.1.round() as i64).rem_euclid(65536) as i32;
        let mut bits = 0;
        for k in 0..6 {
            if r.light[k] > 100 {
                bits |= 1 << k;
            }
            v[46 + k] = r.light[k] as i32;
        }
        v[45] = bits;
        let stall = if r.stalled { 600.0 } else { 0.0 };
        v[54] = ((r.vl.abs() / 0.3 * 150.0) + stall).round() as i32;
        v[55] = ((r.vr.abs() / 0.3 * 150.0) + stall).round() as i32;
        v[58] = (r.v > 0.02) as i32;
        v
    }

    fn unsafe_now(&self) -> bool {
        self.robot.cliff.iter().any(|&c| c) || self.robot.fallen
    }

    fn explorer_command(&mut self, cmd: &str, t: f64) {
        self.pubs.explorer_cmd.publish(std_msgs::msg::String { data: cmd.into() }).ok();
        let pos = self.pos();
        self.event(t, "explorer_command", pos, cmd.into());
    }

    /// Scenario timeline: doors, obstacles appearing, people, events.
    fn timeline(&mut self, t: f64) {
        let pos = self.pos();
        let r = self.robot.cfg.radius_m;
        for (n, d) in self.scenario.doors.clone().iter().enumerate() {
            let id = self.door_ids[n];
            let mut want = d.closed_at_start();
            for e in &d.events {
                if t >= e.t {
                    want = e.state == "closed";
                }
            }
            if want != self.world.bodies[id].active {
                // Never close a door on the robot: wait until it is clear of the doorway.
                if want && self.world.bodies[id].shape.nearest(pos).0 < r + 0.02 {
                    continue;
                }
                self.world.bodies[id].active = want;
                self.world_dirty = true;
                let detail = format!("{} {}", d.name, if want { "closed" } else { "opened" });
                self.log(&format!("door {detail}"));
                self.event(t, "door", self.world.bodies[id].shape.centroid(), detail);
            }
        }
        let obstacle_ids: Vec<usize> = self.world.bodies.iter().enumerate().filter(|(_, b)| b.kind == Kind::Obstacle).map(|(k, _)| k).collect();
        for (o, &id) in self.scenario.obstacles.clone().iter().zip(&obstacle_ids) {
            let want = t >= o.from_s && o.until_s.is_none_or(|u| t < u);
            if want != self.world.bodies[id].active {
                if want && self.world.bodies[id].shape.nearest(pos).0 < r {
                    continue;
                }
                self.world.bodies[id].active = want;
                self.world_dirty = true;
                self.event(t, "obstacle", self.world.bodies[id].shape.centroid(), format!("{} {}", o.name, if want { "appeared" } else { "removed" }));
            }
        }
        for (n, p) in self.scenario.people.clone().iter().enumerate() {
            let id = self.people_ids[n];
            let present = t >= p.from_s && p.until_s.is_none_or(|u| t < u);
            let active = self.world.bodies[id].active;
            if !present {
                if active {
                    self.world.bodies[id].active = false;
                    self.event(t, "person", pos, format!("{} left", p.name));
                }
                continue;
            }
            let target = person_position(p, t - p.from_s);
            // People don't walk into the robot: they wait (unless the scenario says they do).
            if !p.walks_into_robot && (target[0] - pos[0]).hypot(target[1] - pos[1]) < r + p.radius_m + 0.05 {
                continue;
            }
            self.world.bodies[id].shape = roomba_rs::twin::world::Shape::Circle { c: target, r: p.radius_m };
            if !active {
                self.world.bodies[id].active = true;
                self.event(t, "person", target, format!("{} appeared", p.name));
            }
        }
        if t >= self.forced_bump_until {
            self.forced_bump = (false, false);
        }
        for k in 0..self.scenario.events.len() {
            if self.event_done[k] || t < self.scenario.events[k].t {
                continue;
            }
            self.event_done[k] = true;
            let e = self.scenario.events[k].clone();
            if let Some(soc) = e.battery_soc {
                self.pack.soc = soc.clamp(0.0, 1.0);
                self.event(t, "battery", pos, format!("state of charge set to {:.0}%", soc * 100.0));
            }
            if let Some(s) = e.tracking_lost_s {
                self.tracking_lost_until = t + s;
                self.event(t, "tracking_lost", pos, format!("visual tracking lost for {s:.0} s"));
            }
            if let Some(cmd) = &e.explorer {
                self.explorer_command(cmd, t);
            }
            if let Some(p) = e.teleport {
                self.robot.teleport([p.x, p.y, p.yaw_deg.to_radians()]);
                self.event(t, "teleport", [p.x, p.y], "robot moved (kidnapped)".into());
            }
            if let Some(side) = &e.bump {
                self.forced_bump = (side != "right", side != "left");
                self.forced_bump_until = t + 0.3;
                self.metrics.count("phantom_bumps");
                self.event(t, "phantom_bump", pos, format!("bumper ({side}) pressed with nothing there"));
            }
            if e.end {
                self.finish(t, "ended", "the scenario's end event");
            }
        }
    }

    fn physics(&mut self, t: f64) {
        self.robot.target = self.oi.target;
        let events = self.robot.step(&self.world, DT);
        for e in events {
            match e {
                StepEvent::Collision { point, what, bearing } => {
                    let name = self.world.what_name(what);
                    let front = bearing.abs() <= std::f64::consts::FRAC_PI_2;
                    let kind = if front { "bump" } else { "contact" };
                    self.metrics.count(if front { "bumps" } else { "side_rear_contacts" });
                    if matches!(what, roomba_rs::twin::world::What::Body(k) if self.world.bodies[k].kind == Kind::Person) {
                        self.metrics.count("person_collisions");
                    }
                    self.event(t, kind, point, format!("{name} at {:+.0} deg", bearing.to_degrees()));
                }
                StepEvent::Fell { cliff } => {
                    let name = self.world.cliffs[cliff].0.clone();
                    let pos = self.pos();
                    self.event(t, "fell", pos, format!("drove over the edge of {name}"));
                    self.log(&format!("the robot fell down {name}"));
                    self.finish_later(t, 5.0, "fell", &format!("fell down {name}"));
                }
            }
        }
        if self.robot.bump != self.last_bump && (self.robot.bump.0 || self.robot.bump.1) {
            self.metrics.count("bumper_presses");
        }
        self.last_bump = self.robot.bump;
        let unsafe_now = self.unsafe_now();
        let before = self.oi.mode;
        self.oi.check_safety(unsafe_now);
        if before == oi::MODE_SAFE && self.oi.mode == oi::MODE_PASSIVE {
            let pos = self.pos();
            self.metrics.count("safety_stops");
            self.event(t, "safety_stop", pos, format!("cliff sensors {:?}, fallen {}", self.robot.cliff, self.robot.fallen));
        }
        self.pack.step(DT, self.robot.v, self.robot.stalled);
        self.metrics.track(self.pos());
        let p = self.pos();
        let near = self
            .world
            .bodies
            .iter()
            .filter(|b| b.active && matches!(b.kind, Kind::Railing | Kind::Post))
            .map(|b| b.shape.nearest(p).0)
            .fold(f64::INFINITY, f64::min);
        if near.is_finite() {
            self.metrics.railing(near - self.robot.cfg.radius_m, DT);
        }
    }

    fn publish_imu(&mut self, stamp: i64) {
        let s = &self.scenario.sensors;
        let (g, a) = depth::imu_sample(self.q_bi, self.robot.v, self.robot.w, self.robot.accel, s.gyro_bias, s.gyro_noise, s.accel_noise, &mut self.rng);
        let mut m = sensor_msgs::msg::Imu::default();
        m.header = header(stamp, "camera_imu_frame");
        m.orientation_covariance[0] = -1.0;
        m.angular_velocity = geometry_msgs::msg::Vector3 { x: g[0], y: g[1], z: g[2] };
        m.linear_acceleration = geometry_msgs::msg::Vector3 { x: a[0], y: a[1], z: a[2] };
        for k in [0, 4, 8] {
            m.angular_velocity_covariance[k] = 2.5e-5;
            m.linear_acceleration_covariance[k] = 0.0025;
        }
        self.pubs.imu.publish(m).ok();
    }

    /// odom -> base_link (and /odom) like vslam_watchdog; with `map_odom` also map -> odom like
    /// RTAB-Map, which stamps it tf_tolerance (0.1 s) ahead so lookups at "now" never extrapolate.
    fn publish_pose(&mut self, stamp: i64, map_odom: bool, with_truth: bool) {
        let o = self.loc.odom;
        let mo = self.loc.map_odom;
        let mut transforms = vec![transform(stamp, "odom", "base_link", [o[0], o[1], 0.0], yaw_quat(o[2]))];
        if map_odom {
            transforms.push(transform(stamp + 100_000_000, "map", "odom", [mo[0], mo[1], 0.0], yaw_quat(mo[2])));
        }
        self.pubs.tf.publish(tf2_msgs::msg::TFMessage { transforms }).ok();
        let mut od = nav_msgs::msg::Odometry::default();
        od.header = header(stamp, "odom");
        od.child_frame_id = "base_link".into();
        od.pose.pose.position.x = o[0];
        od.pose.pose.position.y = o[1];
        od.pose.pose.orientation = yaw_quat(o[2]);
        od.twist.twist.linear.x = self.loc.v;
        od.twist.twist.angular.z = self.loc.w;
        for k in [0, 7, 14, 21, 28, 35] {
            od.pose.covariance[k] = 1e-3;
            od.twist.covariance[k] = 1e-3;
        }
        self.pubs.odom.publish(od).ok();
        if with_truth {
            let mut tr = nav_msgs::msg::Odometry::default();
            tr.header = header(stamp, "map");
            tr.child_frame_id = "twin_base_link".into();
            tr.pose.pose.position.x = self.robot.x;
            tr.pose.pose.position.y = self.robot.y;
            tr.pose.pose.orientation = yaw_quat(self.robot.th);
            tr.twist.twist.linear.x = self.robot.v;
            tr.twist.twist.angular.z = self.robot.w;
            self.pubs.truth.publish(tr).ok();
        }
    }

    fn tracking_ok(&self, t: f64) -> bool {
        t >= self.tracking_lost_until
    }

    /// The people the robot's person detector would report now (in the camera's view, in range,
    /// not behind a wall; each with person_recall), as detections_3d in base_link.
    fn publish_persons(&mut self, stamp: i64) {
        let s = &self.scenario.sensors;
        let (recall, range, min) = (s.person_recall, s.person_range_m, s.depth_min_m);
        let (x, y, th) = (self.robot.x, self.robot.y, self.robot.th);
        let mut out = vision_msgs::msg::Detection3DArray { header: header(stamp, "base_link"), detections: Vec::new() };
        for (n, &id) in self.people_ids.iter().enumerate() {
            let b = &self.world.bodies[id];
            let roomba_rs::twin::world::Shape::Circle { c, .. } = b.shape else { continue };
            if !b.active {
                continue;
            }
            let (dx, dy) = (c[0] - x, c[1] - y);
            let d = dx.hypot(dy);
            let bearing = wrap(dy.atan2(dx) - th);
            if d < min || d > range || bearing.abs() > 0.7 {
                continue;
            }
            if self.world.grid.ray([x, y], [dx / d, dy / d], d) < d - 0.25 {
                continue; // a wall in between
            }
            if self.person_rng.uniform() >= recall {
                continue;
            }
            let mut det = vision_msgs::msg::Detection3D { header: out.header.clone(), id: self.scenario.people[n].name.clone(), ..Default::default() };
            det.bbox.center.position = geometry_msgs::msg::Point { x: d * bearing.cos(), y: d * bearing.sin(), z: 0.85 };
            det.bbox.center.orientation.w = 1.0;
            det.bbox.size = geometry_msgs::msg::Vector3 { x: 0.4, y: 0.4, z: 1.7 };
            let mut h = vision_msgs::msg::ObjectHypothesisWithPose::default();
            h.hypothesis.class_id = "person".into();
            h.hypothesis.score = 0.8;
            h.pose.pose.position = det.bbox.center.position.clone();
            det.results.push(h);
            out.detections.push(det);
        }
        self.pubs.persons.publish(out).ok();
    }

    fn publish_status(&mut self, stamp: i64, t: f64) {
        let ok = self.tracking_ok(t);
        let state = if ok { "idle" } else { "recovering" };
        let data = format!(
            "{{\"state\": \"{state}\", \"filtered\": 0, \"recoveries\": 0, \"speed_mps\": {:.2}, \"gyro\": true, \
             \"gyro_bias_dps\": 0.0, \"tilt_deg\": 0.0, \"wheels\": true, \"bridged\": 0, \"wheel_followed\": 0, \
             \"output\": \"tf\", \"forwarded\": 0, \"twin\": true}}",
            self.robot.v.abs()
        );
        self.pubs.filter_status.publish(std_msgs::msg::String { data }).ok();
        let mut s = VisualSlamStatus::default();
        s.header.stamp = time_msg(stamp);
        s.vo_state = if ok { 1 } else { 2 };
        self.pubs.vslam_status.publish(s).ok();
    }

    /// Reads what the ROS callbacks collected: explorer status, goals, recoveries, logs.
    fn absorb(&mut self, t: f64) {
        let (explorer, battery, goal_updates, behavior_updates, logs, cm_stops, results) = {
            let mut s = lock(&self.shared);
            (
                s.explorer.clone(),
                s.battery_pct,
                std::mem::take(&mut s.goal_updates),
                std::mem::take(&mut s.behavior_updates),
                std::mem::take(&mut s.logs),
                s.cm_stops,
                std::mem::take(&mut s.set_mode_results),
            )
        };
        let pos = self.pos();
        for r in results {
            self.event(t, "operator", pos, format!("set_mode answered: {r}"));
        }
        if let Some(b) = battery {
            self.battery_first.get_or_insert(b);
            self.battery_min = Some(self.battery_min.map_or(b, |m: f64| m.min(b)));
        }
        if let Some(e) = &explorer {
            let state = e.get("state").and_then(Value::as_str).unwrap_or("").to_string();
            let message = e.get("message").and_then(Value::as_str).unwrap_or("").to_string();
            if state != self.explorer_state {
                self.explorer_since = t;
                self.event(t, "explorer", pos, format!("{state}: {message}"));
                self.explorer_state = state;
                self.explorer_message = message.clone();
            } else if message != self.explorer_message {
                self.explorer_message = message.clone();
                if self.last_explorer_log.as_deref() != Some(message.as_str()) {
                    self.event(t, "explorer_message", pos, message.clone());
                    self.last_explorer_log = Some(message);
                }
            }
        }
        let goal_xy = explorer.as_ref().and_then(|e| e.get("goal")).and_then(|g| {
            let a = g.as_array()?;
            Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?])
        });
        for (uuid, status) in goal_updates {
            match status {
                1 | 2 => {
                    if !self.goal_start.contains_key(&uuid) {
                        self.goal_start.insert(uuid, (t, pos, goal_xy));
                        self.goal_counts[0] += 1;
                    }
                }
                4 | 5 | 6 => {
                    let (t0, p0, goal) = self.goal_start.remove(&uuid).unwrap_or((t, pos, goal_xy));
                    let idx = match status {
                        4 => 1,
                        5 => 3,
                        _ => 2,
                    };
                    self.goal_counts[idx] += 1;
                    let moved = (pos[0] - p0[0]).hypot(pos[1] - p0[1]);
                    let off = goal.map(|g| (pos[0] - g[0]).hypot(pos[1] - g[1]));
                    let name = ["", "succeeded", "aborted", "canceled"][idx];
                    let detail = format!(
                        "goal {name} after {:.0} s, moved {moved:.2} m{}",
                        t - t0,
                        off.map_or(String::new(), |d| format!(", {d:.2} m from the goal"))
                    );
                    if status == 4 && off.is_some_and(|d| d > 0.5) {
                        self.false_successes.push(json!({"t": round(t, 1), "x": round(pos[0], 2), "y": round(pos[1], 2),
                                                         "moved_m": round(moved, 2), "goal_off_m": round(off.unwrap(), 2)}));
                        self.event(t, "false_success", pos, detail.clone());
                    }
                    self.event(t, "nav_goal", pos, detail);
                }
                _ => {}
            }
        }
        for b in behavior_updates {
            self.metrics.count(&format!("nav2_{b}"));
            self.event(t, "recovery", pos, format!("Nav2 {b}"));
        }
        self.metrics.counts.insert("collision_monitor_stops".into(), cm_stops);
        for (name, level, msg) in logs {
            let interesting = match name.as_str() {
                "roomba_driver" => level >= 30 || msg.contains("bump") || msg.contains("OI mode"),
                "explorer" => true,
                _ => level >= 30,
            };
            if name == "roomba_driver" && msg.starts_with("bumper hit") {
                self.metrics.count("driver_bump_stops");
                if msg.contains("backing up") {
                    self.metrics.count("driver_bump_backups");
                }
            }
            if name == "controller_server" && msg.contains("Failed to make progress") {
                self.metrics.count("nav2_progress_failures");
            }
            if name == "planner_server" && level >= 30 && msg.to_lowercase().contains("plan") {
                self.metrics.count("nav2_plan_failures");
            }
            if name == "explorer" && msg.contains(": marked bump_") {
                self.metrics.count("explorer_bump_marks");
            }
            if name == "explorer" && msg.contains("with a person in front") {
                self.metrics.count("explorer_person_bumps");
            }
            if name == "explorer" && msg.contains("Boxed in by its own bump marks (") {
                self.metrics.count("explorer_trap_breaks");
            }
            if name == "explorer" && msg.contains("skipping it") {
                self.metrics.count("explorer_goals_skipped");
            }
            if interesting {
                self.event(t, "log", pos, format!("[{name}] {msg}"));
            }
        }
    }

    /// Starts the explorer, plays the operator, decides when the run is over (1 Hz).
    fn supervise(&mut self, t: f64) {
        let e = self.scenario.explorer.clone();
        let mode = lock(&self.shared).robot_mode.clone();
        if !self.explorer_started && e.autostart_s >= 0.0 && t >= e.autostart_s && t >= self.next_start_try {
            if driving_state(&self.explorer_state) {
                self.explorer_started = true;
            } else if self.start_attempts >= e.max_start_attempts {
                self.finish(t, "explorer_failed_to_start", &format!("explorer never started: {}", self.explorer_message));
            } else if mode.as_deref() == Some("safe") {
                self.start_attempts += 1;
                self.explorer_command("start", t);
                self.next_start_try = t + 5.0;
            } else if t >= e.autostart_s + 120.0 {
                // The driver's latched mode never reached the twin (seen once in ~100 parallel
                // WSL runs): say so, so the runner can run it again, instead of idling 30 min.
                let m = mode.as_deref().unwrap_or("none").to_string();
                self.finish(t, "explorer_failed_to_start", &format!("the robot never reported safe mode (mode: {m})"));
            }
        }
        if !self.explorer_started && driving_state(&self.explorer_state) {
            self.explorer_started = true;
        }
        // The operator: the explorer stopped because the Roomba left safe mode (cliff, wheel drop).
        let op = self.scenario.operator.clone();
        if self.explorer_started && self.explorer_state == "stopped" && op.enabled && self.finished.is_none() && !self.robot.fallen {
            if self.intervene_at.is_none() && self.restart_explorer_at.is_none() {
                if self.interventions < op.max_interventions {
                    self.intervene_at = Some(t + op.delay_s);
                } else if t - self.explorer_since > op.delay_s {
                    self.finish(t, "stopped", &format!("explorer stopped: {}", self.explorer_message));
                }
            }
        }
        if let Some(at) = self.intervene_at {
            if t >= at {
                self.intervene_at = None;
                self.interventions += 1;
                let pos = self.pos();
                let mut what = String::new();
                if self.robot.cliff.iter().any(|&c| c) {
                    // Carry it back from the edge, as a person would.
                    let th = self.robot.th;
                    let back = [self.robot.x - 0.3 * th.cos(), self.robot.y - 0.3 * th.sin(), th];
                    self.robot.teleport(back);
                    what.push_str("carried it back 0.3 m from the edge; ");
                }
                if mode.as_deref() != Some("safe") {
                    lock(&self.shared).set_mode_requests.push(oi::MODE_SAFE);
                    what.push_str("enabled driving (safe mode); ");
                }
                what.push_str("restarting the explorer");
                self.event(t, "operator", pos, what.clone());
                self.log(&format!("operator: {what}"));
                self.restart_explorer_at = Some(t + 3.0);
            }
        }
        if let Some(at) = self.restart_explorer_at {
            if t >= at {
                self.restart_explorer_at = None;
                self.explorer_command("start", t);
            }
        }
        if self.explorer_state == "done" && self.finished.is_none() && self.finish_at.is_none() {
            let msg = self.explorer_message.clone();
            self.finish_later(t, self.scenario.explorer.finish_grace_s, "explored", &msg);
        }
        if t >= self.scenario.duration_s && self.finished.is_none() {
            self.finish(t, "timeout", &format!("{:.0} s simulated; explorer: {} ({})", t, self.explorer_state, self.explorer_message));
        }
    }

    fn finish_later(&mut self, t: f64, after: f64, outcome: &str, detail: &str) {
        if self.finish_at.is_none() {
            self.finish_at = Some(t + after);
            self.finished_pending = Some((outcome.to_string(), detail.to_string()));
        }
    }

    fn finish(&mut self, _t: f64, outcome: &str, detail: &str) {
        if self.finished.is_none() {
            self.finished = Some((outcome.to_string(), detail.to_string()));
        }
    }

    fn sample(&mut self, t: f64, stamp: i64) {
        let (cov, battery, goal_active) = {
            let sh = lock(&self.shared);
            (sh.coverage, sh.battery_pct, sh.goal_active())
        };
        self.metrics.samples.push(Sample {
            t,
            pose: self.robot.pose(),
            coverage: cov.free,
            reachable: cov.reachable,
            battery,
            explorer: self.explorer_state.clone(),
            goal_active,
        });
        let c = &self.metrics.counts;
        let state = json!({
            "t": round(t, 1),
            "pose": [round(self.robot.x, 3), round(self.robot.y, 3), round(self.robot.th.to_degrees(), 1)],
            "coverage": round(cov.free, 4),
            "coverage_reachable": round(cov.reachable, 4),
            "mapped_m2": round(cov.mapped_m2, 2),
            "distance_m": round(self.robot.distance_total, 2),
            "oi_mode": oi::mode_str(Some(self.oi.mode as i64)),
            "bump": [self.robot.bump.0, self.robot.bump.1],
            "battery_soc": round(self.pack.soc, 3),
            "battery_v": round(self.pack.voltage(), 2),
            "explorer": self.explorer_state,
            "bumps": c.get("bumps").copied().unwrap_or(0),
            "interventions": self.interventions,
        });
        let _ = stamp;
        self.pubs.state.publish(std_msgs::msg::String { data: state.to_string() }).ok();
    }

    fn write_results(&mut self, t: f64, outcome: &str, detail: &str) -> Result<()> {
        std::fs::create_dir_all(&self.out_dir)?;
        let s = &self.scenario;
        let mapper = lock(&self.mapper);
        let cov = mapper.coverage();
        let (stuck, idle) = self.metrics.stuck_and_idle(s.metrics.stuck_window_s, s.metrics.stuck_radius_m);
        let people_paths: Vec<Vec<P2>> = s.people.iter().map(|p| p.path.clone()).collect();
        if let Some(png) = metrics::plot_png(&self.metrics, &mapper, &self.world, &people_paths, &stuck) {
            std::fs::write(format!("{}/trajectory.png", self.out_dir), png)?;
        }
        drop(mapper);
        let costmap = lock(&self.shared).costmap.clone();
        let frontiers = match costmap {
            Some(cm) => {
                let (groups, png) = metrics::frontier_report(&cm, self.pos(), &self.explorer_params);
                if let Some(png) = png {
                    std::fs::write(format!("{}/costmap_end.png", self.out_dir), png)?;
                }
                Value::from(groups)
            }
            None => Value::Null,
        };
        let episodes = |eps: &[(f64, f64, P2)]| -> Vec<Value> {
            eps.iter().map(|e| json!({"from_s": round(e.0, 0), "to_s": round(e.1, 0), "x": round(e.2[0], 2), "y": round(e.2[1], 2)})).collect()
        };
        std::fs::write(format!("{}/timeline.csv", self.out_dir), self.metrics.timeline_csv())?;
        std::fs::write(format!("{}/events.jsonl", self.out_dir), self.metrics.events_jsonl())?;
        let sh = lock(&self.shared);
        let explorer = self.explorer_state.clone();
        let ex = sh.explorer.clone().unwrap_or(Value::Null);
        let counts: serde_json::Map<String, Value> = self.metrics.counts.iter().map(|(k, v)| (k.clone(), json!(v))).collect();
        let result = json!({
            "scenario": s.name,
            "description": s.description,
            "seed": s.seed,
            "world": self.world_path,
            "outcome": outcome,
            "outcome_detail": detail,
            "sim_time_s": round(t, 1),
            "wall_time_s": round(self.wall_start.elapsed().as_secs_f64(), 1),
            "coverage": {
                "final": round(cov.free, 4),
                "reachable_final": round(cov.reachable, 4),
                "mapped_m2": round(cov.mapped_m2, 2),
                "true_free_m2": round(cov.truth_m2, 2),
                "t_50pct_s": self.metrics.time_to(0.5).map(|v| round(v, 0)),
                "t_80pct_s": self.metrics.time_to(0.8).map(|v| round(v, 0)),
                "t_90pct_s": self.metrics.time_to(0.9).map(|v| round(v, 0)),
                "every_30s": self.metrics.coverage_series(30.0),
            },
            "distance_m": round(self.robot.distance_total, 2),
            "turned_deg": round(self.robot.turned_total.to_degrees(), 0),
            "map_nodes": sh.map_nodes,
            "bumps": self.metrics.events_of("bump"),
            "contacts": self.metrics.events_of("contact"),
            "counts": counts,
            "stuck": {
                "window_s": s.metrics.stuck_window_s,
                "radius_m": s.metrics.stuck_radius_m,
                "count": stuck.len(),
                "total_s": round(stuck.iter().map(|e| e.1 - e.0).sum::<f64>(), 0),
                "episodes": episodes(&stuck),
                "idle_waits": episodes(&idle),
                "idle_total_s": round(idle.iter().map(|e| e.1 - e.0).sum::<f64>(), 0),
            },
            "frontiers_at_end": frontiers,
            "railing": {
                "min_clearance_m": self.metrics.railing_min_clearance.map(|c| round(c, 3)),
                "within_0_10m_s": round(self.metrics.railing_close_s, 1),
            },
            "nav2_goals": {
                "sent": self.goal_counts[0],
                "succeeded": self.goal_counts[1],
                "aborted": self.goal_counts[2],
                "canceled": self.goal_counts[3],
                "false_successes": self.false_successes,
            },
            "explorer": {
                "final_state": explorer,
                "final_message": self.explorer_message,
                "status": ex,
                "start_attempts": self.start_attempts,
            },
            "operator_interventions": self.interventions,
            "battery": {
                "start_pct": self.battery_first.map(|b| round(b * 100.0, 1)),
                "min_pct": self.battery_min.map(|b| round(b * 100.0, 1)),
                "end_pct": sh.battery_pct.map(|b| round(b * 100.0, 1)),
                "soc_end": round(self.pack.soc, 3),
            },
            "sensors": {
                "depth_frames": sh.depth_frames,
                "depth_frames_skipped": sh.depth_skipped,
                "render_ms_avg": round(sh.render_ms, 2),
                "oi_frames": self.oi.frames,
                "oi_commands": self.oi.commands,
            },
            "plot_legend": metrics::LEGEND.iter().map(|(name, bgr)| json!({"what": name, "rgb": [bgr[2], bgr[1], bgr[0]]})).collect::<Vec<_>>(),
        });
        drop(sh);
        std::fs::write(format!("{}/result.json", self.out_dir), serde_json::to_string_pretty(&result)? + "\n")?;
        Ok(())
    }
}

// ---------------------------------------------------------------- render thread

fn render_loop(
    rx: mpsc::Receiver<Snapshot>,
    grid: Arc<Grid>,
    scenario: Scenario,
    mapper: Arc<Mutex<Mapper>>,
    shared: Arc<Mutex<Shared>>,
    pubs: RenderPubs,
    mut rng: Rng,
) {
    let cam = Camera::new(D1000);
    let s = &scenario.sensors;
    let opts = RenderOpts {
        min_range: s.depth_min_m,
        max_range: s.depth_max_m,
        scale: s.depth_scale,
        noise_a: s.depth_noise_a,
        noise_b: s.depth_noise_b,
        dropout: s.depth_dropout,
    };
    let cliffs = scenario.cliffs.iter().map(|c| (c.name.clone(), roomba_rs::twin::world::Shape::Polygon(c.polygon.clone()))).collect();
    let cliff_drops = scenario.cliffs.iter().map(|c| c.drop_m).collect();
    let mut world = World { grid: Arc::clone(&grid), bodies: Vec::new(), cliffs, cliff_drops };
    let map_period = 1.0 / scenario.mapping.update_hz.max(0.01);
    let mut last_map = f64::NEG_INFINITY;
    let mut preview = Vec::new();
    let mut avg_ms = 0.0;
    let publish_map = |m: &Mapper, stamp: i64| {
        pubs.map.publish(grid_msg(stamp, &grid, m.occupancy())).ok();
        let poses = m
            .nodes
            .iter()
            .map(|n| {
                let mut p = geometry_msgs::msg::PoseStamped::default();
                p.header = header(stamp, "map");
                p.pose.position.x = n[0];
                p.pose.position.y = n[1];
                p.pose.orientation = yaw_quat(n[2]);
                p
            })
            .collect();
        pubs.path.publish(nav_msgs::msg::Path { header: header(stamp, "map"), poses }).ok();
    };
    {
        let m = lock(&mapper);
        if m.version > 0 {
            publish_map(&m, 0);
        }
    }
    for snap in rx {
        world.bodies = snap.bodies;
        let started = Instant::now();
        let depth = cam.render(&world, snap.lens_true, &opts, &mut rng, snap.preview.then_some(&mut preview));
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        avg_ms = if avg_ms == 0.0 { ms } else { 0.95 * avg_ms + 0.05 * ms };
        let (w, h) = (cam.k.width, cam.k.height);
        let mut data = Vec::with_capacity(w * h * 2);
        for d in &depth {
            data.extend_from_slice(&d.to_le_bytes());
        }
        let img = sensor_msgs::msg::Image {
            header: header(snap.stamp, "camera_left_optical_frame"),
            height: h as u32,
            width: w as u32,
            encoding: "16UC1".into(),
            is_bigendian: 0,
            step: (w * 2) as u32,
            data,
        };
        pubs.depth.publish(img).ok();
        pubs.depth_info.publish(camera_info(snap.stamp)).ok();
        if snap.preview {
            let img = sensor_msgs::msg::Image {
                header: header(snap.stamp, "camera_left_optical_frame"),
                height: h as u32,
                width: w as u32,
                encoding: "mono8".into(),
                is_bigendian: 0,
                step: w as u32,
                data: preview.clone(),
            };
            pubs.left.publish(img).ok();
            pubs.left_info.publish(camera_info(snap.stamp)).ok();
        }
        if snap.t - last_map >= map_period && snap.tracking_ok {
            last_map = snap.t;
            let mut m = lock(&mapper);
            if m.wants_node(snap.base_est) {
                m.add_node(&cam, &depth, snap.lens_est, snap.base_est);
                publish_map(&m, snap.stamp);
                let cov = m.coverage();
                let nodes = m.nodes.len();
                drop(m);
                let mut sh = lock(&shared);
                sh.coverage = cov;
                sh.map_nodes = nodes;
            }
        }
        let mut sh = lock(&shared);
        sh.depth_frames += 1;
        sh.render_ms = avg_ms;
    }
}

// ---------------------------------------------------------------- main

struct Params {
    world: String,
    scenario: String,
    out_dir: String,
    pty_link: String,
    mount: [f64; 3],
    cam_pitch: f64,
    seed: i64,
    exit_on_finish: bool,
    explorer_params: String,
}

fn main() -> Result<()> {
    roomba_rs::default_signals();
    let signals = block_signals();
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();
    let node = executor.create_node("twin2d")?;
    let mut d = NodeParams::new(&node);
    let p = Params {
        world: d.string("world", "/data/twin/twin2d/world/house.yaml")?,
        scenario: d.string("scenario", "")?,
        out_dir: d.string("out_dir", "/data/twin/twin2d/runs/latest")?,
        pty_link: d.string("pty_link", "")?,
        mount: [d.float("cam_x", 0.153)?, d.float("cam_y", 0.0)?, d.float("cam_z", 0.13)?],
        cam_pitch: d.float("cam_pitch", 0.0)?,
        seed: d.int("seed", -1)?,
        exit_on_finish: d.boolean("exit_on_finish", true)?,
        explorer_params: d.string("explorer_params", "")?,
    };
    if p.cam_pitch.abs() > 1e-6 {
        log!(node.warn(), "cam_pitch {} ignored: the twin's camera renders level", p.cam_pitch);
    }
    // Explorer parameter overrides of an experiment: the frontier report judges with the same values.
    let explorer_params: serde_json::Map<String, Value> = if p.explorer_params.trim().is_empty() {
        serde_json::Map::new()
    } else {
        let v: serde_yaml::Value = serde_yaml::from_str(&p.explorer_params).context("explorer_params")?;
        match serde_json::to_value(v)? {
            Value::Object(m) => m,
            _ => bail!("explorer_params must be a mapping"),
        }
    };
    let mut scenario = if p.scenario.is_empty() { Scenario::parse("")? } else { Scenario::load(&p.scenario)? };
    if p.seed >= 0 {
        scenario.seed = p.seed as u64;
    }
    std::fs::create_dir_all(&p.out_dir).with_context(|| format!("cannot create {}", p.out_dir))?;
    if !p.scenario.is_empty() {
        let _ = std::fs::copy(&p.scenario, format!("{}/scenario.yaml", p.out_dir));
    }
    let grid = Grid::load(&p.world)?;
    if let Some(path) = roomba_rs::twin::world::overlay_path(&p.world)? {
        let overlay = WorldOverlay::load(&path)?;
        scenario.merge_world(&overlay)?;
        log!(
            node.info(),
            "world overlay {path} ('{}'): {} doors, {} railings, {} obstacles, {} cliffs",
            overlay.name,
            overlay.doors.len(),
            overlay.railings.len(),
            overlay.obstacles.len(),
            overlay.cliffs.len()
        );
    }
    let mut world = World::new(grid);
    let (door_ids, people_ids) = scenario.populate(&mut world)?;
    let start = scenario.start_pose();
    {
        let probe = Robot::new(&scenario.robot, start);
        let c = world.nearest([start[0], start[1]], scenario.robot.radius_m, scenario.robot.climb_m, scenario.robot.height_m);
        if let Some(c) = c {
            if c.dist < probe.cfg.radius_m - 0.01 {
                bail!("start pose ({:.2}, {:.2}) overlaps {} (clearance {:.3} m)", start[0], start[1], world.what_name(c.what), c.dist - probe.cfg.radius_m);
            }
        }
    }
    let pty_link = if p.pty_link.is_empty() { format!("{}/ttyROOMBA", p.out_dir) } else { p.pty_link.clone() };
    if pty_link.starts_with("/dev/") {
        bail!("pty_link must not be under /dev (got {pty_link})");
    }
    let pty = Pty::open(&pty_link)?;
    log!(node.info(), "Roomba Open Interface on {} (linked at {pty_link})", pty.name);

    let latched = || QoSProfile::topics_default().keep_last(1).reliable().transient_local();
    let pubs = Pubs {
        tf: node.create_publisher("/tf".qos(QoSProfile::topics_default().keep_last(100).reliable()))?,
        odom: node.create_publisher("/odom".keep_last(20))?,
        truth: node.create_publisher("/twin/ground_truth".keep_last(10))?,
        imu: node.create_publisher("/camera/imu/data_raw".keep_last(10))?,
        filter_status: node.create_publisher("/vslam_filter/status".keep_last(2))?,
        vslam_status: node.create_publisher("/visual_slam/status".keep_last(10))?,
        state: node.create_publisher("/twin/state".keep_last(10))?,
        events: node.create_publisher("/twin/events".keep_last(100))?,
        explorer_cmd: node.create_publisher("/explorer/command".keep_last(10))?,
        world: node.create_publisher("/twin/world".qos(latched()))?,
        persons: node.create_publisher("/perception/person_detector/detections_3d".keep_last(5))?,
    };
    let render_pubs = RenderPubs {
        depth: node.create_publisher("/camera/depth/image_rect_raw".keep_last(10))?,
        depth_info: node.create_publisher("/camera/depth/camera_info".keep_last(10))?,
        left: node.create_publisher("/camera/left/image_rect_mono".keep_last(10))?,
        left_info: node.create_publisher("/camera/left/camera_info".keep_last(10))?,
        map: node.create_publisher("/rtabmap/map".qos(latched()))?,
        path: node.create_publisher("/rtabmap/mapPath".keep_last(1))?,
    };
    // The camera driver's static frames (base_link -> camera_link comes from the launch's mount).
    let tf_static: Publisher<tf2_msgs::msg::TFMessage> = node.create_publisher("/tf_static".qos(latched()))?;
    let now = node.get_clock().now().nsec;
    tf_static
        .publish(tf2_msgs::msg::TFMessage {
            transforms: vec![
                transform(now, "camera_link", "camera_left_optical_frame", [0.0, depth::LEFT_OPTICAL_Y, 0.0], quat(depth::Q_LINK_OPTICAL)),
                transform(now, "camera_link", "camera_right_optical_frame", [0.0, -depth::LEFT_OPTICAL_Y, 0.0], quat(depth::Q_LINK_OPTICAL)),
                transform(now, "camera_left_optical_frame", "camera_imu_frame", depth::IMU_IN_OPTICAL, quat(depth::Q_OPTICAL_IMU)),
            ],
        })
        .ok();

    let shared = Arc::new(Mutex::new(Shared::default()));
    let mut mapper = Mapper::new(&scenario.mapping, &world.grid, [start[0], start[1]], scenario.robot.radius_m);
    if scenario.initial_map == "known" {
        mapper.known_from_world(&world.grid);
        lock(&shared).coverage = mapper.coverage();
    }
    let mapper = Arc::new(Mutex::new(mapper));

    // ROS inputs for the metrics and the operator.
    let mut subs: Vec<Box<dyn std::any::Any>> = Vec::new();
    {
        let sh = Arc::clone(&shared);
        subs.push(Box::new(node.create_subscription("/explorer/status".qos(latched()), move |m: std_msgs::msg::String| {
            if let Ok(v) = serde_json::from_str::<Value>(&m.data) {
                lock(&sh).explorer = Some(v);
            }
        })?));
        let sh = Arc::clone(&shared);
        subs.push(Box::new(node.create_subscription("/roomba/mode".qos(latched()), move |m: std_msgs::msg::String| {
            lock(&sh).robot_mode = Some(m.data);
        })?));
        let sh = Arc::clone(&shared);
        subs.push(Box::new(node.create_subscription("/battery_state".keep_last(2), move |m: sensor_msgs::msg::BatteryState| {
            if !m.percentage.is_nan() {
                lock(&sh).battery_pct = Some(m.percentage as f64);
            }
        })?));
        let sh = Arc::clone(&shared);
        subs.push(Box::new(node.create_subscription(
            "/navigate_to_pose/_action/status".keep_last(10),
            move |m: action_msgs::msg::GoalStatusArray| {
                let mut s = lock(&sh);
                for g in &m.status_list {
                    let id = g.goal_info.goal_id.uuid;
                    if s.goals.get(&id) != Some(&g.status) {
                        s.goals.insert(id, g.status);
                        s.goal_updates.push((id, g.status));
                    }
                }
            },
        )?));
        for b in ["spin", "backup", "wait"] {
            let sh = Arc::clone(&shared);
            subs.push(Box::new(node.create_subscription(
                format!("/{b}/_action/status").as_str().keep_last(10),
                move |m: action_msgs::msg::GoalStatusArray| {
                    let mut s = lock(&sh);
                    for g in &m.status_list {
                        let key = (b.to_string(), g.goal_info.goal_id.uuid);
                        if !s.behaviors.contains_key(&key) {
                            s.behavior_updates.push(b.to_string());
                        }
                        s.behaviors.insert(key, g.status);
                    }
                },
            )?));
        }
        let sh = Arc::clone(&shared);
        subs.push(Box::new(node.create_subscription(
            "/collision_monitor_state".keep_last(10),
            move |m: nav2_msgs::msg::CollisionMonitorState| {
                let mut s = lock(&sh);
                if m.action_type == 1 && s.cm_state != 1 {
                    s.cm_stops += 1;
                }
                s.cm_state = m.action_type;
            },
        )?));
        let sh = Arc::clone(&shared);
        subs.push(Box::new(node.create_subscription("/global_costmap/costmap".qos(latched()), move |m: nav_msgs::msg::OccupancyGrid| {
            lock(&sh).costmap = Some(CostGrid {
                width: m.info.width as usize,
                height: m.info.height as usize,
                res: m.info.resolution as f64,
                origin: [m.info.origin.position.x, m.info.origin.position.y],
                data: m.data,
            });
        })?));
        let sh = Arc::clone(&shared);
        subs.push(Box::new(node.create_subscription("/rosout".keep_last(200), move |m: rcl_interfaces::msg::Log| {
            if matches!(m.name.as_str(), "roomba_driver" | "explorer" | "controller_server" | "planner_server" | "bt_navigator" | "behavior_server" | "collision_monitor") {
                lock(&sh).logs.push((m.name, m.level, m.msg));
            }
        })?));
    }
    let set_mode = node.create_client::<roomba_msgs::srv::SetMode>("/roomba/set_mode")?;
    let mut timers: Vec<Box<dyn std::any::Any>> = Vec::new();
    {
        let sh = Arc::clone(&shared);
        timers.push(Box::new(node.create_timer_repeating(Duration::from_millis(100).node_time(), move || {
            let requests = std::mem::take(&mut lock(&sh).set_mode_requests);
            for mode in requests {
                let sh2 = Arc::clone(&sh);
                let sent = set_mode.call_then(roomba_msgs::srv::SetMode_Request { mode }, move |r: roomba_msgs::srv::SetMode_Response| {
                    lock(&sh2).set_mode_results.push(format!("success {} mode {} {}", r.success, r.mode, r.message));
                });
                if let Err(e) = sent {
                    lock(&sh).set_mode_results.push(format!("call failed: {e}"));
                }
            }
        })?));
    }

    let (render_tx, render_rx) = mpsc::sync_channel::<Snapshot>(1);
    let rng = Rng::new(scenario.seed);
    {
        let (grid, sc, mp, sh, rng) = (Arc::clone(&world.grid), scenario.clone(), Arc::clone(&mapper), Arc::clone(&shared), rng.fork(2));
        std::thread::Builder::new().name("render".into()).spawn(move || render_loop(render_rx, grid, sc, mp, sh, render_pubs, rng))?;
    }

    let n_events = scenario.events.len();
    let robot = Robot::new(&scenario.robot, start);
    let pack = Pack::new(&scenario.battery);
    let wheel_base = scenario.robot.wheel_base_m;
    let mut sim = Sim {
        node: node.clone(),
        world,
        door_ids,
        people_ids,
        robot,
        pack,
        oi: OiSim::new(wheel_base),
        pty,
        loc: Localization::new(start),
        rng: rng.fork(1),
        person_rng: rng.fork(3),
        mount: p.mount,
        q_bi: depth::q_base_imu(0.0),
        pubs,
        shared: Arc::clone(&shared),
        mapper: Arc::clone(&mapper),
        render_tx,
        metrics: Metrics::default(),
        out_dir: p.out_dir.clone(),
        world_path: p.world.clone(),
        explorer_params: explorer_params.clone(),
        clock: node.get_clock(),
        event_done: vec![false; n_events],
        tracking_lost_until: f64::NEG_INFINITY,
        explorer_started: false,
        start_attempts: 0,
        next_start_try: 0.0,
        explorer_state: String::new(),
        explorer_message: String::new(),
        explorer_since: 0.0,
        interventions: 0,
        intervene_at: None,
        restart_explorer_at: None,
        finished: None,
        finished_pending: None,
        finish_at: None,
        last_explorer_log: None,
        goal_start: HashMap::new(),
        goal_counts: [0; 4],
        false_successes: Vec::new(),
        battery_first: None,
        battery_min: None,
        last_bump: (false, false),
        forced_bump: (false, false),
        forced_bump_until: f64::NEG_INFINITY,
        world_dirty: true,
        wall_start: Instant::now(),
        scenario,
    };
    log!(
        node.info(),
        "twin2d: scenario '{}' (seed {}), world {}, start ({:.2}, {:.2}, {:.0} deg), results in {}",
        sim.scenario.name,
        sim.scenario.seed,
        p.world,
        start[0],
        start[1],
        start[2].to_degrees(),
        p.out_dir
    );

    let stop = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    {
        let (stop, done) = (Arc::clone(&stop), Arc::clone(&done));
        on_signal(signals, move || {
            stop.store(true, Ordering::SeqCst);
            for _ in 0..100 {
                if done.load(Ordering::SeqCst) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        });
    }
    let finished = Arc::new(AtomicBool::new(false));
    let exit_on_finish = p.exit_on_finish;
    {
        let (stop, done, finished) = (Arc::clone(&stop), Arc::clone(&done), Arc::clone(&finished));
        let commands = Arc::clone(executor.commands());
        std::thread::Builder::new().name("sim".into()).spawn(move || {
            sim_loop(&mut sim, &stop);
            done.store(true, Ordering::SeqCst);
            if exit_on_finish || stop.load(Ordering::SeqCst) {
                finished.store(true, Ordering::SeqCst);
                commands.halt_spinning();
            }
        })?;
    }
    // Keeps spinning through the spurious timeouts a stepped-back clock causes (node_params::spin).
    let _ = roomba_rs::node_params::spin_until(&mut executor, || finished.load(Ordering::SeqCst));
    // The results are written. Exit without tearing down the render thread's publishers one by one
    // (rmw complains about each once the context is gone).
    std::process::exit(0);
}

fn sim_loop(sim: &mut Sim, stop: &AtomicBool) {
    let s = sim.scenario.clone();
    let depth_period = 1.0 / s.sensors.depth_rate_hz.max(0.1);
    let preview_period = if s.sensors.preview_hz > 0.0 { 1.0 / s.sensors.preview_hz } else { f64::INFINITY };
    let person_period = if s.sensors.person_rate_hz > 0.0 && !s.people.is_empty() { 1.0 / s.sensors.person_rate_hz } else { f64::INFINITY };
    let mut next_person = 0.0;
    let imu_every = (1.0 / s.sensors.imu_rate_hz.max(1.0) / DT).round().max(1.0) as u64;
    let sample_every = (s.metrics.sample_period_s / DT).round().max(1.0) as u64;
    let (mut next_depth, mut next_preview) = (0.0, 0.0);
    let start = Instant::now();
    let mut base = start;
    let mut k: u64 = 0;
    let mut behind_log = 0.0;
    loop {
        k += 1;
        let t = k as f64 * DT;
        let due = base + Duration::from_secs_f64(t);
        let now = Instant::now();
        if due > now {
            std::thread::sleep(due - now);
        } else if now - due > Duration::from_millis(250) {
            // Way behind (machine overloaded): don't race to catch up.
            base += now - due;
            if t - behind_log > 10.0 {
                behind_log = t;
                log!(sim.node.warn(), "simulation fell {:.0} ms behind real time", (now - due).as_secs_f64() * 1000.0);
            }
        }
        let stamp = sim.clock.now().nsec;
        let mono_now = mono();

        // The driver's bytes.
        let data = sim.pty.read();
        if !data.is_empty() {
            let raw = sim.raw_packets();
            let unsafe_now = sim.unsafe_now();
            let reply = sim.oi.on_bytes(&data, mono_now, unsafe_now, &|pid: u8| raw[pid as usize]);
            if !reply.is_empty() {
                sim.pty.write(&reply);
            }
        }
        for line in std::mem::take(&mut sim.oi.log) {
            log!(sim.node.info(), "OI: {line}");
            let pos = sim.pos();
            sim.event(t, "oi", pos, line);
        }

        sim.timeline(t);
        sim.physics(t);
        let truth = sim.robot.pose();
        sim.loc.update(truth, t, &sim.scenario);

        if k % imu_every == 0 {
            sim.publish_imu(stamp);
        }
        if sim.oi.frame_due(mono_now) {
            sim.robot.sense_ir(&sim.world, &mut sim.rng);
            let raw = sim.raw_packets();
            if let Some(frame) = sim.oi.frame(&|pid: u8| raw[pid as usize]) {
                sim.pty.write(&frame);
            }
        }
        if k % 4 == 0 {
            sim.publish_pose(stamp, k % 8 == 0, k % 20 == 0);
        }
        if t >= next_depth {
            next_depth += depth_period;
            if next_depth < t {
                next_depth = t + depth_period;
            }
            let preview = t >= next_preview;
            if preview {
                next_preview = t + preview_period;
            }
            let est = sim.loc.map_base();
            let snap = Snapshot {
                stamp,
                t,
                lens_true: depth::lens_pose(truth, sim.mount),
                lens_est: depth::lens_pose(est, sim.mount),
                base_est: est,
                bodies: sim.world.bodies.clone(),
                tracking_ok: sim.tracking_ok(t),
                preview,
            };
            if sim.render_tx.try_send(snap).is_err() {
                lock(&sim.shared).depth_skipped += 1;
            }
        }
        if k % 40 == 0 {
            sim.publish_status(stamp, t);
        }
        if t >= next_person {
            next_person = t + person_period;
            sim.publish_persons(stamp);
        }
        if k % 200 == 0 {
            sim.absorb(t);
            sim.supervise(t);
            if sim.world_dirty {
                sim.world_dirty = false;
                let data = world_occupancy(&sim.world);
                sim.pubs.world.publish(grid_msg(stamp, &sim.world.grid, data)).ok();
            }
        }
        if k % sample_every == 0 {
            sim.sample(t, stamp);
        }
        if let Some(at) = sim.finish_at {
            if t >= at && sim.finished.is_none() {
                sim.finished = sim.finished_pending.take();
            }
        }
        let interrupted = stop.load(Ordering::SeqCst);
        if interrupted && sim.finished.is_none() {
            sim.finished = Some(("interrupted".into(), "SIGINT/SIGTERM before the scenario ended".into()));
        }
        if let Some((outcome, detail)) = sim.finished.clone() {
            sim.absorb(t);
            sim.sample(t, stamp);
            log!(sim.node.info(), "run over after {t:.0} s: {outcome} ({detail}); writing results to {}", sim.out_dir);
            if let Err(e) = sim.write_results(t, &outcome, &detail) {
                log!(sim.node.error(), "could not write the results: {e:#}");
            }
            let data = json!({"finished": true, "outcome": outcome, "out_dir": sim.out_dir}).to_string();
            sim.pubs.events.publish(std_msgs::msg::String { data }).ok();
            return;
        }
    }
}
