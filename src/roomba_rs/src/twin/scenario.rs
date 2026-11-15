//! Scenario files (YAML): where the robot starts, what happens in the house and when, and the
//! twin's robot, sensor and mapping settings. Every field has a default; an empty file is a plain
//! exploration from the start pose. Unknown keys are errors (typos would silently do nothing).
//! See README.md ("Digital twin (2D)") and src/roomba_bringup/twin2d/scenarios/*.yaml.

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use super::world::{Body, DepthVis, Kind, Shape, World, P2};

#[derive(Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct Scenario {
    pub name: String,
    pub description: String,
    pub seed: u64,
    /// Simulated seconds before the run ends (if the explorer hasn't finished earlier).
    pub duration_s: f64,
    pub start: Pose,
    /// Put a dock (a 0.32 x 0.15 m box, 0.1 m high) right behind the start pose.
    pub dock: bool,
    /// "unknown" (exploring from scratch, the normal case) or "known" (RTAB-Map starts with the
    /// whole house mapped, as with a saved map).
    pub initial_map: String,
    pub explorer: ExplorerCfg,
    pub operator: OperatorCfg,
    pub robot: RobotCfg,
    pub battery: BatteryCfg,
    pub sensors: SensorCfg,
    pub odometry: OdometryCfg,
    pub mapping: MappingCfg,
    pub metrics: MetricsCfg,
    pub doors: Vec<DoorCfg>,
    pub railings: Vec<RailingCfg>,
    pub obstacles: Vec<ObstacleCfg>,
    pub people: Vec<PersonCfg>,
    pub cliffs: Vec<CliffCfg>,
    pub events: Vec<EventCfg>,
}

impl Default for Scenario {
    fn default() -> Self {
        Scenario {
            name: "unnamed".into(),
            description: String::new(),
            seed: 1,
            duration_s: 1200.0,
            start: Pose::default(),
            dock: false,
            initial_map: "unknown".into(),
            explorer: ExplorerCfg::default(),
            operator: OperatorCfg::default(),
            robot: RobotCfg::default(),
            battery: BatteryCfg::default(),
            sensors: SensorCfg::default(),
            odometry: OdometryCfg::default(),
            mapping: MappingCfg::default(),
            metrics: MetricsCfg::default(),
            doors: Vec::new(),
            railings: Vec::new(),
            obstacles: Vec::new(),
            people: Vec::new(),
            cliffs: Vec::new(),
            events: Vec::new(),
        }
    }
}

#[derive(Deserialize, Clone, Copy, Debug, Default, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Pose {
    pub x: f64,
    pub y: f64,
    /// Degrees, counter-clockwise from the map's +x axis.
    pub yaw_deg: f64,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct ExplorerCfg {
    /// Send "start" to the explorer this many seconds after the twin starts (and retry until it
    /// explores: Nav2 takes a while to come up). Negative: never (drive it yourself, web app).
    pub autostart_s: f64,
    /// End the run this long after the explorer reports done.
    pub finish_grace_s: f64,
    /// Give up starting the explorer after this many attempts.
    pub max_start_attempts: u32,
}

impl Default for ExplorerCfg {
    fn default() -> Self {
        ExplorerCfg { autostart_s: 15.0, finish_grace_s: 3.0, max_start_attempts: 30 }
    }
}

/// What a person watching the robot does when it stops (cliff, wheel drop, explorer stopped).
#[derive(Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct OperatorCfg {
    pub enabled: bool,
    /// Seconds before the operator re-enables driving (safe mode) and restarts the explorer.
    pub delay_s: f64,
    pub max_interventions: u32,
}

impl Default for OperatorCfg {
    fn default() -> Self {
        OperatorCfg { enabled: true, delay_s: 10.0, max_interventions: 3 }
    }
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct RobotCfg {
    pub radius_m: f64,
    /// Same values as the driver config (roomba_driver_robot.yaml).
    pub wheel_base_m: f64,
    pub wheel_diameter_m: f64,
    pub encoder_counts_per_rev: f64,
    /// Wheel acceleration limit (m/s^2).
    pub max_accel_mps2: f64,
    /// Fraction of the commanded turn rate the body really turns (wheel scrub; carpet ~0.5).
    pub turn_efficiency: f64,
    /// Top of the robot including the camera (m): obstacles above this are driven under.
    pub height_m: f64,
    /// Obstacles up to this height are driven over.
    pub climb_m: f64,
    /// Light-bumper gains (left, front left, center left, center right, front right, right): this
    /// unit's front-left reads ~235 with an object 1-2 cm away, the others stay under ~140.
    pub light_gain: [f64; 6],
    /// Report the right bumper switch as always pressed, like this unit (the driver ignores it).
    pub stuck_bump_right: bool,
    /// Fixed extra on the light-bumper signals while driving (thick carpet pile ~150).
    pub light_floor_signal: f64,
}

impl Default for RobotCfg {
    fn default() -> Self {
        RobotCfg {
            radius_m: 0.17,
            wheel_base_m: 0.235,
            wheel_diameter_m: 0.072,
            encoder_counts_per_rev: 508.8,
            max_accel_mps2: 1.5,
            turn_efficiency: 1.0,
            height_m: 0.16,
            climb_m: 0.015,
            light_gain: [0.5, 1.0, 0.55, 0.55, 0.55, 0.5],
            stuck_bump_right: false,
            light_floor_signal: 30.0,
        }
    }
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct BatteryCfg {
    /// State of charge at the start (0-1).
    pub soc: f64,
    pub capacity_ah: f64,
    /// Li-ion cells in series (Ridgid 18 V pack: 5).
    pub cells: u32,
    pub resistance_ohm: f64,
    /// Current (A) standing still, extra while driving at 0.25 m/s, extra while pushing against something.
    pub idle_a: f64,
    pub drive_a: f64,
    pub stall_a: f64,
    /// Multiplies the drain (e.g. 20 to see a whole discharge in a short run).
    pub drain_scale: f64,
}

impl Default for BatteryCfg {
    fn default() -> Self {
        BatteryCfg {
            soc: 0.95,
            capacity_ah: 4.0,
            cells: 5,
            resistance_ohm: 0.1,
            idle_a: 0.22,
            drive_a: 0.4,
            stall_a: 0.6,
            drain_scale: 1.0,
        }
    }
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct SensorCfg {
    pub depth_rate_hz: f64,
    /// Depth noise sigma (m) = depth_noise_a + depth_noise_b * Z^2.
    pub depth_noise_a: f64,
    pub depth_noise_b: f64,
    /// Fraction of depth pixels dropped at random (stereo holes).
    pub depth_dropout: f64,
    /// Reported depth = true depth x this (the real camera reads ~15 % short: 0.85).
    pub depth_scale: f64,
    pub depth_min_m: f64,
    pub depth_max_m: f64,
    pub imu_rate_hz: f64,
    pub gyro_noise: f64,
    pub accel_noise: f64,
    /// rad/s, IMU frame (measured on this camera).
    pub gyro_bias: [f64; 3],
    /// Synthetic grayscale view on /camera/left/image_rect_mono for the web app (0 = off).
    pub preview_hz: f64,
}

impl Default for SensorCfg {
    fn default() -> Self {
        SensorCfg {
            depth_rate_hz: 15.0,
            depth_noise_a: 0.002,
            depth_noise_b: 0.003,
            depth_dropout: 0.0,
            depth_scale: 1.0,
            depth_min_m: 0.3,
            depth_max_m: 7.0,
            imu_rate_hz: 200.0,
            gyro_noise: 0.0012,
            accel_noise: 0.017,
            gyro_bias: [-0.0013, 0.0007, 0.0006],
            preview_hz: 2.0,
        }
    }
}

/// Error of the odometry (odom -> base_link) and how often RTAB-Map corrects it (map -> odom).
#[derive(Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct OdometryCfg {
    /// Relative distance error (0.02 = 2 % long).
    pub scale_error: f64,
    /// Heading error per metre driven (deg/m).
    pub yaw_drift_deg_per_m: f64,
    /// Heading drift while standing or turning (deg/s).
    pub yaw_bias_deg_s: f64,
    /// map -> odom is snapped to the truth this often (s); 0 = never (pure odometry drift).
    pub correction_period_s: f64,
}

impl Default for OdometryCfg {
    fn default() -> Self {
        OdometryCfg { scale_error: 0.0, yaw_drift_deg_per_m: 0.0, yaw_bias_deg_s: 0.0, correction_period_s: 1.0 }
    }
}

/// RTAB-Map's 2D grid settings (vslam.launch.py on the robot).
#[derive(Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct MappingCfg {
    pub update_hz: f64,
    pub range_max_m: f64,
    pub linear_update_m: f64,
    pub angular_update_rad: f64,
    pub prob_hit: f64,
    pub prob_miss: f64,
    pub clamp_min: f64,
    pub clamp_max: f64,
    pub footprint_radius_m: f64,
    pub max_ground_height_m: f64,
    pub max_obstacle_height_m: f64,
    pub noise_radius_m: f64,
    pub noise_min_neighbors: usize,
    pub decimation: usize,
}

impl Default for MappingCfg {
    fn default() -> Self {
        MappingCfg {
            update_hz: 1.0,
            range_max_m: 2.5,
            linear_update_m: 0.1,
            angular_update_rad: 0.1,
            prob_hit: 0.7,
            prob_miss: 0.45,
            clamp_min: 0.1192,
            clamp_max: 0.971,
            footprint_radius_m: 0.2,
            max_ground_height_m: 0.06,
            max_obstacle_height_m: 1.0,
            noise_radius_m: 0.1,
            noise_min_neighbors: 5,
            decimation: 4,
        }
    }
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct MetricsCfg {
    /// Stuck: while the explorer is driving, the robot stays within stuck_radius_m for stuck_window_s.
    pub stuck_window_s: f64,
    pub stuck_radius_m: f64,
    /// Coverage samples for the timeline (s).
    pub sample_period_s: f64,
}

impl Default for MetricsCfg {
    fn default() -> Self {
        MetricsCfg { stuck_window_s: 30.0, stuck_radius_m: 0.25, sample_period_s: 1.0 }
    }
}

fn default_thickness() -> f64 {
    0.05
}

/// The area of a door or obstacle: exactly one of `polygon` ([[x, y], ...]), `circle` ([x, y, r])
/// or `segment` ([[x1, y1], [x2, y2]], `thickness` m wide).
fn area_shape(polygon: &Option<Vec<P2>>, circle: &Option<[f64; 3]>, segment: &Option<[P2; 2]>, thickness: f64) -> Result<Shape> {
    Ok(match (polygon, circle, segment) {
        (Some(p), None, None) => {
            if p.len() < 3 {
                bail!("a polygon needs at least 3 points");
            }
            Shape::Polygon(p.clone())
        }
        (None, Some(c), None) => Shape::Circle { c: [c[0], c[1]], r: c[2] },
        (None, None, Some(s)) => Shape::Capsule { a: s[0], b: s[1], r: thickness / 2.0 },
        _ => bail!("needs exactly one of polygon, circle or segment"),
    })
}

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct DoorCfg {
    pub name: String,
    pub polygon: Option<Vec<P2>>,
    pub circle: Option<[f64; 3]>,
    pub segment: Option<[P2; 2]>,
    #[serde(default = "default_thickness")]
    pub thickness: f64,
    #[serde(default)]
    pub closed: bool,
    #[serde(default)]
    pub events: Vec<DoorEvent>,
}

impl DoorCfg {
    pub fn shape(&self) -> Result<Shape> {
        area_shape(&self.polygon, &self.circle, &self.segment, self.thickness).with_context(|| format!("door {}", self.name))
    }
}

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct DoorEvent {
    pub t: f64,
    /// "open" or "closed"
    pub state: String,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct RailingCfg {
    pub name: String,
    /// Polyline along the railing.
    pub points: Vec<P2>,
    pub post_spacing_m: f64,
    pub post_radius_m: f64,
    pub height_m: f64,
    /// "none" (depth camera never sees it), "sparse" (posts seen now and then) or "solid".
    pub depth: String,
    /// Chance a post is seen in a depth frame ("sparse").
    pub sparse_prob: f64,
    /// Seen by the light bumpers.
    pub ir: bool,
}

impl Default for RailingCfg {
    fn default() -> Self {
        RailingCfg {
            name: "railing".into(),
            points: Vec::new(),
            post_spacing_m: 0.12,
            post_radius_m: 0.015,
            height_m: 0.9,
            depth: "none".into(),
            sparse_prob: 0.15,
            ir: true,
        }
    }
}

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct ObstacleCfg {
    pub name: String,
    pub polygon: Option<Vec<P2>>,
    pub circle: Option<[f64; 3]>,
    pub segment: Option<[P2; 2]>,
    #[serde(default = "default_thickness")]
    pub thickness: f64,
    #[serde(default)]
    pub zmin: f64,
    #[serde(default = "default_obstacle_top")]
    pub zmax: f64,
    /// "solid", "none" or "sparse"
    #[serde(default = "default_solid")]
    pub depth: String,
    #[serde(default = "default_true")]
    pub ir: bool,
    /// Appears / disappears at these times (s); default: always there.
    #[serde(default)]
    pub from_s: f64,
    #[serde(default)]
    pub until_s: Option<f64>,
}

impl ObstacleCfg {
    pub fn shape(&self) -> Result<Shape> {
        area_shape(&self.polygon, &self.circle, &self.segment, self.thickness).with_context(|| format!("obstacle {}", self.name))
    }
}

fn default_obstacle_top() -> f64 {
    0.5
}
fn default_solid() -> String {
    "solid".into()
}
fn default_true() -> bool {
    true
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct PersonCfg {
    pub name: String,
    pub radius_m: f64,
    pub height_m: f64,
    pub speed_mps: f64,
    pub path: Vec<P2>,
    /// "loop", "pingpong" or "once" (stays at the last point; leaves at until_s).
    pub mode: String,
    pub from_s: f64,
    pub until_s: Option<f64>,
    /// Pause at each path point (s).
    pub pause_s: f64,
}

impl Default for PersonCfg {
    fn default() -> Self {
        PersonCfg {
            name: "person".into(),
            radius_m: 0.22,
            height_m: 1.7,
            speed_mps: 0.7,
            path: Vec::new(),
            mode: "pingpong".into(),
            from_s: 0.0,
            until_s: None,
            pause_s: 1.0,
        }
    }
}

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct CliffCfg {
    pub name: String,
    pub polygon: Vec<P2>,
}

/// A timed event. Exactly one action per entry.
#[derive(Deserialize, Clone, Debug, Default)]
#[serde(default, deny_unknown_fields)]
pub struct EventCfg {
    pub t: f64,
    /// Set the battery's state of charge (0-1).
    pub battery_soc: Option<f64>,
    /// Visual tracking lost for this long (s): /vslam_filter/status "recovering", no map updates.
    pub tracking_lost_s: Option<f64>,
    /// Send this to /explorer/command ("start", "stop", "home").
    pub explorer: Option<String>,
    /// Move the robot (kidnap): {x, y, yaw_deg}.
    pub teleport: Option<Pose>,
    /// End the run.
    pub end: bool,
}

impl Scenario {
    pub fn load(path: &str) -> Result<Scenario> {
        let text = std::fs::read_to_string(path).with_context(|| format!("cannot read scenario {path}"))?;
        Scenario::parse(&text).with_context(|| format!("scenario {path}"))
    }

    pub fn parse(text: &str) -> Result<Scenario> {
        let s: Scenario = if text.trim().is_empty() { Scenario::default() } else { serde_yaml::from_str(text)? };
        s.check()?;
        Ok(s)
    }

    fn check(&self) -> Result<()> {
        if !matches!(self.initial_map.as_str(), "unknown" | "known") {
            bail!("initial_map must be unknown or known, not '{}'", self.initial_map);
        }
        for d in &self.doors {
            d.shape()?;
            for e in &d.events {
                if !matches!(e.state.as_str(), "open" | "closed") {
                    bail!("door {}: state must be open or closed, not '{}'", d.name, e.state);
                }
            }
        }
        for r in &self.railings {
            if r.points.len() < 2 {
                bail!("railing {}: needs at least 2 points", r.name);
            }
            parse_vis(&r.depth, r.sparse_prob)?;
        }
        for o in &self.obstacles {
            o.shape()?;
            parse_vis(&o.depth, 0.15)?;
            if o.zmax <= o.zmin {
                bail!("obstacle {}: zmax must be above zmin", o.name);
            }
        }
        for p in &self.people {
            if p.path.is_empty() {
                bail!("person {}: needs a path", p.name);
            }
            if !matches!(p.mode.as_str(), "loop" | "pingpong" | "once") {
                bail!("person {}: mode must be loop, pingpong or once", p.name);
            }
        }
        for c in &self.cliffs {
            if c.polygon.len() < 3 {
                bail!("cliff {}: polygon needs at least 3 points", c.name);
            }
        }
        for e in &self.events {
            let n = e.battery_soc.is_some() as u8
                + e.tracking_lost_s.is_some() as u8
                + e.explorer.is_some() as u8
                + e.teleport.is_some() as u8
                + e.end as u8;
            if n != 1 {
                bail!("event at t={}: needs exactly one of battery_soc, tracking_lost_s, explorer, teleport, end", e.t);
            }
        }
        Ok(())
    }

    /// The start pose (x, y, yaw rad).
    pub fn start_pose(&self) -> [f64; 3] {
        [self.start.x, self.start.y, self.start.yaw_deg.to_radians()]
    }

    /// Adds the scenario's doors, railings, obstacles, people (inactive until they appear), dock
    /// and cliffs to the world. Returns the body index of each door and person, in file order.
    pub fn populate(&self, world: &mut World) -> Result<(Vec<usize>, Vec<usize>)> {
        let mut doors = Vec::new();
        for d in &self.doors {
            doors.push(world.bodies.len());
            world.bodies.push(Body {
                name: d.name.clone(),
                kind: Kind::Door,
                shape: d.shape()?,
                zmin: 0.0,
                zmax: 2.0,
                depth: DepthVis::Solid,
                ir: true,
                blocks: true,
                active: d.closed,
            });
        }
        for r in &self.railings {
            let radius = r.post_radius_m.max(0.005);
            for (k, w) in r.points.windows(2).enumerate() {
                world.bodies.push(Body {
                    name: format!("{}#{k}", r.name),
                    kind: Kind::Railing,
                    shape: Shape::Capsule { a: w[0], b: w[1], r: radius },
                    zmin: 0.0,
                    zmax: r.height_m,
                    depth: if r.depth == "solid" { DepthVis::Solid } else { DepthVis::Invisible },
                    ir: r.ir,
                    blocks: true,
                    active: true,
                });
            }
            if r.depth == "sparse" {
                for (k, c) in posts(&r.points, r.post_spacing_m).into_iter().enumerate() {
                    world.bodies.push(Body {
                        name: format!("{}@{k}", r.name),
                        kind: Kind::Post,
                        shape: Shape::Circle { c, r: radius },
                        zmin: 0.0,
                        zmax: r.height_m,
                        depth: DepthVis::Sparse(r.sparse_prob),
                        ir: false,
                        blocks: false,
                        active: true,
                    });
                }
            }
        }
        for o in &self.obstacles {
            world.bodies.push(Body {
                name: o.name.clone(),
                kind: Kind::Obstacle,
                shape: o.shape()?,
                zmin: o.zmin,
                zmax: o.zmax,
                depth: parse_vis(&o.depth, 0.15)?,
                ir: o.ir,
                blocks: true,
                active: o.from_s <= 0.0,
            });
        }
        let mut people = Vec::new();
        for p in &self.people {
            people.push(world.bodies.len());
            world.bodies.push(Body {
                name: p.name.clone(),
                kind: Kind::Person,
                shape: Shape::Circle { c: p.path[0], r: p.radius_m },
                zmin: 0.0,
                zmax: p.height_m,
                depth: DepthVis::Solid,
                ir: true,
                blocks: true,
                active: false,
            });
        }
        if self.dock {
            let [x, y, yaw] = self.start_pose();
            // Behind the robot: from 0.18 to 0.33 m behind its centre, 0.32 m wide.
            let (c, s) = (yaw.cos(), yaw.sin());
            let corner = |bx: f64, by: f64| [x + c * bx - s * by, y + s * bx + c * by];
            world.bodies.push(Body {
                name: "dock".into(),
                kind: Kind::Dock,
                shape: Shape::Polygon(vec![corner(-0.18, -0.16), corner(-0.33, -0.16), corner(-0.33, 0.16), corner(-0.18, 0.16)]),
                zmin: 0.0,
                zmax: 0.1,
                depth: DepthVis::Solid,
                ir: true,
                blocks: true,
                active: true,
            });
        }
        for c in &self.cliffs {
            world.cliffs.push((c.name.clone(), Shape::Polygon(c.polygon.clone())));
        }
        Ok((doors, people))
    }
}

fn parse_vis(name: &str, p: f64) -> Result<DepthVis> {
    Ok(match name {
        "solid" => DepthVis::Solid,
        "none" => DepthVis::Invisible,
        "sparse" => DepthVis::Sparse(p),
        other => bail!("depth must be solid, none or sparse, not '{other}'"),
    })
}

/// Post positions every `spacing` metres along a polyline (both ends included).
pub fn posts(points: &[P2], spacing: f64) -> Vec<P2> {
    let mut out = vec![points[0]];
    let mut carry = 0.0;
    for w in points.windows(2) {
        let (a, b) = (w[0], w[1]);
        let len = (b[0] - a[0]).hypot(b[1] - a[1]);
        let mut s = spacing - carry;
        while s < len - 1e-9 {
            out.push([a[0] + (b[0] - a[0]) * s / len, a[1] + (b[1] - a[1]) * s / len]);
            s += spacing;
        }
        carry = len - (s - spacing);
        if carry >= spacing - 1e-9 {
            carry = 0.0;
        }
    }
    let last = *points.last().unwrap();
    if out.last().is_none_or(|p| (p[0] - last[0]).hypot(p[1] - last[1]) > 1e-6) {
        out.push(last);
    }
    out
}

/// Position of a person along their path at time t (s since they appeared).
pub fn person_position(p: &PersonCfg, t: f64) -> P2 {
    let path = &p.path;
    if path.len() == 1 || p.speed_mps <= 0.0 {
        return path[0];
    }
    let mut legs: Vec<(P2, P2)> = path.windows(2).map(|w| (w[0], w[1])).collect();
    if p.mode == "loop" {
        legs.push((*path.last().unwrap(), path[0]));
    } else if p.mode == "pingpong" {
        let back: Vec<(P2, P2)> = path.windows(2).rev().map(|w| (w[1], w[0])).collect();
        legs.extend(back);
    }
    let durations: Vec<f64> =
        legs.iter().map(|(a, b)| (b[0] - a[0]).hypot(b[1] - a[1]) / p.speed_mps + p.pause_s.max(0.0)).collect();
    let total: f64 = durations.iter().sum();
    if total <= 0.0 {
        return path[0];
    }
    let mut t = if p.mode == "once" { t.min(total - 1e-9) } else { t.rem_euclid(total) };
    for ((a, b), d) in legs.iter().zip(&durations) {
        if t < *d {
            let walk = d - p.pause_s.max(0.0);
            if t >= walk {
                return *b;
            }
            let f = if walk > 0.0 { t / walk } else { 1.0 };
            return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
        }
        t -= d;
    }
    *path.last().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::twin::world::Grid;

    #[test]
    fn empty_scenario_is_a_plain_exploration() {
        let s = Scenario::parse("").unwrap();
        assert_eq!(s.initial_map, "unknown");
        assert_eq!(s.robot.wheel_base_m, 0.235);
        assert!(s.explorer.autostart_s > 0.0 && s.operator.enabled);
    }

    #[test]
    fn full_scenario_parses_and_populates() {
        let text = r#"
name: test
seed: 3
start: {x: 1.0, y: 2.0, yaw_deg: 90}
dock: true
doors:
  - name: hall
    segment: [[0, 0], [1, 0]]
    events: [{t: 10, state: closed}, {t: 20, state: open}]
railings:
  - name: stair
    points: [[0, 1], [1.2, 1]]
    depth: sparse
obstacles:
  - name: box
    polygon: [[2, 2], [3, 2], [3, 3]]
    zmax: 0.03
    depth: none
people:
  - name: walker
    path: [[0, 0], [2, 0]]
    speed_mps: 1.0
    pause_s: 0
cliffs:
  - name: stairs
    polygon: [[5, 5], [6, 5], [6, 6], [5, 6]]
events:
  - {t: 30, battery_soc: 0.15}
  - {t: 40, tracking_lost_s: 5}
"#;
        let s = Scenario::parse(text).unwrap();
        let mut w = World::new(Grid::new(10, 10, 0.5, [0.0, 0.0]));
        let (doors, people) = s.populate(&mut w).unwrap();
        assert_eq!(doors, vec![0]);
        assert!(!w.bodies[0].active);
        assert_eq!(w.bodies[1].kind, Kind::Railing);
        let post_count = w.bodies.iter().filter(|b| b.kind == Kind::Post).count();
        assert_eq!(post_count, 11); // every 0.12 m over 1.2 m, both ends
        let person = &w.bodies[people[0]];
        assert_eq!(person.kind, Kind::Person);
        assert!(w.bodies.iter().any(|b| b.kind == Kind::Dock));
        assert_eq!(w.cliffs.len(), 1);
        assert!((s.start_pose()[2] - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn typos_and_bad_values_are_rejected() {
        assert!(Scenario::parse("durration_s: 5").is_err());
        assert!(Scenario::parse("initial_map: maybe").is_err());
        assert!(Scenario::parse("events: [{t: 1}]").is_err());
        assert!(Scenario::parse("events: [{t: 1, battery_soc: 0.1, end: true}]").is_err());
        assert!(Scenario::parse("doors: [{name: d, segment: [[0,0],[1,0]], events: [{t: 1, state: ajar}]}]").is_err());
    }

    #[test]
    fn people_walk_their_paths() {
        let mut p = PersonCfg { path: vec![[0.0, 0.0], [2.0, 0.0]], speed_mps: 1.0, pause_s: 0.0, ..Default::default() };
        assert_eq!(person_position(&p, 0.5), [0.5, 0.0]);
        assert_eq!(person_position(&p, 3.0), [1.0, 0.0]); // pingpong: on the way back
        p.mode = "once".into();
        assert!((person_position(&p, 10.0)[0] - 2.0).abs() < 1e-6);
        p.mode = "loop".into();
        assert_eq!(person_position(&p, 4.5), [0.5, 0.0]);
    }
}
