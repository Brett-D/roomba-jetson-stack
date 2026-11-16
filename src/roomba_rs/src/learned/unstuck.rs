//! The unstuck policy: observation layout, action mapping and the ONNX model.
//!
//! Observation (596 values, exactly UnstuckEnv._get_observations in
//! twin/learned/unstuck/unstuck_env.py):
//!
//! | index    | value                                                                    |
//! |----------|--------------------------------------------------------------------------|
//! | 0..576   | lethal local-costmap patch (see [`crate::learned::patch`])               |
//! | 576..582 | light bumpers L, FL, CL, CR, FR, R as proximity 0..1                     |
//! | 582, 583 | bumper left, right (0/1; this robot ignores its stuck right switch: 0)   |
//! | 584, 585 | wheel-encoder v / 0.25 m/s, w / 0.8 rad/s                                |
//! | 586      | IMU yaw rate / 0.8 rad/s                                                 |
//! | 587, 588 | previous policy action (-1..1)                                           |
//! | 589..592 | motion over the last 1 s in the current robot frame: dx/0.25, dy/0.25, dyaw/0.8 |
//! | 592..595 | goal: min(distance/3 m, 1), cos(bearing), sin(bearing) (robot frame)     |
//! | 595      | time since the skill started / 15 s                                       |
//!
//! Action: a = (a0, a1) in [-1, 1]; v = 0.25 a0 forward or 0.15 a0 backward, w = 0.8 a1. The
//! commands go through the velocity smoother and the collision monitor like Nav2's.

use std::path::Path;

use anyhow::Result;

use super::onnx::OnnxModel;
use super::patch::{GridView, PATCH};

pub const OBS_DIM: usize = PATCH * PATCH + 20;
pub const V_MAX: f64 = 0.25;
pub const V_MIN: f64 = -0.15;
pub const W_MAX: f64 = 0.8;
/// Time normalisation used in training (episode length).
pub const EPISODE_S: f64 = 15.0;

/// Everything besides the costmap patch, in robot units.
#[derive(Clone, Debug, Default)]
pub struct UnstuckInputs {
    /// Light-bumper proximity 0..1 (see [`light_proximity`]), order L, FL, CL, CR, FR, R.
    pub light: [f32; 6],
    pub bump_left: bool,
    pub bump_right: bool,
    /// Wheel-encoder odometry (m/s, rad/s).
    pub v_enc: f64,
    pub w_enc: f64,
    /// IMU yaw rate (rad/s).
    pub yaw_rate: f64,
    pub prev_action: [f32; 2],
    /// Pose change over the last 1 s expressed in the current robot frame (m, m, rad).
    pub motion_1s: [f64; 3],
    /// The goal the robot was driving to, in the robot frame (m).
    pub goal: (f64, f64),
    pub elapsed_s: f64,
}

/// Light-bumper signal -> proximity 0..1 as in training (1 - d / 0.12 m). `reference` is the
/// signal this sensor reads with an obstacle about 1 cm away (calibrate per sensor; this robot's
/// front-left reads ~235, the others under 140).
pub fn light_proximity(signal: u16, reference: f32) -> f32 {
    if reference <= 0.0 {
        return 0.0;
    }
    (signal as f32 / reference).clamp(0.0, 1.0)
}

pub fn observation(patch: &[f32], i: &UnstuckInputs) -> Vec<f32> {
    assert_eq!(patch.len(), PATCH * PATCH);
    let mut o = Vec::with_capacity(OBS_DIM);
    o.extend_from_slice(patch);
    o.extend_from_slice(&i.light);
    o.push(i.bump_left as u8 as f32);
    o.push(i.bump_right as u8 as f32);
    o.push((i.v_enc / V_MAX) as f32);
    o.push((i.w_enc / W_MAX) as f32);
    o.push((i.yaw_rate / W_MAX) as f32);
    o.extend_from_slice(&i.prev_action);
    o.push((i.motion_1s[0] / 0.25) as f32);
    o.push((i.motion_1s[1] / 0.25) as f32);
    o.push((i.motion_1s[2] / W_MAX) as f32);
    let (gx, gy) = i.goal;
    let d = gx.hypot(gy);
    let b = gy.atan2(gx);
    o.push((d / 3.0).min(1.0) as f32);
    o.push(b.cos() as f32);
    o.push(b.sin() as f32);
    o.push((i.elapsed_s / EPISODE_S) as f32);
    o
}

/// Policy action -> (v m/s, w rad/s).
pub fn action_to_cmd(a: [f32; 2]) -> (f64, f64) {
    let a0 = (a[0] as f64).clamp(-1.0, 1.0);
    let a1 = (a[1] as f64).clamp(-1.0, 1.0);
    let v = if a0 > 0.0 { a0 * V_MAX } else { -a0 * V_MIN };
    (v, a1 * W_MAX)
}

pub struct UnstuckPolicy {
    model: OnnxModel,
}

impl UnstuckPolicy {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Ok(UnstuckPolicy { model: OnnxModel::load(path, &[&[1, OBS_DIM]])? })
    }

    /// The deterministic action (the actor's mean, clamped to [-1, 1]).
    pub fn act(&self, obs: &[f32]) -> Result<[f32; 2]> {
        let y = self.model.run(&[obs])?;
        anyhow::ensure!(y.len() == 2, "unexpected policy output of {} values", y.len());
        Ok([y[0], y[1]])
    }
}

/// Distance (m) from (x, y) to the nearest lethal cell centre of `g` within `max` (else `max`).
pub fn clearance(g: &GridView, x: f64, y: f64, max: f64) -> f64 {
    let res = g.resolution;
    let n = (max / res).ceil() as i64 + 1;
    let c0 = ((x - g.origin_x) / res).floor() as i64;
    let r0 = ((y - g.origin_y) / res).floor() as i64;
    let mut best = max;
    for r in (r0 - n)..=(r0 + n) {
        for c in (c0 - n)..=(c0 + n) {
            if r < 0 || c < 0 || r >= g.height as i64 || c >= g.width as i64 {
                continue;
            }
            if g.data[r as usize * g.width + c as usize] >= super::patch::LETHAL {
                let cx = g.origin_x + (c as f64 + 0.5) * res;
                let cy = g.origin_y + (r as f64 + 0.5) * res;
                best = best.min((cx - x).hypot(cy - y));
            }
        }
    }
    best
}

#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    Succeeded,
    TimedOut,
    Failed(String),
}

/// One control tick's measurements (robot units; poses in the map frame from VSLAM).
pub struct Tick<'a> {
    pub now: f64,
    pub patch: &'a [f32],
    /// Free space around the robot on the local costmap (see [`clearance`]).
    pub clearance_m: f64,
    pub light: [f32; 6],
    pub bump_left: bool,
    pub bump_right: bool,
    pub v_enc: f64,
    pub w_enc: f64,
    pub yaw_rate: f64,
    pub pose: (f64, f64, f64),
    pub goal: Option<(f64, f64)>,
    /// A cliff sensor or wheel drop fired, or the Roomba left Safe / Full mode.
    pub cliff_or_passive: bool,
    /// floor_obstacles reported a blind depth frame recently (closer than 0.3 m to something: the
    /// costmap can look free while the robot is nose to a wall).
    pub camera_blind: bool,
}

/// One run of the skill: time limit, success check (0.3 m clearance held 0.5 s, bumper free,
/// camera not blind),
/// pose history for the motion features, and the previous action.
pub struct Session {
    start: f64,
    limit: f64,
    hist: std::collections::VecDeque<(f64, [f64; 3])>,
    prev_action: [f32; 2],
    free_since: Option<f64>,
    was_bumped: bool,
    pub bumps: u32,
    pub success_clearance: f64,
}

impl Session {
    pub fn new(now: f64, limit_s: f64) -> Self {
        Session {
            start: now,
            limit: limit_s,
            hist: Default::default(),
            prev_action: [0.0; 2],
            free_since: None,
            was_bumped: false,
            bumps: 0,
            success_clearance: 0.30,
        }
    }

    pub fn elapsed(&self, now: f64) -> f64 {
        now - self.start
    }

    /// The observation for this tick (also records the pose history).
    pub fn observation(&mut self, t: &Tick) -> Vec<f32> {
        let (x, y, yaw) = t.pose;
        self.hist.push_back((t.now, [x, y, yaw]));
        while self.hist.front().is_some_and(|h| t.now - h.0 > 1.05) {
            self.hist.pop_front();
        }
        let old = self.hist.front().map_or([x, y, yaw], |h| h.1);
        let (c, s) = (yaw.cos(), yaw.sin());
        let (dx, dy) = (x - old[0], y - old[1]);
        let dyaw = (yaw - old[2]).sin().atan2((yaw - old[2]).cos());
        // no goal: 1.5 m straight back, away from what is in front
        let goal = t.goal.map_or((-1.5, 0.0), |(gx, gy)| (c * (gx - x) + s * (gy - y), -s * (gx - x) + c * (gy - y)));
        let inputs = UnstuckInputs {
            light: t.light,
            bump_left: t.bump_left,
            bump_right: t.bump_right,
            v_enc: t.v_enc,
            w_enc: t.w_enc,
            yaw_rate: t.yaw_rate,
            prev_action: self.prev_action,
            motion_1s: [c * dx + s * dy, -s * dx + c * dy, dyaw],
            goal,
            elapsed_s: self.elapsed(t.now),
        };
        observation(t.patch, &inputs)
    }

    /// Decide this tick: Ok((v, w)) to drive, or the outcome (the caller then stops the robot).
    pub fn tick(&mut self, policy: &UnstuckPolicy, t: &Tick) -> std::result::Result<(f64, f64), Outcome> {
        if t.cliff_or_passive {
            return Err(Outcome::Failed("cliff, wheel drop or the Roomba left Safe mode".into()));
        }
        let bumped = t.bump_left || t.bump_right;
        if bumped && !self.was_bumped {
            self.bumps += 1;
        }
        self.was_bumped = bumped;
        if t.clearance_m >= self.success_clearance && !bumped && !t.camera_blind {
            let since = *self.free_since.get_or_insert(t.now);
            if t.now - since >= 0.5 {
                return Err(Outcome::Succeeded);
            }
        } else {
            self.free_since = None;
        }
        if self.elapsed(t.now) >= self.limit {
            return Err(Outcome::TimedOut);
        }
        let obs = self.observation(t);
        match policy.act(&obs) {
            Ok(a) => {
                self.prev_action = a;
                Ok(action_to_cmd(a))
            }
            Err(e) => Err(Outcome::Failed(format!("policy: {e}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn models() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models")
    }

    #[test]
    fn layout_and_actions() {
        let patch = vec![0f32; PATCH * PATCH];
        let i = UnstuckInputs { bump_left: true, goal: (0.0, 2.0), elapsed_s: 7.5, ..Default::default() };
        let o = observation(&patch, &i);
        assert_eq!(o.len(), OBS_DIM);
        assert_eq!(o[582], 1.0);
        assert!((o[592] - 2.0 / 3.0).abs() < 1e-6 && o[593].abs() < 1e-6 && (o[594] - 1.0).abs() < 1e-6);
        assert!((o[595] - 0.5).abs() < 1e-6);
        assert_eq!(action_to_cmd([1.0, -1.0]), (0.25, -0.8));
        let (v, _) = action_to_cmd([-1.0, 0.0]);
        assert!((v + 0.15).abs() < 1e-12);
        assert_eq!(light_proximity(470, 235.0), 1.0);
    }

    /// The exported ONNX policy reproduces the PyTorch actor on recorded observations.
    #[test]
    fn onnx_matches_pytorch() {
        let path = models().join("unstuck_policy.onnx");
        let vec_path = models().join("unstuck_vectors.json");
        let policy = UnstuckPolicy::load(&path).expect("load unstuck_policy.onnx");
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(vec_path).unwrap()).unwrap();
        let inputs = v["inputs"].as_array().unwrap();
        let outputs = v["outputs"].as_array().unwrap();
        assert!(!inputs.is_empty());
        let mut worst = 0f32;
        for (x, y) in inputs.iter().zip(outputs) {
            let x: Vec<f32> = x.as_array().unwrap().iter().map(|a| a.as_f64().unwrap() as f32).collect();
            let a = policy.act(&x).unwrap();
            for (k, want) in y.as_array().unwrap().iter().enumerate() {
                worst = worst.max((a[k] - want.as_f64().unwrap() as f32).abs());
            }
        }
        assert!(worst <= 1e-5, "max |rust - pytorch| = {worst}");
    }

    #[test]
    fn session_succeeds_times_out_and_stops_on_cliff() {
        let policy = UnstuckPolicy::load(models().join("unstuck_policy.onnx")).unwrap();
        let patch = vec![0f32; PATCH * PATCH];
        let tick = |now: f64, clear: f64, cliff: bool, blind: bool| Tick {
            now,
            patch: &patch,
            clearance_m: clear,
            light: [0.0; 6],
            bump_left: false,
            bump_right: false,
            v_enc: 0.0,
            w_enc: 0.0,
            yaw_rate: 0.0,
            pose: (0.0, 0.0, 0.0),
            goal: Some((2.0, 0.0)),
            cliff_or_passive: cliff,
            camera_blind: blind,
        };
        let mut s = Session::new(0.0, 15.0);
        let (v, w) = s.tick(&policy, &tick(0.0, 0.1, false, false)).unwrap();
        assert!((-0.15..=0.25).contains(&v) && w.abs() <= 0.8);
        assert!(s.tick(&policy, &tick(0.1, 0.4, false, false)).is_ok()); // free, but not for 0.5 s yet
        assert_eq!(s.tick(&policy, &tick(0.7, 0.4, false, false)), Err(Outcome::Succeeded));
        let mut s = Session::new(0.0, 15.0);
        assert_eq!(s.tick(&policy, &tick(15.0, 0.1, false, false)), Err(Outcome::TimedOut));
        let mut s = Session::new(0.0, 15.0);
        assert!(matches!(s.tick(&policy, &tick(1.0, 0.1, true, false)), Err(Outcome::Failed(_))));
        // nose to a wall: the costmap looks free but the camera is blind -> keep going
        let mut s = Session::new(0.0, 15.0);
        assert!(s.tick(&policy, &tick(0.0, 0.6, false, true)).is_ok());
        assert!(s.tick(&policy, &tick(1.0, 0.6, false, true)).is_ok());
    }

    #[test]
    fn clearance_finds_the_nearest_lethal_cell() {
        let mut d = vec![0i8; 40 * 40];
        d[20 * 40 + 30] = 100; // cell centre (0.525, 0.025) with origin (-1, -1)
        let g = GridView { width: 40, height: 40, resolution: 0.05, origin_x: -1.0, origin_y: -1.0, data: &d };
        assert!((clearance(&g, 0.025, 0.025, 1.0) - 0.5).abs() < 1e-9);
        assert_eq!(clearance(&g, -0.9, -0.9, 0.3), 0.3);
    }
}
