//! nav_coach's logic: turns Nav2's velocity commands, the collision monitor's state, obstacles in
//! the stop zone and the navigate_to_pose goal status into a held, flicker-free instruction (or,
//! with robot_mode, a description of what the robot is doing). Same behaviour and JSON as
//! roomba_perception/nav_coach.py, checked by tools/ab_nav_coach_offline.py.

use serde_json::{Map, Value};

use crate::watchdog::ParamSource;
use crate::zones::{py_json, py_round};

/// (code, arrow, instruction for a person, description of the robot).
pub const TEXTS: [(&str, &str, &str, &str); 12] = [
    ("idle", "", "Tap the map to set a goal", "Tap the map to send the robot"),
    ("forward", "⬆", "Go forward", "Driving forward"),
    ("bear_left", "↖", "Forward, bear left", "Driving, bearing left"),
    ("bear_right", "↗", "Forward, bear right", "Driving, bearing right"),
    ("turn_left", "⟲", "Turn left", "Turning left"),
    ("turn_right", "⟳", "Turn right", "Turning right"),
    ("back", "⬇", "Back up slowly", "Backing up"),
    ("wait", "⏸", "Wait…", "Planning…"),
    ("stop_obstacle", "✋", "STOP – obstacle ahead", "Stopped – obstacle ahead"),
    ("arrived", "✅", "You have arrived", "Arrived"),
    ("failed", "⚠", "No path found", "Could not reach the goal"),
    ("canceled", "⏹", "Navigation stopped", "Navigation stopped"),
];

/// nav2_msgs/CollisionMonitorState STOP.
pub const CM_STOP: u8 = 1;

#[derive(Clone, Debug)]
pub struct Params {
    pub min_lin: f64,
    pub min_ang: f64,
    pub bear_ratio: f64,
    pub hold_s: f64,
    pub robot_mode: bool,
    pub stop_dist: f64,
    pub stop_half_width: f64,
    pub stop_min_points: i64,
    pub base_frame: String,
}

impl Params {
    /// Declares the parameters in the Python node's order, with its names, types and defaults.
    pub fn load(p: &mut impl ParamSource) -> anyhow::Result<Self> {
        Ok(Params {
            min_lin: p.float("min_linear_mps", 0.04)?,
            min_ang: p.float("min_angular_rps", 0.15)?,
            bear_ratio: p.float("bear_angular_rps", 0.35)?,
            hold_s: p.float("hold_s", 0.6)?,
            robot_mode: p.boolean("robot_mode", false)?,
            stop_dist: p.float("stop_distance_m", 0.45)?,
            stop_half_width: p.float("stop_half_width_m", 0.22)?,
            stop_min_points: p.int("stop_min_points", 15)?,
            base_frame: p.string("base_frame", "camera_link")?,
        })
    }
}

/// The rotation matrix of a quaternion, as nav_coach.py's _quat_to_matrix computes it.
pub fn quat_to_matrix(x: f64, y: f64, z: f64, w: f64) -> [[f64; 3]; 3] {
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

/// Number of points (float32 x, y, z triples, little endian) inside the stop zone after the
/// transform (translation, quaternion xyzw) into the base frame: 0 < x < stop_dist, |y| < half_width.
/// The rotation is accumulated with fused multiply-adds, as numpy's matmul (BLAS) does here.
pub fn points_ahead(data: &[u8], t: [f64; 3], q: [f64; 4], stop_dist: f64, half_width: f64) -> usize {
    let r = quat_to_matrix(q[0], q[1], q[2], q[3]);
    let mut n = 0;
    for p in data.chunks_exact(12) {
        let f = |i: usize| f32::from_le_bytes([p[i], p[i + 1], p[i + 2], p[i + 3]]) as f64;
        let (px, py, pz) = (f(0), f(4), f(8));
        let x = pz.mul_add(r[0][2], py.mul_add(r[0][1], px * r[0][0])) + t[0];
        let y = pz.mul_add(r[1][2], py.mul_add(r[1][1], px * r[1][0])) + t[1];
        if x > 0.0 && x < stop_dist && y.abs() < half_width {
            n += 1;
        }
    }
    n
}

pub struct NavCoach {
    p: Params,
    /// (monotonic time, linear.x, angular.z) of the last /cmd_vel_nav and /cmd_vel.
    cmd_nav: (f64, f64, f64),
    cmd_out: (f64, f64, f64),
    cm_stop: (f64, bool),
    active: bool,
    result: (f64, Option<&'static str>),
    distance: Option<f64>,
    current: &'static str,
    since: f64,
    pending: (Option<&'static str>, f64),
    ahead: (f64, bool),
}

impl NavCoach {
    pub fn new(p: Params) -> Self {
        NavCoach {
            p,
            cmd_nav: (0.0, 0.0, 0.0),
            cmd_out: (0.0, 0.0, 0.0),
            cm_stop: (0.0, false),
            active: false,
            result: (0.0, None),
            distance: None,
            current: "idle",
            since: 0.0,
            pending: (None, 0.0),
            ahead: (0.0, false),
        }
    }

    pub fn params(&self) -> &Params {
        &self.p
    }

    /// Whether the last obstacle cloud had enough points in the stop zone.
    pub fn blocked(&self) -> bool {
        self.ahead.1
    }

    pub fn on_cmd_nav(&mut self, vx: f64, wz: f64, now: f64) {
        self.cmd_nav = (now, vx, wz);
    }

    pub fn on_cmd_out(&mut self, vx: f64, wz: f64, now: f64) {
        self.cmd_out = (now, vx, wz);
    }

    pub fn on_collision_monitor(&mut self, action_type: u8, now: f64) {
        self.cm_stop = (now, action_type == CM_STOP);
    }

    /// An obstacle cloud, with its transform into base_frame (None: TF unavailable, ignored).
    pub fn on_obstacles(&mut self, data: &[u8], width: u32, tf: Option<([f64; 3], [f64; 4])>, now: f64) {
        let Some((t, q)) = tf else { return };
        let data = if width > 0 { data } else { &[] };
        let n = points_ahead(data, t, q, self.p.stop_dist, self.p.stop_half_width);
        self.ahead = (now, n as i64 >= self.p.stop_min_points);
    }

    /// /navigate_to_pose/_action/status: (stamp sec, nanosec, status) of each goal.
    pub fn on_status(&mut self, list: &[(i32, u32, i8)], now: f64) {
        let Some(mut latest) = list.first() else { return };
        for s in &list[1..] {
            if (s.0, s.1) > (latest.0, latest.1) {
                latest = s;
            }
        }
        let st = latest.2;
        let was_active = self.active;
        self.active = matches!(st, 1..=3); // ACCEPTED, EXECUTING, CANCELING
        if was_active && !self.active {
            let code = match st {
                4 => Some("arrived"),
                5 => Some("canceled"),
                6 => Some("failed"),
                _ => None,
            };
            self.result = (now, code);
            self.distance = None;
        }
    }

    pub fn on_feedback(&mut self, distance_remaining: f32) {
        self.distance = Some(distance_remaining as f64);
    }

    fn classify(&self, now: f64) -> &'static str {
        let p = &self.p;
        if !self.active {
            let (t, code) = self.result;
            return match code {
                Some(c) if now - t < 15.0 => c,
                _ => "idle",
            };
        }
        let (t_cm, stop) = self.cm_stop;
        if stop && now - t_cm < 1.0 {
            return "stop_obstacle";
        }
        let (t, cvx, cwz) = self.cmd_out;
        let (vx, wz) = if now - t <= 0.6 { (cvx, cwz) } else { (0.0, 0.0) };
        let turning_in_place = vx.abs() < p.min_lin && wz.abs() >= p.min_ang;
        let (t_ahead, blocked) = self.ahead;
        if blocked && now - t_ahead < 1.0 && !turning_in_place && vx > -p.min_lin {
            return "stop_obstacle";
        }
        if now - t > 0.6 {
            return "wait";
        }
        let (t_nav, nvx, nwz) = self.cmd_nav;
        if vx.abs() < p.min_lin
            && wz.abs() < p.min_ang
            && now - t_nav < 0.6
            && (nvx.abs() >= p.min_lin || nwz.abs() >= p.min_ang)
        {
            return "stop_obstacle"; // collision monitor zeroed a real command
        }
        if vx <= -p.min_lin {
            return "back";
        }
        if vx >= p.min_lin {
            if wz > p.bear_ratio {
                return "bear_left";
            }
            if wz < -p.bear_ratio {
                return "bear_right";
            }
            return "forward";
        }
        if wz >= p.min_ang {
            return "turn_left";
        }
        if wz <= -p.min_ang {
            return "turn_right";
        }
        "wait"
    }

    /// The 0.2 s timer: the instruction JSON to publish.
    pub fn tick(&mut self, now: f64) -> String {
        let code = self.classify(now);
        // Stops and results apply immediately; movement changes must persist hold_s (no flicker).
        let immediate = matches!(code, "stop_obstacle" | "arrived" | "failed" | "canceled" | "idle");
        if code != self.current {
            if immediate {
                self.current = code;
                self.since = now;
            } else if self.pending.0 != Some(code) {
                self.pending = (Some(code), now);
            } else if now - self.pending.1 >= self.p.hold_s {
                self.current = code;
                self.since = now;
            }
        } else {
            self.pending = (None, 0.0);
        }
        let &(code, arrow, person, robot) = TEXTS.iter().find(|t| t.0 == self.current).unwrap();
        let mut m = Map::new();
        m.insert("code".into(), code.into());
        m.insert("arrow".into(), arrow.into());
        m.insert("text".into(), if self.p.robot_mode { robot } else { person }.into());
        m.insert("active".into(), self.active.into());
        m.insert("distance_m".into(), self.distance.map_or(Value::Null, |d| Value::from(py_round(d, 2))));
        py_json(&Value::Object(m), None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> Params {
        Params {
            min_lin: 0.04,
            min_ang: 0.15,
            bear_ratio: 0.35,
            hold_s: 0.6,
            robot_mode: true,
            stop_dist: 0.45,
            stop_half_width: 0.22,
            stop_min_points: 15,
            base_frame: "base_link".into(),
        }
    }

    #[test]
    fn holds_movement_but_not_results() {
        let mut c = NavCoach::new(params());
        c.on_status(&[(1, 0, 2)], 0.0);
        c.on_cmd_out(0.2, 0.0, 0.0);
        assert!(c.tick(0.1).contains("\"idle\""));
        c.on_cmd_out(0.2, 0.0, 0.5);
        assert!(c.tick(0.5).contains("\"idle\""));
        c.on_cmd_out(0.2, 0.0, 0.8);
        assert!(c.tick(0.8).contains("\"forward\""));
        c.on_status(&[(1, 0, 4)], 0.9);
        let s = c.tick(1.0);
        assert_eq!(s, r#"{"code": "arrived", "arrow": "\u2705", "text": "Arrived", "active": false, "distance_m": null}"#);
    }

    #[test]
    fn latest_goal_wins_ties_to_first() {
        let mut c = NavCoach::new(params());
        c.on_status(&[(5, 0, 2), (5, 0, 4), (3, 0, 6)], 0.0);
        assert!(c.active);
    }

    #[test]
    fn stop_zone_count() {
        let pts: Vec<u8> = [[0.2f32, 0.0, 0.0], [0.5, 0.0, 0.0], [0.2, 0.3, 0.0]]
            .iter()
            .flat_map(|p| p.iter().flat_map(|v| v.to_le_bytes()))
            .collect();
        assert_eq!(points_ahead(&pts, [0.0; 3], [0.0, 0.0, 0.0, 1.0], 0.45, 0.22), 1);
    }
}
