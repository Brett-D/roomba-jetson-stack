//! RosState of roomba_web/server.py without the ROS plumbing: what each subscription stores, the
//! state snapshot (websocket and /api/state), the JSON API's bodies and request validation, with
//! the same rounding, thresholds, freshness windows and JSON text as the Python node.

use std::sync::Arc;

use super::http::Reply;
use super::image;
use super::json::{self, py_eq, py_float, py_sort_by_key, py_str, Py, PyErr};
use crate::tf_buffer::Transform;
use crate::zones::py_round;

/// Drive pad command lifetime; the page repeats it every 0.1 s.
pub const DRIVE_HOLD_S: f64 = 0.35;
/// Concurrent /stream.mjpg viewers.
pub const MAX_STREAMS: usize = 4;
/// aiohttp's client_max_size: request bodies of this size or more are refused (413).
pub const MAX_BODY: usize = 256 * 1024;
/// The page's CSRF header (X-Requested-With) value.
pub const CSRF_VALUE: &str = "roomba-web";

/// SensorState flags shown as hazards, in order (several share a text).
pub const HAZARDS: [&str; 10] = [
    "Bumper pressed",
    "Bumper pressed",
    "Wheel lifted",
    "Wheel lifted",
    "Cliff detected",
    "Cliff detected",
    "Cliff detected",
    "Cliff detected",
    "On the dock",
    "Charger plugged in",
];

/// yaw of a quaternion (x, y, z, w), as server.py's _yaw.
pub fn yaw(q: [f64; 4]) -> f64 {
    let [x, y, z, w] = q;
    (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z))
}

/// One detection as the page lists it.
#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    pub cls: String,
    pub score: f64,
    pub dist: f64,
    pub reason: Option<String>,
    pub xy: Option<[f64; 2]>,
}

impl Detection {
    /// From a Detection3D: the first hypothesis (class, score), the box centre's z as the distance,
    /// the id as the exclusion reason and the centre in the map frame when TF had it.
    pub fn new(hypothesis: Option<(&str, f64)>, z: f64, id: &str, map_point: Option<[f64; 3]>) -> Detection {
        Detection {
            cls: hypothesis.map_or("?", |h| h.0).to_string(),
            score: hypothesis.map_or(0.0, |h| py_round(h.1, 2)),
            dist: py_round(z, 2),
            reason: (!id.is_empty()).then(|| id.to_string()),
            xy: map_point.map(|p| [py_round(p[0], 3), py_round(p[1], 3)]),
        }
    }

    fn to_py(&self, excluded: bool) -> Py {
        let mut d = vec![
            ("cls".into(), Py::str(&self.cls)),
            ("score".into(), Py::Float(self.score)),
            ("dist".into(), Py::Float(self.dist)),
            ("reason".into(), self.reason.as_deref().map_or(Py::None, Py::str)),
        ];
        if let Some([x, y]) = self.xy {
            d.push(("x".into(), Py::Float(x)));
            d.push(("y".into(), Py::Float(y)));
        }
        d.push(("excluded".into(), Py::Bool(excluded)));
        Py::Dict(d)
    }
}

/// An OccupancyGrid's fields the server uses.
pub struct Grid<'a> {
    pub width: usize,
    pub height: usize,
    pub resolution: f32,
    pub origin: [f64; 2],
    pub orientation: [f64; 4],
    pub frame: &'a str,
    pub data: &'a [i8],
}

/// The robot's battery as the page shows it.
#[derive(Clone, Debug, PartialEq)]
pub struct Battery {
    percent: Option<Py>,
    voltage: f64,
    charging: bool,
}

/// Python's round(x) (no digits) of a finite float: an int, halves to even.
fn round_int(x: f64) -> Py {
    let r = x.round_ties_even();
    if r.abs() < 9.0e18 {
        Py::Int(r as i64)
    } else {
        Py::BigInt(format!("{r:.0}"))
    }
}

/// [[round(x, 3), round(y, 3)], ...] of a path, every ceil(n / max)-th point when longer than max.
pub fn path_points(raw: &[[f64; 2]], max: usize) -> Vec<[f64; 2]> {
    let pts = raw.iter().map(|p| [py_round(p[0], 3), py_round(p[1], 3)]);
    if raw.len() > max {
        pts.step_by(raw.len().div_ceil(max)).collect()
    } else {
        pts.collect()
    }
}

/// server.py _on_costmap: lethal cells (>= 100) in row-major order, at most 3000 picked by
/// np.linspace, cell centres transformed into the map frame and rounded to mm.
pub fn costmap_points(grid: &Grid, tf: &Transform) -> Option<Vec<[f64; 2]>> {
    if grid.data.len() != grid.width * grid.height {
        return None;
    }
    let cells: Vec<usize> = grid.data.iter().enumerate().filter(|(_, v)| **v >= 100).map(|(i, _)| i).collect();
    let keep: Vec<usize> = if cells.len() > 3000 {
        let n = cells.len();
        let step = (n - 1) as f64 / 2999.0;
        (0..3000).map(|i| if i == 2999 { cells[n - 1] } else { cells[(i as f64 * step) as usize] }).collect()
    } else {
        cells
    };
    let r = grid.resolution as f64;
    Some(
        keep.into_iter()
            .map(|i| {
                let (row, col) = (i / grid.width, i % grid.width);
                let p = tf.apply([grid.origin[0] + (col as f64 + 0.5) * r, grid.origin[1] + (row as f64 + 0.5) * r, 0.0]);
                [py_round(p[0], 3), py_round(p[1], 3)]
            })
            .collect(),
    )
}

/// The pose sent to the page: map_frame <- base_frame, mm and mrad.
pub fn pose_py(translation: [f64; 3], rotation: [f64; 4]) -> Py {
    Py::Dict(vec![
        ("x".into(), Py::Float(py_round(translation[0], 3))),
        ("y".into(), Py::Float(py_round(translation[1], 3))),
        ("yaw".into(), Py::Float(py_round(yaw(rotation), 3))),
    ])
}

fn versioned(version: i64, key: &str, value: Py) -> String {
    json::dumps(&Py::Dict(vec![("version".into(), Py::Int(version)), (key.into(), value)]))
}

/// RosState's fields (everything the ROS callbacks store). Times are time.monotonic() seconds.
pub struct Core {
    pub localization: bool,
    pub max_drive_speed: f64,
    pub max_drive_turn: f64,
    pub map_png: Option<Arc<Vec<u8>>>,
    map_meta: Option<Py>,
    pub map_version: i64,
    zones: Py,
    zones_version: i64,
    path: Vec<[f64; 2]>,
    path_version: i64,
    /// active, excluded: (time, items)
    dets: [(f64, Vec<Detection>); 2],
    vo_state: u8,
    vo_time: f64,
    localized: bool,
    last_filled_map: f64,
    nav_instruction: Option<(f64, Py)>,
    pub nav_goal: Option<[f64; 2]>,
    plan: Vec<[f64; 2]>,
    plan_version: i64,
    obstacles: Vec<[f64; 2]>,
    obstacles_version: i64,
    floor: Py,
    vslam_filter: Py,
    robot_mode: Option<String>,
    robot_mode_time: f64,
    battery: Option<Battery>,
    battery_time: f64,
    hazards: Vec<&'static str>,
    robot_msg: Option<(f64, String)>,
    /// (v, w, time) of the latest drive pad command
    pub drive: Option<(f64, f64, f64)>,
    explore: Option<(f64, Py)>,
    pub floor_grid: bool,
}

impl Core {
    pub fn new(localization: bool, max_drive_speed: f64, max_drive_turn: f64, floor_grid: bool) -> Core {
        Core {
            localization,
            max_drive_speed,
            max_drive_turn,
            map_png: None,
            map_meta: None,
            map_version: 0,
            zones: Py::List(Vec::new()),
            zones_version: 0,
            path: Vec::new(),
            path_version: 0,
            dets: [(0.0, Vec::new()), (0.0, Vec::new())],
            vo_state: 0,
            vo_time: 0.0,
            localized: !localization,
            last_filled_map: -1e9,
            nav_instruction: None,
            nav_goal: None,
            plan: Vec::new(),
            plan_version: 0,
            obstacles: Vec::new(),
            obstacles_version: 0,
            floor: Py::None,
            vslam_filter: Py::None,
            robot_mode: None,
            robot_mode_time: 0.0,
            battery: None,
            battery_time: 0.0,
            hazards: Vec::new(),
            robot_msg: None,
            drive: None,
            explore: None,
            floor_grid,
        }
    }

    // ---- subscriptions

    pub fn on_dets(&mut self, excluded: bool, now: f64, items: Vec<Detection>) {
        self.dets[excluded as usize] = (now, items);
    }

    /// Whether a grid from /rtabmap/map is used now (one from /rtabmap/map_filled is, and holds
    /// the raw map off for 10 s). Call before encoding.
    pub fn take_map(&mut self, now: f64, filled: bool) -> bool {
        if filled {
            self.last_filled_map = now;
            true
        } else {
            now - self.last_filled_map >= 10.0
        }
    }

    /// Stores an encoded map (image::map_png of `grid`).
    pub fn set_map(&mut self, grid: &Grid, png: Vec<u8>) {
        self.map_version += 1;
        self.map_png = Some(Arc::new(png));
        self.map_meta = Some(Py::Dict(vec![
            ("version".into(), Py::Int(self.map_version)),
            ("resolution".into(), Py::Float(grid.resolution as f64)),
            ("width".into(), Py::Int(grid.width as i64)),
            ("height".into(), Py::Int(grid.height as i64)),
            ("origin".into(), Py::List(vec![Py::Float(grid.origin[0]), Py::Float(grid.origin[1])])),
            ("yaw".into(), Py::Float(yaw(grid.orientation))),
            ("frame".into(), Py::str(grid.frame)),
        ]));
    }

    /// _on_map: returns whether the map changed.
    pub fn on_map(&mut self, now: f64, filled: bool, grid: &Grid) -> bool {
        if !self.take_map(now, filled) {
            return false;
        }
        match image::map_png(grid.width, grid.height, grid.data) {
            Some(png) => {
                self.set_map(grid, png);
                true
            }
            None => false,
        }
    }

    pub fn on_path(&mut self, raw: &[[f64; 2]]) {
        self.path = path_points(raw, 1000);
        self.path_version += 1;
    }

    pub fn on_plan(&mut self, raw: &[[f64; 2]]) {
        self.plan = path_points(raw, 500);
        self.plan_version += 1;
    }

    pub fn set_obstacles(&mut self, points: Vec<[f64; 2]>) {
        self.obstacles = points;
        self.obstacles_version += 1;
    }

    pub fn on_zones(&mut self, text: &str) {
        if let Ok(v @ Py::Dict(_)) = json::loads(text) {
            self.zones = v.get("zones").cloned().unwrap_or(Py::List(Vec::new()));
            self.zones_version += 1;
        }
    }

    pub fn on_status(&mut self, vo_state: u8, now: f64) {
        self.vo_state = vo_state;
        self.vo_time = now;
    }

    pub fn on_localized(&mut self) {
        self.localized = true;
    }

    pub fn on_instruction(&mut self, text: &str, now: f64) {
        let Ok(data) = json::loads(text) else { return };
        let done = matches!(&data, Py::Dict(_))
            && !data.get_or_none("active").truthy()
            && matches!(data.get_or_none("code"), Py::Str(c) if c == "arrived" || c == "failed" || c == "canceled");
        self.nav_instruction = Some((now, data));
        if done {
            self.nav_goal = None;
        }
    }

    pub fn on_floor(&mut self, text: &str) {
        if let Ok(v) = json::loads(text) {
            self.floor = v;
        }
    }

    pub fn on_filter(&mut self, text: &str) {
        if let Ok(v) = json::loads(text) {
            self.vslam_filter = v;
        }
    }

    pub fn on_robot_mode(&mut self, mode: &str, now: f64) {
        self.robot_mode = Some(mode.to_string());
        self.robot_mode_time = now;
    }

    /// BatteryState: percentage (0..1, NaN when unknown), voltage, charging = status 1.
    pub fn on_battery(&mut self, percentage: f32, voltage: f32, power_supply_status: u8, now: f64) {
        let pct = percentage as f64;
        self.battery = Some(Battery {
            percent: pct.is_finite().then(|| round_int(pct * 100.0)),
            voltage: py_round(voltage as f64, 2),
            charging: power_supply_status == 1,
        });
        self.battery_time = now;
    }

    /// SensorState: the HAZARDS flags (bump_left, bump_right, wheel_drop_left, wheel_drop_right,
    /// cliff_left, cliff_front_left, cliff_front_right, cliff_right, home_base_available,
    /// internal_charger_available) and the OI mode.
    pub fn on_sensors(&mut self, flags: [bool; 10], oi_mode: u8, now: f64) {
        let mut found: Vec<&'static str> = Vec::new();
        for (on, text) in flags.iter().zip(HAZARDS) {
            if *on && !found.contains(&text) {
                found.push(text);
            }
        }
        self.hazards = found;
        self.robot_mode_time = now;
        let mode = match oi_mode {
            0 => Some("off"),
            1 => Some("passive"),
            2 => Some("safe"),
            3 => Some("full"),
            _ => None,
        };
        if let Some(m) = mode {
            self.robot_mode = Some(m.into());
        }
    }

    pub fn on_mode_result(&mut self, now: f64, text: String) {
        self.robot_msg = Some((now, text));
    }

    pub fn on_explore(&mut self, text: &str, now: f64) {
        if let Ok(v) = json::loads(text) {
            self.explore = Some((now, v));
        }
    }

    // ---- web side

    pub fn robot_available(&self, now: f64) -> bool {
        now - self.battery_time < 3.0
    }

    /// set_drive: clamps to the drive limits.
    pub fn set_drive(&mut self, v: f64, w: f64, now: f64) {
        let v = (-self.max_drive_speed).max(self.max_drive_speed.min(v));
        let w = (-self.max_drive_turn).max(self.max_drive_turn.min(w));
        self.drive = Some((v, w, now));
    }

    /// The drive command still held at `now`, if any.
    pub fn held_drive(&self, now: f64) -> Option<(f64, f64)> {
        self.drive.filter(|d| now - d.2 < DRIVE_HOLD_S).map(|d| (d.0, d.1))
    }

    /// explore_command('stop') only sends while the explorer is driving.
    pub fn exploring(&self) -> bool {
        self.explore.as_ref().is_some_and(|(_, s)| {
            matches!(s, Py::Dict(_)) && matches!(s.get_or_none("state"), Py::Str(st) if st == "exploring" || st == "returning")
        })
    }

    /// The state dict, given the parts that come from ROS objects (navigation server ready,
    /// robot pose, explorer subscribed).
    pub fn snapshot(&self, now: f64, nav_available: bool, pose: Py, explore_subscribed: bool) -> Py {
        let mut dets = Vec::new();
        for (kind, (stamp, items)) in self.dets.iter().enumerate() {
            if now - stamp < 0.6 {
                dets.extend(items.iter().map(|d| (d.dist, d.to_py(kind == 1))));
            }
        }
        let dets = py_sort_by_key(dets, |d| d.0).into_iter().map(|d| d.1).collect();
        let tracking = if now - self.vo_time > 2.0 {
            "no data"
        } else {
            match self.vo_state {
                1 => "ok",
                2 => "lost",
                _ => "starting",
            }
        };
        let nav = match &self.nav_instruction {
            Some((t, data)) if now - t < 2.0 => data.clone(),
            _ => Py::None,
        };
        let robot = if now - self.battery_time < 3.0 {
            let mode = if now - self.robot_mode_time < 3.0 { self.robot_mode.as_deref().map_or(Py::None, Py::str) } else { Py::str("unknown") };
            let battery = self.battery.as_ref().map_or(Py::None, |b| {
                Py::Dict(vec![
                    ("percent".into(), b.percent.clone().unwrap_or(Py::None)),
                    ("voltage".into(), Py::Float(b.voltage)),
                    ("charging".into(), Py::Bool(b.charging)),
                ])
            });
            let message = match &self.robot_msg {
                Some((t, text)) if now - t < 6.0 => Py::str(text),
                _ => Py::None,
            };
            Py::Dict(vec![
                ("mode".into(), mode),
                ("battery".into(), battery),
                ("hazards".into(), Py::List(self.hazards.iter().map(|h| Py::str(h)).collect())),
                ("max_speed".into(), Py::Float(self.max_drive_speed)),
                ("max_turn".into(), Py::Float(self.max_drive_turn)),
                ("message".into(), message),
            ])
        } else {
            Py::None
        };
        let explore = match &self.explore {
            Some((_, status)) if explore_subscribed => status.clone(),
            _ => Py::None,
        };
        let nav_goal = self.nav_goal.map_or(Py::None, |[x, y]| Py::Dict(vec![("x".into(), Py::Float(x)), ("y".into(), Py::Float(y))]));
        Py::Dict(vec![
            ("tracking".into(), Py::str(tracking)),
            ("mode".into(), Py::str(if self.localization { "localization" } else { "mapping" })),
            ("localized".into(), Py::Bool(self.localized)),
            ("detections".into(), Py::List(dets)),
            ("map_version".into(), Py::Int(self.map_version)),
            ("zones_version".into(), Py::Int(self.zones_version)),
            ("path_version".into(), Py::Int(self.path_version)),
            ("plan_version".into(), Py::Int(self.plan_version)),
            ("obstacles_version".into(), Py::Int(self.obstacles_version)),
            ("nav_goal".into(), nav_goal),
            ("floor".into(), self.floor.clone()),
            ("vslam_filter".into(), self.vslam_filter.clone()),
            ("nav".into(), nav),
            ("robot".into(), robot),
            ("nav_available".into(), Py::Bool(nav_available)),
            ("pose".into(), pose),
            ("explore".into(), explore),
            ("floor_grid".into(), Py::Bool(self.floor_grid)),
        ])
    }

    pub fn map_json(&self) -> Option<String> {
        self.map_meta.as_ref().map(json::dumps)
    }

    pub fn path_json(&self) -> String {
        versioned(self.path_version, "points", Py::float_list(&self.path))
    }

    pub fn plan_json(&self) -> String {
        versioned(self.plan_version, "points", Py::float_list(&self.plan))
    }

    pub fn obstacles_json(&self) -> String {
        versioned(self.obstacles_version, "points", Py::float_list(&self.obstacles))
    }

    pub fn zones_json(&self) -> String {
        versioned(self.zones_version, "zones", self.zones.clone())
    }

    /// `name and any(z['name'] == name for z in zones)`; Err where Python raises (a 500).
    pub fn zone_exists(&self, name: &Py) -> Result<bool, ()> {
        if !name.truthy() {
            return Ok(false);
        }
        for z in self.zones.iter().map_err(|_| ())? {
            if py_eq(z.get("name").map_err(|_| ())?, name) {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

// ---------------------------------------------------------------- request bodies

/// await request.json(): Err(None) for a ValueError (bad charset data or JSON), Err(Some(500))
/// for an unknown charset.
fn json_body(body: &[u8], charset: Option<&str>) -> Result<Py, Option<Reply>> {
    let text = super::http::decode(body, charset).map_err(|unknown| unknown.then(Reply::internal_error))?;
    json::loads(&text).map_err(|_| None)
}

/// The handlers catch (ValueError, KeyError, TypeError[, IndexError]) as a 400 with `bad`;
/// OverflowError (an int too large for float()) is a 500 as in Python.
fn fail(bad: &str) -> impl Fn(PyErr) -> Reply + '_ {
    move |e| if e == PyErr::Overflow { Reply::internal_error() } else { Reply::text(400, bad) }
}

fn parse<'a>(body: &[u8], charset: Option<&str>, bad: &'a str) -> Result<Py, Reply> {
    json_body(body, charset).map_err(|e| e.unwrap_or_else(|| Reply::text(400, bad)))
}

/// POST /api/zones: (name, points, kind)
pub fn parse_add_zone(body: &[u8], charset: Option<&str>) -> Result<(Py, Vec<[f64; 2]>, String), Reply> {
    const BAD: &str = "expected {\"points\": [[x, y], ...]}";
    let v = parse(body, charset, BAD)?;
    let f = fail(BAD);
    let mut pts = Vec::new();
    for p in v.get("points").map_err(&f)?.iter().map_err(&f)? {
        let x = py_float(&p.index(0).map_err(&f)?).map_err(&f)?;
        let y = py_float(&p.index(1).map_err(&f)?).map_err(&f)?;
        pts.push([x, y]);
    }
    if !(3..=64).contains(&pts.len()) {
        return Err(Reply::text(400, "a zone needs 3 to 64 corners"));
    }
    let kind = v.get_or_none("kind");
    let kind = match kind.truthy() {
        false => "exclusion".to_string(),
        true => match kind {
            Py::Str(s) if s == "exclusion" || s == "keepout" => s,
            _ => return Err(Reply::text(400, "kind must be \"exclusion\" or \"keepout\"")),
        },
    };
    Ok((v.get_or_none("name"), pts, kind))
}

pub fn zone_conflict(name: &Py) -> Reply {
    Reply::text(409, &format!("zone \"{}\" already exists", py_str(name)))
}

/// The /perception/ignore_zones/command JSON for an added zone.
pub fn add_zone_command(name: Py, pts: &[[f64; 2]], kind: &str) -> String {
    json::dumps(&Py::Dict(vec![
        ("op".into(), Py::str("add")),
        ("name".into(), name),
        ("points".into(), Py::float_list(pts)),
        ("kind".into(), Py::str(kind)),
    ]))
}

pub fn delete_zone_command(name: &str) -> String {
    json::dumps(&Py::Dict(vec![("op".into(), Py::str("delete")), ("name".into(), Py::str(name))]))
}

fn two_floats(body: &[u8], charset: Option<&str>, keys: [&str; 2], bad: &str) -> Result<[f64; 2], Reply> {
    let v = parse(body, charset, bad)?;
    let f = fail(bad);
    let a = py_float(v.get(keys[0]).map_err(&f)?).map_err(&f)?;
    let b = py_float(v.get(keys[1]).map_err(&f)?).map_err(&f)?;
    Ok([a, b])
}

/// POST /api/goal: (x, y), finite and within 1 km.
pub fn parse_goal(body: &[u8], charset: Option<&str>) -> Result<[f64; 2], Reply> {
    let [x, y] = two_floats(body, charset, ["x", "y"], "expected {\"x\": .., \"y\": ..}")?;
    if !(x.is_finite() && y.is_finite()) || x.abs() > 1000.0 || y.abs() > 1000.0 {
        return Err(Reply::text(400, "goal out of range"));
    }
    Ok([x, y])
}

/// POST /api/drive: (v, w), finite (clamped later).
pub fn parse_drive(body: &[u8], charset: Option<&str>) -> Result<[f64; 2], Reply> {
    let [v, w] = two_floats(body, charset, ["v", "w"], "expected {\"v\": .., \"w\": ..}")?;
    if !(v.is_finite() && w.is_finite()) {
        return Err(Reply::text(400, "bad velocity"));
    }
    Ok([v, w])
}

/// POST /api/robot/mode: the SetMode mode (passive 1, safe 2).
pub fn parse_mode(body: &[u8], charset: Option<&str>) -> Result<u8, Reply> {
    const BAD: &str = "expected {\"mode\": \"safe\" | \"passive\"}";
    let v = parse(body, charset, BAD)?;
    match v.get("mode").map_err(fail(BAD))? {
        Py::Str(m) if m == "passive" => Ok(1),
        Py::Str(m) if m == "safe" => Ok(2),
        _ => Err(Reply::text(400, BAD)),
    }
}

/// POST /api/explore: start, stop or home.
pub fn parse_explore(body: &[u8], charset: Option<&str>) -> Result<String, Reply> {
    const BAD: &str = "expected {\"action\": \"start\" | \"stop\" | \"home\"}";
    let v = parse(body, charset, BAD)?;
    match v.get("action").map_err(fail(BAD))? {
        Py::Str(a) if a == "start" || a == "stop" || a == "home" => Ok(a.clone()),
        _ => Err(Reply::text(400, "action must be start, stop or home")),
    }
}

/// POST /api/floor_grid: bool(body['on'])
pub fn parse_floor_grid(body: &[u8], charset: Option<&str>) -> Result<bool, Reply> {
    const BAD: &str = "expected {\"on\": true | false}";
    let v = parse(body, charset, BAD)?;
    Ok(v.get("on").map_err(fail(BAD))?.truthy())
}

pub fn ok_json() -> Reply {
    Reply::json("{\"ok\": true}".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body_status<T: std::fmt::Debug>(r: Result<T, Reply>) -> (u16, String) {
        match r {
            Ok(v) => (200, format!("{v:?}")),
            Err(e) => (e.status, String::from_utf8_lossy(&e.body).into_owned()),
        }
    }

    #[test]
    fn request_validation_matches_server_py() {
        let z = |b: &str| body_status(parse_add_zone(b.as_bytes(), None));
        assert_eq!(z(r#"{"points": [[0, 0], [1, "2.5"], [true, 1]]}"#).0, 200);
        assert_eq!(z(r#"{"points": [[0, 0], [1, 2]]}"#), (400, "a zone needs 3 to 64 corners".into()));
        assert_eq!(z(r#"{"points": {"12": 1, "34": 2, "56": 3}}"#).0, 200, "dict keys are iterated like Python");
        assert_eq!(z(r#"{"points": "123456"}"#).0, 400);
        assert_eq!(z(r#"[1]"#).0, 400);
        assert_eq!(z("not json").0, 400);
        assert_eq!(z(r#"{"points": [[0,0],[1,0],[1,1]], "kind": "wall"}"#).1, "kind must be \"exclusion\" or \"keepout\"");
        assert!(z(r#"{"points": [[0,0],[1,0],[1,1]], "kind": ""}"#).1.contains("exclusion"));
        assert_eq!(z(&format!(r#"{{"points": [[{}, 0], [1, 0], [1, 1]]}}"#, "9".repeat(400))).0, 500);
        let g = |b: &str| body_status(parse_goal(b.as_bytes(), None));
        assert_eq!(g(r#"{"x": 1, "y": -2.5}"#), (200, "[1.0, -2.5]".into()));
        assert_eq!(g(r#"{"x": NaN, "y": 0}"#), (400, "goal out of range".into()));
        assert_eq!(g(r#"{"x": 1001, "y": 0}"#), (400, "goal out of range".into()));
        assert_eq!(g(r#"{"x": 1}"#), (400, "expected {\"x\": .., \"y\": ..}".into()));
        let d = |b: &str| body_status(parse_drive(b.as_bytes(), None));
        assert_eq!(d(r#"{"v": "inf", "w": 0}"#), (400, "bad velocity".into()));
        assert_eq!(d(r#"{"v": null, "w": 0}"#).0, 400);
        assert_eq!(body_status(parse_mode(br#"{"mode": "safe"}"#, None)), (200, "2".into()));
        assert_eq!(body_status(parse_mode(br#"{"mode": "full"}"#, None)).0, 400);
        assert_eq!(body_status(parse_mode(br#"{"mode": ["safe"]}"#, None)).0, 400);
        assert_eq!(body_status(parse_explore(br#"{"action": 1}"#, None)), (400, "action must be start, stop or home".into()));
        assert_eq!(body_status(parse_floor_grid(br#"{"on": "no"}"#, None)), (200, "true".into()));
        assert_eq!(body_status(parse_floor_grid(br#"{"on": 0}"#, None)), (200, "false".into()));
        assert_eq!(body_status(parse_floor_grid(b"\xff", None)).0, 400);
        assert_eq!(body_status(parse_floor_grid(b"{}", Some("klingon"))).0, 500);
        assert_eq!(add_zone_command(Py::None, &[[1.0, 2.5]], "keepout"), r#"{"op": "add", "name": null, "points": [[1.0, 2.5]], "kind": "keepout"}"#);
    }

    #[test]
    fn drive_clamps_and_expires() {
        let mut c = Core::new(false, 0.2, 1.0, true);
        c.set_drive(5.0, -3.0, 10.0);
        assert_eq!(c.held_drive(10.34), Some((0.2, -1.0)));
        assert_eq!(c.held_drive(10.36), None);
        c.set_drive(-0.1, 0.5, 11.0);
        assert_eq!(c.held_drive(11.0), Some((-0.1, 0.5)));
    }

    #[test]
    fn snapshot_layout_and_freshness() {
        let mut c = Core::new(true, 0.2, 1.0, false);
        c.on_status(1, 100.0);
        c.on_dets(false, 100.0, vec![Detection::new(Some(("person", 0.876)), 2.345, "", Some([1.0, 2.0, 0.0]))]);
        c.on_dets(true, 100.0, vec![Detection::new(None, 1.0, "zone_1", None)]);
        c.on_battery(0.555, 14.236, 1, 100.0);
        c.on_sensors([false, true, false, false, true, false, false, false, true, false], 2, 100.0);
        c.on_explore(r#"{"state": "exploring"}"#, 100.0);
        let s = json::dumps(&c.snapshot(100.5, false, Py::None, true));
        assert_eq!(
            s,
            r#"{"tracking": "ok", "mode": "localization", "localized": false, "detections": [{"cls": "?", "score": 0.0, "dist": 1.0, "reason": "zone_1", "excluded": true}, {"cls": "person", "score": 0.88, "dist": 2.35, "reason": null, "x": 1.0, "y": 2.0, "excluded": false}], "map_version": 0, "zones_version": 0, "path_version": 0, "plan_version": 0, "obstacles_version": 0, "nav_goal": null, "floor": null, "vslam_filter": null, "nav": null, "robot": {"mode": "safe", "battery": {"percent": 56, "voltage": 14.24, "charging": true}, "hazards": ["Bumper pressed", "Cliff detected", "On the dock"], "max_speed": 0.2, "max_turn": 1.0, "message": null}, "nav_available": false, "pose": null, "explore": {"state": "exploring"}, "floor_grid": false}"#
        );
        let later = json::dumps(&c.snapshot(103.0, false, Py::None, false));
        assert!(later.contains(r#""tracking": "no data""#) && later.contains(r#""detections": []"#));
        assert!(later.contains(r#""robot": null"#) && later.contains(r#""explore": null"#));
        assert!(c.exploring());
    }

    #[test]
    fn maps_paths_and_costmaps() {
        let mut c = Core::new(false, 0.2, 1.0, true);
        let data = [0i8, 100, -1, 65];
        let grid = Grid { width: 2, height: 2, resolution: 0.05, origin: [1.0, -2.0], orientation: [0.0, 0.0, 0.0, 1.0], frame: "map", data: &data };
        assert!(c.on_map(50.0, true, &grid));
        assert!(!c.on_map(55.0, false, &grid), "raw map held off by the filled one");
        assert!(c.on_map(60.0, false, &grid));
        assert_eq!(
            c.map_json().unwrap(),
            r#"{"version": 2, "resolution": 0.05000000074505806, "width": 2, "height": 2, "origin": [1.0, -2.0], "yaw": 0.0, "frame": "map"}"#
        );
        let raw: Vec<[f64; 2]> = (0..1001).map(|i| [i as f64 * 0.0011, 0.0]).collect();
        let p = path_points(&raw, 1000);
        assert_eq!((p.len(), p[1][0]), (501, 0.002));
        let lethal = vec![100i8; 5000];
        let grid = Grid { width: 100, height: 50, resolution: 0.1, origin: [0.0, 0.0], orientation: [0.0, 0.0, 0.0, 1.0], frame: "odom", data: &lethal };
        let pts = costmap_points(&grid, &Transform::IDENTITY).unwrap();
        assert_eq!(pts.len(), 3000);
        assert_eq!((pts[0], pts[2999]), ([0.05, 0.05], [9.95, 4.95]));
    }

    #[test]
    fn instructions_clear_the_goal() {
        let mut c = Core::new(false, 0.2, 1.0, true);
        c.nav_goal = Some([1.0, 2.0]);
        c.on_instruction(r#"{"code": "forward", "active": true}"#, 1.0);
        assert!(c.nav_goal.is_some());
        c.on_instruction(r#"{"code": "arrived", "active": false, "text": "\u2191"}"#, 1.0);
        assert!(c.nav_goal.is_none());
        assert!(json::dumps(&c.snapshot(1.5, true, Py::None, false)).contains(r#""nav": {"code": "arrived", "active": false, "text": "\u2191"}"#));
    }
}
