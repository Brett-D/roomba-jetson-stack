//! Deterministic replay of explorer's core for tools/ab_explorer_offline.py. Reads events from stdin,
//! one JSON array per line, [kind, mono_s, ros_ns, ...]:
//!   ["P", 0, 0, {params}, wall0, grid_file]      parameters (first line)
//!   ["G", t, ros, offset, len, width, height, resolution, origin_x, origin_y]   costmap (data in grid_file)
//!   ["TF", t, ros, is_static, [[parent, child, stamp_ns, tx, ty, tz, qx, qy, qz, qw], ...]]
//!   ["M", t, ros, mode] / ["B", t, ros, percentage|null] / ["F", t, ros, filter_status] / ["C", t, ros, command]
//!   ["S", t, ros, bump_left, bump_right, wheel_drop_left, wheel_drop_right, [cliff x4], [light x6]]
//!   ["K", t, ros] 1 s tick / ["CT", t, ros] cliff check timer
//!   ["DR", t, ros, frame, [[x, y, z], ...]] drop points (float32) / ["PD", t, ros, frame, [[x, y, z], ...]] people seen
//!   ["NR", t, ros, nav_ready, get_params_ready, set_params_ready(, spin_ready)]   server availability
//!   ["GR", t, ros, seq, accepted] / ["RS", t, ros, seq, status]       NavigateToPose responses
//!   ["SR", t, ros, seq, accepted] / ["SS", t, ros, seq, status]       Spin (look-around) responses
//!   ["PR", t, ros, type|null, double_value, error|null]              GetParameters response
//!   ["Q", t, ros(, x, y)]   frontier query at the TF pose (or x, y): prints FQ
//!   ["X", t, ros]           shutdown
//! and prints one JSON array per output: ["ST", json] / ["ZN", json] / ["LG", level, message] /
//! ["GL", seq, frame, x, y, qz, qw, stamp_ns] / ["CN", seq] / ["GP"] / ["SP", value] / ["TM"] /
//! ["SG", seq, target_yaw, sec, nanosec] / ["SC", seq] /
//! ["MK", [[frame, stamp_ns, ns, id, type, action, [x, y, z, qw], [sx, sy, sz], [r, g, b, a], [[x, y, z], ...], text], ...]]
//! (~/markers, a DELETEALL first) / ["DC", frame, [[x, y, z], ...]] (~/drops, float32) /
//! ["FQ", [[goal_row, goal_col, centroid_row, centroid_col, size], ...], seed|null, [row, col], [peeks], reach_cells]
//! or ["FQ", null].
use std::fs::File;
use std::io::{self, BufRead, BufWriter, Read, Seek, SeekFrom, Write};

use serde_json::{json, Map, Value};

use roomba_rs::explorer::{frontier_goals, Explorer, FrontierParams, GoalRequest, Grid, Io, Params, Sensors};
use roomba_rs::tf_buffer::{TfBuffer, Transform};
use roomba_rs::watchdog::{Level, ParamSource};
use roomba_rs::zones::{MarkerSpec, DELETEALL};

struct Overrides(Map<String, Value>);

impl ParamSource for Overrides {
    fn float(&mut self, name: &str, d: f64) -> anyhow::Result<f64> {
        Ok(self.0.get(name).and_then(Value::as_f64).unwrap_or(d))
    }
    fn int(&mut self, name: &str, d: i64) -> anyhow::Result<i64> {
        Ok(self.0.get(name).and_then(Value::as_i64).unwrap_or(d))
    }
    fn boolean(&mut self, name: &str, d: bool) -> anyhow::Result<bool> {
        Ok(self.0.get(name).and_then(Value::as_bool).unwrap_or(d))
    }
    fn string(&mut self, name: &str, d: &str) -> anyhow::Result<String> {
        Ok(self.0.get(name).and_then(Value::as_str).unwrap_or(d).to_string())
    }
}

struct Replay<W: Write> {
    out: W,
    t: f64,
    ros: i64,
    wall0: f64,
    tf: TfBuffer,
    ready: [bool; 4],
}

impl<W: Write> Replay<W> {
    fn emit(&mut self, v: Value) {
        writeln!(self.out, "{v}").unwrap();
    }
}

impl<W: Write> Io for Replay<W> {
    fn now(&mut self) -> f64 {
        // rclpy: self.get_clock().now().nanoseconds * 1e-9
        self.ros as f64 * 1e-9
    }
    fn lookup(&mut self, target: &str, source: &str) -> Option<([f64; 3], [f64; 4])> {
        let t = self.tf.lookup_transform(target, source, 0).ok()?;
        Some((t.transform.translation, t.transform.rotation))
    }
    fn nav_ready(&mut self) -> bool {
        self.ready[0]
    }
    fn send_goal(&mut self, g: GoalRequest) {
        let ros = self.ros;
        self.emit(json!(["GL", g.seq, g.frame, g.x, g.y, g.qz, g.qw, ros]));
    }
    fn cancel_goal(&mut self, seq: u64) {
        self.emit(json!(["CN", seq]));
    }
    fn get_params_ready(&mut self) -> bool {
        self.ready[1]
    }
    fn set_params_ready(&mut self) -> bool {
        self.ready[2]
    }
    fn get_rotate(&mut self) {
        self.emit(json!(["GP"]));
    }
    fn set_rotate(&mut self, value: f64) {
        self.emit(json!(["SP", value]));
    }
    fn status(&mut self, s: String) {
        self.emit(json!(["ST", s]));
    }
    fn zone_command(&mut self, s: String) {
        self.emit(json!(["ZN", s]));
    }
    fn log(&mut self, level: Level, msg: String) {
        let l = match level {
            Level::Info => "info",
            Level::Warn => "warn",
            Level::Error => "error",
        };
        self.emit(json!(["LG", l, msg]));
    }
    fn local_hms(&mut self) -> String {
        let secs = (self.wall0 + self.t).floor() as libc::time_t;
        // SAFETY: gmtime_r only writes `tm`.
        let tm = unsafe {
            let mut tm: libc::tm = std::mem::zeroed();
            libc::gmtime_r(&secs, &mut tm);
            tm
        };
        format!("{:02}{:02}{:02}", tm.tm_hour, tm.tm_min, tm.tm_sec)
    }
    fn start_cliff_timer(&mut self) {
        self.emit(json!(["TM"]));
    }
    fn spin_ready(&mut self) -> bool {
        self.ready[3]
    }
    fn send_spin(&mut self, seq: u64, target_yaw: f32, sec: i32, nanosec: u32) {
        self.emit(json!(["SG", seq, target_yaw as f64, sec, nanosec]));
    }
    fn cancel_spin(&mut self, seq: u64) {
        self.emit(json!(["SC", seq]));
    }
    fn markers_wanted(&mut self) -> bool {
        true
    }
    fn markers(&mut self, frame: &str, specs: &[MarkerSpec]) {
        let ros = self.ros;
        let one = |m: &MarkerSpec| {
            let c = m.color.map(|v| v as f64);
            json!([frame, ros, m.ns, m.id, m.kind, m.action, [m.position[0], m.position[1], m.position[2], m.orientation_w],
                   m.scale, c, m.points, m.text])
        };
        let clear = MarkerSpec {
            ns: String::new(),
            id: 0,
            kind: 0,
            action: DELETEALL,
            position: [0.0; 3],
            orientation_w: 1.0,
            scale: [0.0; 3],
            color: [0.0; 4],
            points: Vec::new(),
            text: String::new(),
        };
        let all: Vec<Value> = std::iter::once(&clear).chain(specs.iter()).map(one).collect();
        self.emit(json!(["MK", all]));
    }
    fn drops(&mut self, frame: &str, points: &[[f32; 3]]) {
        let pts: Vec<Value> = points.iter().map(|p| json!([p[0] as f64, p[1] as f64, p[2] as f64])).collect();
        self.emit(json!(["DC", frame, pts]));
    }
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn b(v: &Value) -> bool {
    v.as_bool().unwrap_or(false)
}

/// The [[x, y, z], ...] of a DR / PD event.
fn points(e: &Value) -> Vec<[f64; 3]> {
    e[4].as_array().unwrap().iter().map(|p| [f(&p[0]), f(&p[1]), f(&p[2])]).collect()
}

fn frontier_query(io: &mut Replay<impl Write>, p: &Params, grid: Option<&Grid>, at: Option<(f64, f64)>) -> Value {
    let Some(grid) = grid else { return json!(["FQ", null]) };
    let pose = match at {
        Some(xy) => Some(xy),
        None => io.lookup(&p.map_frame, &p.base_frame).map(|(t, _)| (t[0], t[1])),
    };
    let Some((x, y)) = pose else { return json!(["FQ", null]) };
    let res = grid.resolution as f64;
    let robot = (((y - grid.origin_y) / res) as i64, ((x - grid.origin_x) / res) as i64);
    let fp = FrontierParams {
        free_max: p.free_max,
        pass_max: p.pass_max,
        clear_max: p.clear_max,
        min_cells: p.min_cells,
        standoff_cells: p.standoff / res,
        search_cells: p.goal_search / res,
        cost_weight: p.cost_weight,
        peek_cells: p.peek_search / res,
        split_cells: p.frontier_split / res,
    };
    let (goals, seed, peeks, reach) = frontier_goals(&grid.data, grid.width as usize, grid.height as usize, robot, &fp);
    let list = |gs: &[roomba_rs::explorer::FrontierGoal]| -> Vec<Value> {
        gs.iter().map(|g| json!([g.goal.0, g.goal.1, g.centroid.0, g.centroid.1, g.size])).collect()
    };
    json!(["FQ", list(&goals), seed.map(|s| json!([s.0, s.1])), [robot.0, robot.1], list(&peeks), reach])
}

fn main() -> anyhow::Result<()> {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let first: Value = serde_json::from_str(&lines.next().expect("no parameter line")?)?;
    let params = first[3].as_object().cloned().unwrap_or_default();
    let grid_path = first[5].as_str().unwrap_or("").to_string();
    let mut grids = if grid_path.is_empty() { None } else { Some(File::open(&grid_path)?) };
    let mut io =
        Replay { out: BufWriter::new(io::stdout().lock()), t: 0.0, ros: 0, wall0: f(&first[4]), tf: TfBuffer::new(), ready: [true; 4] };
    let p = Params::load(&mut Overrides(params))?;
    let mut core = Explorer::new(p.clone(), &mut io);
    let mut last_grid: Option<Grid> = None;
    for line in lines {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let e: Value = serde_json::from_str(&line)?;
        let kind = e[0].as_str().unwrap_or("");
        io.t = f(&e[1]);
        io.ros = e[2].as_i64().unwrap_or(0);
        match kind {
            "G" => {
                let (offset, len) = (e[3].as_u64().unwrap(), e[4].as_u64().unwrap() as usize);
                let file = grids.as_mut().expect("no grid file");
                file.seek(SeekFrom::Start(offset))?;
                let mut buf = vec![0u8; len];
                file.read_exact(&mut buf)?;
                let grid = Grid {
                    width: e[5].as_u64().unwrap() as u32,
                    height: e[6].as_u64().unwrap() as u32,
                    resolution: f(&e[7]) as f32,
                    origin_x: f(&e[8]),
                    origin_y: f(&e[9]),
                    data: buf.into_iter().map(|v| v as i8).collect(),
                };
                last_grid = Some(grid.clone());
                core.on_grid(grid);
            }
            "TF" => {
                let is_static = b(&e[3]);
                for t in e[4].as_array().unwrap() {
                    let tr = Transform { translation: [f(&t[3]), f(&t[4]), f(&t[5])], rotation: [f(&t[6]), f(&t[7]), f(&t[8]), f(&t[9])] };
                    let _ = io.tf.set_transform(
                        t[0].as_str().unwrap(),
                        t[1].as_str().unwrap(),
                        t[2].as_i64().unwrap(),
                        &tr,
                        "default_authority",
                        is_static,
                    );
                }
            }
            "M" => core.on_mode(e[3].as_str().unwrap(), &mut io),
            "B" => core.on_battery(e[3].as_f64().map_or(f32::NAN, |v| v as f32)),
            "F" => core.on_filter(e[3].as_str().unwrap()),
            "C" => core.on_command(e[3].as_str().unwrap(), &mut io),
            "S" => {
                let arr = |v: &Value| v.as_array().unwrap().clone();
                let cliff = arr(&e[7]);
                let light = arr(&e[8]);
                let s = Sensors {
                    bump_left: b(&e[3]),
                    bump_right: b(&e[4]),
                    wheel_drop_left: b(&e[5]),
                    wheel_drop_right: b(&e[6]),
                    cliff: std::array::from_fn(|i| b(&cliff[i])),
                    light: std::array::from_fn(|i| light[i].as_u64().unwrap() as u16),
                };
                core.on_sensors(&s, &mut io);
            }
            "K" => core.tick(&mut io),
            "CT" => core.cliff_check(&mut io),
            "DR" => {
                let pts: Vec<[f32; 3]> = points(&e).iter().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect();
                core.on_drops(e[3].as_str().unwrap(), &pts, &mut io);
            }
            "PD" => core.on_persons(e[3].as_str().unwrap(), &points(&e), &mut io),
            "NR" => io.ready = [b(&e[3]), b(&e[4]), b(&e[5]), e.get(6).map_or(io.ready[3], b)],
            "GR" => core.on_goal_response(e[3].as_u64().unwrap(), b(&e[4]), &mut io),
            "RS" => core.on_result(e[3].as_u64().unwrap(), e[4].as_i64().unwrap() as i8, &mut io),
            "SR" => core.on_spin_response(e[3].as_u64().unwrap(), b(&e[4]), &mut io),
            "SS" => core.on_spin_result(e[3].as_u64().unwrap(), e[4].as_i64().unwrap() as i8, &mut io),
            "PR" => {
                let value = match (&e[5], &e[3], &e[4]) {
                    (Value::String(err), _, _) => Err(err.clone()),
                    (_, Value::Null, _) => Ok(None),
                    (_, t, v) => Ok(Some((t.as_u64().unwrap() as u8, f(v)))),
                };
                core.on_saved_rotate(value, &mut io);
            }
            "Q" => {
                let at = if e.as_array().unwrap().len() >= 5 { Some((f(&e[3]), f(&e[4]))) } else { None };
                let v = frontier_query(&mut io, &p, last_grid.as_ref(), at);
                io.emit(v);
            }
            "X" => core.shutdown(&mut io),
            other => anyhow::bail!("unknown event {other}"),
        }
    }
    io.out.flush()?;
    Ok(())
}
