//! explorer's logic: frontier exploration on Nav2's global costmap. Same behaviour as
//! roomba_perception/explorer.py (see its docstring): the same frontier groups, goal cells, scores
//! and tie-breaking (OpenCV's connected-component order, numpy's argmin/argwhere order, its
//! float32 distance transform and float rounding), the same state machine (blacklist, visited
//! spots, bump and cliff keep-out marks, retries, return home, finish checks), status JSON, log
//! lines and Nav2 / parameter-service requests. Checked by tools/ab_explorer_offline.py.
//!
//! ROS-free: [`Explorer`] reacts to inputs and talks to the outside through [`Io`] (the node in
//! src/bin/explorer.rs, or the replay in examples/explorer_replay.rs).

use std::collections::VecDeque;

use serde_json::{Map, Value};

use crate::watchdog::{Level, ParamSource};
use crate::zones::{py_json, py_round};

pub const ROTATE_PARAM: &str = "FollowPath.rotate_to_heading_angular_vel";
/// rcl_interfaces/ParameterType PARAMETER_DOUBLE.
pub const PARAMETER_DOUBLE: u8 = 3;
/// action_msgs/GoalStatus STATUS_SUCCEEDED.
pub const STATUS_SUCCEEDED: i8 = 4;
/// Cliff sensors (SensorState fields) and their direction on the Roomba (rad, left positive).
pub const CLIFF_ANGLES: [(&str, f64); 4] =
    [("cliff_left", 1.05), ("cliff_front_left", 0.35), ("cliff_front_right", -0.35), ("cliff_right", -1.05)];
/// Light-bumper directions: left, front left, center left, center right, front right, right.
pub const LIGHT_ANGLES: [f64; 6] = [1.2, 0.65, 0.2, -0.2, -0.65, -1.2];

// ---- Python / numpy numerics ----

fn frexp_exp(x: f64) -> i32 {
    let e = ((x.to_bits() >> 52) & 0x7ff) as i32;
    if e == 0 {
        frexp_exp(x * f64::from_bits((1023 + 54) << 52)) - 54
    } else {
        e - 1022
    }
}

fn ldexp1(e: i32) -> f64 {
    if e >= -1022 {
        f64::from_bits(((e + 1023) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (e + 1074))
    }
}

fn dl_mul(x: f64, y: f64) -> (f64, f64) {
    let z = x * y;
    (z, x.mul_add(y, -z))
}

fn dl_fast_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    (x, (a - x) + b)
}

fn vector_norm(vec: &mut [f64], max: f64, found_nan: bool) -> f64 {
    if max.is_infinite() {
        return max;
    }
    if found_nan {
        return f64::NAN;
    }
    if max == 0.0 || vec.len() <= 1 {
        return max;
    }
    let max_e = frexp_exp(max);
    if max_e < -1023 {
        for v in vec.iter_mut() {
            *v /= f64::MIN_POSITIVE;
        }
        return f64::MIN_POSITIVE * vector_norm(vec, max / f64::MIN_POSITIVE, found_nan);
    }
    let scale = ldexp1(-max_e);
    let (mut csum, mut frac1, mut frac2) = (1.0f64, 0.0f64, 0.0f64);
    for &v in vec.iter() {
        let x = v * scale;
        let pr = dl_mul(x, x);
        let sm = dl_fast_sum(csum, pr.0);
        csum = sm.0;
        frac1 += pr.1;
        frac2 += sm.1;
    }
    let mut h = (csum - 1.0 + (frac1 + frac2)).sqrt();
    let pr = dl_mul(-h, h);
    let sm = dl_fast_sum(csum, pr.0);
    csum = sm.0;
    frac1 += pr.1;
    frac2 += sm.1;
    let x = csum - 1.0 + (frac1 + frac2);
    h += x / (2.0 * h);
    h / scale
}

/// CPython 3.12's math.hypot(x, y): not libm's hypot (exact squares, compensated sum and one
/// correction step), so distances compare exactly as in the Python node.
pub fn py_hypot(x: f64, y: f64) -> f64 {
    let mut v = [x.abs(), y.abs()];
    let mut max = 0.0f64;
    for &c in &v {
        if c > max {
            max = c;
        }
    }
    let nan = v[0].is_nan() || v[1].is_nan();
    vector_norm(&mut v, max, nan)
}

/// Python's sum() of floats (start 0): Neumaier-compensated as in CPython 3.12.
pub fn py_sum(items: impl IntoIterator<Item = f64>) -> f64 {
    let mut it = items.into_iter();
    let Some(first) = it.next() else { return 0.0 };
    let (mut f, mut c) = (0.0 + first, 0.0f64);
    for x in it {
        let t = f + x;
        if f.abs() >= x.abs() {
            c += (f - t) + x;
        } else {
            c += (x - t) + f;
        }
        f = t;
    }
    if c != 0.0 && c.is_finite() {
        f += c;
    }
    f
}

/// round(numpy.float64, 2): numpy's x * 100 -> rint -> / 100 (not Python's correctly rounded round()).
pub fn np_round2(x: f64) -> f64 {
    (x * 100.0).round_ties_even() / 100.0
}

/// Python str.strip() whitespace.
fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

// ---- keep-out polygons ----

/// Thin keep-out strip (map frame) on the surface the bumper touched, `contact` m from the robot
/// centre along `bearing`, extending `depth` m into the obstacle (explorer.py bump_zone).
pub fn bump_zone(x: f64, y: f64, yaw: f64, bearing: f64, width: f64) -> Vec<[f64; 2]> {
    let (contact, depth) = (0.18f64, 0.06f64);
    let a = yaw + bearing;
    let (ca, sa) = (a.cos(), a.sin());
    let h = width / 2.0;
    let corners = [(contact, -h), (contact + depth, -h), (contact + depth, h), (contact, h)];
    corners.iter().map(|&(u, v)| [py_round(x + u * ca - v * sa, 3), py_round(y + u * sa + v * ca, 3)]).collect()
}

/// Direction of a bumper hit (rad, + = left): from the light bumpers reading at least `light_min`,
/// else from the switches (explorer.py bump_bearing).
pub fn bump_bearing(s: &Sensors, light_min: i64) -> f64 {
    let hit = |i: usize| s.light[i] as i64 >= light_min;
    let total: i64 = (0..6).filter(|&i| hit(i)).map(|i| s.light[i] as i64).sum();
    if total > 0 {
        let sy = py_sum((0..6).filter(|&i| hit(i)).map(|i| s.light[i] as f64 * LIGHT_ANGLES[i].sin()));
        let sx = py_sum((0..6).filter(|&i| hit(i)).map(|i| s.light[i] as f64 * LIGHT_ANGLES[i].cos()));
        return sy.atan2(sx);
    }
    let (left, right) = (s.bump_left, s.bump_right);
    if left && !right {
        0.5
    } else if right && !left {
        -0.5
    } else {
        0.0
    }
}

/// Square keep-out polygon (map frame) centred `distance` ahead of the robot along `bearing`.
pub fn cliff_zone(x: f64, y: f64, yaw: f64, bearing: f64) -> Vec<[f64; 2]> {
    let (distance, size) = (0.42f64, 0.35f64);
    let a = yaw + bearing;
    let (cx, cy) = (x + distance * a.cos(), y + distance * a.sin());
    let h = size / 2.0;
    let corners = [(h, h), (-h, h), (-h, -h), (h, -h)];
    corners
        .iter()
        .map(|&(u, v)| [py_round(cx + u * a.cos() - v * a.sin(), 3), py_round(cy + u * a.sin() + v * a.cos(), 3)])
        .collect()
}

// ---- frontiers (OpenCV 4.6 / numpy 1.26 equivalents) ----

fn find(parent: &mut [u32], mut i: u32) -> u32 {
    while parent[i as usize] != i {
        parent[i as usize] = parent[parent[i as usize] as usize];
        i = parent[i as usize];
    }
    i
}

/// cv2.connectedComponents(img, connectivity=8): labels (0 = background) numbered as OpenCV does
/// (by each component's first 2 x 2 block in raster order), and the number of components.
pub fn labels8(img: &[u8], w: usize, h: usize) -> (Vec<u32>, usize) {
    let n = w * h;
    let mut parent: Vec<u32> = (0..n as u32).collect();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if img[i] == 0 {
                continue;
            }
            let mut join = |j: usize| {
                if img[j] != 0 {
                    let (a, b) = (find(&mut parent, i as u32), find(&mut parent, j as u32));
                    if a != b {
                        parent[a.max(b) as usize] = a.min(b);
                    }
                }
            };
            if x > 0 {
                join(i - 1);
            }
            if y > 0 {
                if x > 0 {
                    join(i - w - 1);
                }
                join(i - w);
                if x + 1 < w {
                    join(i - w + 1);
                }
            }
        }
    }
    let blocks_per_row = w.div_ceil(2);
    let mut first = vec![usize::MAX; n];
    let mut roots = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if img[i] == 0 {
                continue;
            }
            let r = find(&mut parent, i as u32) as usize;
            let block = (y / 2) * blocks_per_row + x / 2;
            if first[r] == usize::MAX {
                roots.push(r);
            }
            first[r] = first[r].min(block);
        }
    }
    roots.sort_by_key(|&r| first[r]);
    let mut label_of = vec![0u32; n];
    for (k, &r) in roots.iter().enumerate() {
        label_of[r] = k as u32 + 1;
    }
    let labels = (0..n).map(|i| if img[i] == 0 { 0 } else { label_of[find(&mut parent, i as u32) as usize] }).collect();
    (labels, roots.len())
}

/// cv2.distanceTransform(src, cv2.DIST_L2, 3) (OpenCV's 3 x 3 chamfer in 16.16 fixed point; float32).
pub fn distance_l2_3x3(src: &[u8], w: usize, h: usize) -> Vec<f32> {
    let hv = (0.955f32 * 65536.0).round_ties_even() as u32;
    let diag = (1.3693f32 * 65536.0).round_ties_even() as u32;
    const INIT: u32 = i32::MAX as u32;
    const DIST_MAX: u32 = (i32::MAX >> 2) as u32;
    let scale = 1.0f32 / 65536.0;
    let step = w + 2;
    let mut tmp = vec![0u32; (h + 2) * step];
    tmp[..step].fill(INIT);
    tmp[(h + 1) * step..].fill(INIT);
    for i in 0..h {
        let row = (i + 1) * step + 1;
        tmp[row - 1] = INIT;
        tmp[row + w] = INIT;
        for j in 0..w {
            let k = row + j;
            tmp[k] = if src[i * w + j] == 0 {
                0
            } else {
                let mut t0 = tmp[k - step - 1].wrapping_add(diag);
                t0 = t0.min(tmp[k - step].wrapping_add(hv));
                t0 = t0.min(tmp[k - step + 1].wrapping_add(diag));
                t0.min(tmp[k - 1].wrapping_add(hv))
            };
        }
    }
    let mut dist = vec![0f32; w * h];
    for i in (0..h).rev() {
        let row = (i + 1) * step + 1;
        for j in (0..w).rev() {
            let k = row + j;
            let mut t0 = tmp[k];
            if t0 > hv {
                t0 = t0.min(tmp[k + step + 1].wrapping_add(diag));
                t0 = t0.min(tmp[k + step].wrapping_add(hv));
                t0 = t0.min(tmp[k + step - 1].wrapping_add(diag));
                t0 = t0.min(tmp[k + 1].wrapping_add(hv));
                tmp[k] = t0;
            }
            dist[i * w + j] = (t0.min(DIST_MAX) as f32) * scale;
        }
    }
    dist
}

/// A frontier group: its cells' count, mean (row, col) and whether any cell is in `reach`.
#[derive(Clone, Debug, PartialEq)]
pub struct Frontier {
    pub size: usize,
    pub centroid: (f64, f64),
    pub reachable: bool,
}

/// explorer.py find_frontiers: free cells next to unknown ones (outside the grid counts as unknown),
/// grouped 8-connected in OpenCV's label order; groups of at least `min_cells`.
pub fn find_frontiers(grid: &[i8], w: usize, h: usize, free_max: i64, min_cells: i64, reach: &[bool]) -> Vec<Frontier> {
    let unknown = |r: isize, c: isize| r < 0 || c < 0 || r >= h as isize || c >= w as isize || grid[r as usize * w + c as usize] < 0;
    let mut frontier = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            let g = grid[r * w + c] as i64;
            let (ri, ci) = (r as isize, c as isize);
            if g >= 0
                && g <= free_max
                && (unknown(ri - 1, ci) || unknown(ri + 1, ci) || unknown(ri, ci - 1) || unknown(ri, ci + 1))
            {
                frontier[r * w + c] = 1;
            }
        }
    }
    let (labels, count) = labels8(&frontier, w, h);
    let mut stats = vec![(0usize, 0u64, 0u64, false); count + 1];
    for r in 0..h {
        for c in 0..w {
            let l = labels[r * w + c] as usize;
            if l > 0 {
                let s = &mut stats[l];
                s.0 += 1;
                s.1 += r as u64;
                s.2 += c as u64;
                s.3 |= reach[r * w + c];
            }
        }
    }
    stats
        .into_iter()
        .skip(1)
        .filter(|s| s.0 as i64 >= min_cells)
        .map(|(n, sr, sc, any)| Frontier { size: n, centroid: (sr as f64 / n as f64, sc as f64 / n as f64), reachable: any })
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
pub struct FrontierGoal {
    /// Goal cell (row, col): clear, known space near the frontier.
    pub goal: (usize, usize),
    /// Frontier centroid (row, col).
    pub centroid: (f64, f64),
    pub size: usize,
}

#[derive(Clone, Debug)]
pub struct FrontierParams {
    pub free_max: i64,
    pub pass_max: i64,
    pub clear_max: i64,
    pub min_cells: i64,
    pub standoff_cells: f64,
    pub search_cells: f64,
}

/// explorer.py frontier_goals: reachable frontier groups with a goal cell parked in clear, known
/// space next to each, and the robot's seed cell (None if it isn't next to passable space).
pub fn frontier_goals(grid: &[i8], w: usize, h: usize, robot: (i64, i64), p: &FrontierParams) -> (Vec<FrontierGoal>, Option<(usize, usize)>) {
    let passable: Vec<bool> = grid.iter().map(|&g| g >= 0 && (g as i64) <= p.pass_max).collect();
    let (r, c) = robot;
    let mut seed = None;
    for rad in 0..7i64 {
        let (r0, r1) = (r.saturating_sub(rad).max(0), r.saturating_add(rad + 1).min(h as i64));
        let (c0, c1) = (c.saturating_sub(rad).max(0), c.saturating_add(rad + 1).min(w as i64));
        if r0 >= r1 || c0 >= c1 {
            continue;
        }
        let mut best: Option<(i64, usize, usize)> = None;
        for rr in r0..r1 {
            for cc in c0..c1 {
                if passable[rr as usize * w + cc as usize] {
                    let d = (rr - r) as i128 * (rr - r) as i128 + (cc - c) as i128 * (cc - c) as i128;
                    let d = d.min(i64::MAX as i128) as i64;
                    if best.is_none_or(|b| d < b.0) {
                        best = Some((d, rr as usize, cc as usize));
                    }
                }
            }
        }
        if let Some((_, sr, sc)) = best {
            seed = Some((sr, sc));
            break;
        }
    }
    let Some(seed) = seed else { return (Vec::new(), None) };
    // The seed's 8-connected passable component.
    let mut reach = vec![false; w * h];
    let mut queue = VecDeque::from([seed]);
    reach[seed.0 * w + seed.1] = true;
    while let Some((rr, cc)) = queue.pop_front() {
        for dr in -1i64..=1 {
            for dc in -1i64..=1 {
                let (nr, nc) = (rr as i64 + dr, cc as i64 + dc);
                if nr < 0 || nc < 0 || nr >= h as i64 || nc >= w as i64 {
                    continue;
                }
                let k = nr as usize * w + nc as usize;
                if passable[k] && !reach[k] {
                    reach[k] = true;
                    queue.push_back((nr as usize, nc as usize));
                }
            }
        }
    }
    // Distance (cells) to the nearest unknown cell; outside the grid counts as unknown.
    let (w2, h2) = (w + 2, h + 2);
    let mut known = vec![0u8; w2 * h2];
    for rr in 0..h {
        for cc in 0..w {
            known[(rr + 1) * w2 + cc + 1] = (grid[rr * w + cc] >= 0) as u8;
        }
    }
    let d_unknown = distance_l2_3x3(&known, w2, h2);
    // numpy compares the float32 distances with the threshold cast to float32.
    let standoff = p.standoff_cells as f32;
    let mut clear = Vec::new();
    for rr in 0..h {
        for cc in 0..w {
            let k = rr * w + cc;
            let g = grid[k] as i64;
            if reach[k] && g >= 0 && g <= p.clear_max && d_unknown[(rr + 1) * w2 + cc + 1] >= standoff {
                clear.push((rr, cc));
            }
        }
    }
    let mut goals = Vec::new();
    if clear.is_empty() {
        return (goals, Some(seed));
    }
    for f in find_frontiers(grid, w, h, p.free_max, p.min_cells, &reach) {
        if !f.reachable {
            continue; // only seen through a gap the robot can't drive through
        }
        let (cy, cx) = f.centroid;
        let mut best = (f64::INFINITY, 0usize);
        for (i, &(rr, cc)) in clear.iter().enumerate() {
            let (a, b) = (rr as f64 - cy, cc as f64 - cx);
            let d2 = a * a + b * b;
            if d2 < best.0 || i == 0 {
                best = (d2, i);
            }
        }
        // Python float ** 2 (libm pow, as CPython calls it).
        if best.0 > p.search_cells.powf(std::hint::black_box(2.0)) {
            continue; // no clear spot near it to look from
        }
        goals.push(FrontierGoal { goal: clear[best.1], centroid: (cy, cx), size: f.size });
    }
    (goals, Some(seed))
}

// ---- the node's state machine ----

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Idle,
    Exploring,
    Returning,
    Done,
    Stopped,
}

impl State {
    pub fn as_str(self) -> &'static str {
        match self {
            State::Idle => "idle",
            State::Exploring => "exploring",
            State::Returning => "returning",
            State::Done => "done",
            State::Stopped => "stopped",
        }
    }

    fn driving(self) -> bool {
        matches!(self, State::Exploring | State::Returning)
    }
}

#[derive(Clone, Debug)]
pub struct Params {
    pub map_frame: String,
    pub base_frame: String,
    pub min_cells: i64,
    pub free_max: i64,
    pub pass_max: i64,
    pub clear_max: i64,
    pub standoff: f64,
    pub goal_search: f64,
    pub min_goal_dist: f64,
    pub blacklist_radius: f64,
    pub visited_radius: f64,
    pub goal_timeout: f64,
    pub low_battery: f64,
    pub rotate_speed: f64,
    pub return_home: bool,
    pub home_tolerance: f64,
    pub finish_confirm: f64,
    pub auto_keepout: bool,
    pub bump_marks: bool,
    pub bump_mark_ttl: f64,
    pub bump_mark_margin: f64,
    pub bump_mark_spacing: f64,
    pub bump_mark_width: f64,
    pub bump_light_min: i64,
    pub bumps_per_goal: i64,
    pub retry_failed: bool,
}

impl Params {
    /// Declares the parameters in the Python node's order, with its names, types and defaults.
    pub fn load(p: &mut impl ParamSource) -> anyhow::Result<Self> {
        Ok(Params {
            map_frame: p.string("map_frame", "map")?,
            base_frame: p.string("base_frame", "base_link")?,
            min_cells: p.int("min_frontier_cells", 10)?,
            free_max: p.int("free_cost_max", 40)?,
            pass_max: p.int("passable_cost_max", 98)?,
            clear_max: p.int("goal_cost_max", 15)?,
            standoff: p.float("goal_standoff_m", 0.3)?,
            goal_search: p.float("goal_search_radius_m", 0.8)?,
            min_goal_dist: p.float("min_goal_distance_m", 0.4)?,
            blacklist_radius: p.float("blacklist_radius_m", 0.75)?,
            visited_radius: p.float("visited_radius_m", 0.6)?,
            goal_timeout: p.float("goal_timeout_s", 60.0)?,
            low_battery: p.float("low_battery_fraction", 0.2)?,
            rotate_speed: p.float("rotate_speed_rps", 0.7)?,
            return_home: p.boolean("return_home", true)?,
            home_tolerance: p.float("home_tolerance_m", 0.5)?,
            finish_confirm: p.float("finish_confirm_s", 20.0)?,
            auto_keepout: p.boolean("auto_keepout", true)?,
            bump_marks: p.boolean("bump_marks", true)?,
            bump_mark_ttl: p.float("bump_mark_ttl_s", 600.0)?,
            bump_mark_margin: p.float("bump_mark_margin_m", 0.15)?,
            bump_mark_spacing: p.float("bump_mark_spacing_m", 0.15)?,
            bump_mark_width: p.float("bump_mark_width_m", 0.3)?,
            bump_light_min: p.int("bump_light_signal", 0)?,
            bumps_per_goal: p.int("bumps_per_goal", 2)?,
            retry_failed: p.boolean("retry_failed_before_finish", true)?,
        })
    }
}

/// nav_msgs/OccupancyGrid content the explorer uses.
#[derive(Clone, Debug, Default)]
pub struct Grid {
    pub width: u32,
    pub height: u32,
    pub resolution: f32,
    pub origin_x: f64,
    pub origin_y: f64,
    pub data: Vec<i8>,
}

/// roomba_msgs/SensorState fields the explorer uses.
#[derive(Clone, Debug, Default)]
pub struct Sensors {
    pub bump_left: bool,
    pub bump_right: bool,
    pub wheel_drop_left: bool,
    pub wheel_drop_right: bool,
    /// cliff_left, cliff_front_left, cliff_front_right, cliff_right
    pub cliff: [bool; 4],
    /// light_bump_*_signal in LIGHT_ANGLES order
    pub light: [u16; 6],
}

/// A NavigateToPose goal to send (stamped by the node with its clock).
#[derive(Clone, Debug, PartialEq)]
pub struct GoalRequest {
    pub seq: u64,
    pub frame: String,
    pub x: f64,
    pub y: f64,
    pub qz: f64,
    pub qw: f64,
}

/// The explorer's connection to ROS (or to a replay).
pub trait Io {
    /// time.monotonic()
    fn mono(&mut self) -> f64;
    /// Latest transform `source` -> `target` (translation, quaternion xyzw); None on a TF error.
    fn lookup(&mut self, target: &str, source: &str) -> Option<([f64; 3], [f64; 4])>;
    fn nav_ready(&mut self) -> bool;
    /// Send a goal; answer with [`Explorer::on_goal_response`] and, if accepted, [`Explorer::on_result`].
    fn send_goal(&mut self, goal: GoalRequest);
    /// Cancel the accepted goal `seq`.
    fn cancel_goal(&mut self, seq: u64);
    fn get_params_ready(&mut self) -> bool;
    fn set_params_ready(&mut self) -> bool;
    /// GetParameters [ROTATE_PARAM] on the controller; answer with [`Explorer::on_saved_rotate`].
    fn get_rotate(&mut self);
    /// SetParameters ROTATE_PARAM = value (double) on the controller.
    fn set_rotate(&mut self, value: f64);
    /// ~/status
    fn status(&mut self, json: String);
    /// /perception/ignore_zones/command
    fn zone_command(&mut self, json: String);
    fn log(&mut self, level: Level, msg: String);
    /// time.strftime('%H%M%S') (local time)
    fn local_hms(&mut self) -> String;
    /// A one-shot 0.6 s timer (node clock) calling [`Explorer::cliff_check`].
    fn start_cliff_timer(&mut self);
}

#[derive(Clone, Copy, Debug)]
struct Goal {
    x: f64,
    y: f64,
    sent: f64,
    /// A frontier goal (numpy float64 coordinates in Python, which round() differently).
    numpy: bool,
}

pub struct Explorer {
    p: Params,
    /// Monotonic time each cliff sensor (CLIFF_ANGLES order), then a wheel drop, last fired.
    cliff_seen: [Option<f64>; 5],
    reached_at_retry: i64,
    goal_bumps: i64,
    bumped: bool,
    bump_points: Vec<(f64, f64, f64)>,
    grid: Option<Grid>,
    robot_mode: Option<String>,
    battery: Option<f64>,
    tracking_ok: bool,
    state: State,
    message: String,
    logged_message: Option<String>,
    home: Option<(f64, f64)>,
    goal: Option<Goal>,
    goal_handle: Option<u64>,
    goal_seq: u64,
    blacklist: Vec<(f64, f64)>,
    visited: Vec<(f64, f64)>,
    reached: i64,
    failed: i64,
    frontier_count: i64,
    saved_rotate: Option<f64>,
    sending: bool,
    no_frontier_since: Option<f64>,
    resume_after_home: bool,
    final_check: bool,
}

const WHEEL_DROP: usize = 4;

impl Explorer {
    pub fn new(p: Params, io: &mut impl Io) -> Self {
        let mut e = Explorer {
            p,
            cliff_seen: [None; 5],
            reached_at_retry: -1,
            goal_bumps: 0,
            bumped: false,
            bump_points: Vec::new(),
            grid: None,
            robot_mode: None,
            battery: None,
            tracking_ok: true,
            state: State::Idle,
            message: "Ready".into(),
            logged_message: None,
            home: None,
            goal: None,
            goal_handle: None,
            goal_seq: 0,
            blacklist: Vec::new(),
            visited: Vec::new(),
            reached: 0,
            failed: 0,
            frontier_count: 0,
            saved_rotate: None,
            sending: false,
            no_frontier_since: None,
            resume_after_home: false,
            final_check: false,
        };
        e.publish_status(io);
        e
    }

    pub fn params(&self) -> &Params {
        &self.p
    }

    pub fn state(&self) -> State {
        self.state
    }

    /// Whether /roomba/sensors is needed (auto keep-out or bump marks).
    pub fn wants_sensors(&self) -> bool {
        self.p.auto_keepout || self.p.bump_marks
    }

    // ---- inputs ----

    pub fn on_grid(&mut self, grid: Grid) {
        self.grid = Some(grid);
    }

    pub fn on_mode(&mut self, mode: &str, io: &mut impl Io) {
        let previous = self.robot_mode.replace(mode.to_string());
        if matches!(previous.as_deref(), Some("safe" | "full")) && mode == "passive" && self.p.auto_keepout {
            // Sensors arrive at ~5 Hz: decide once the frames around the stop are in.
            io.start_cliff_timer();
        }
        if self.state.driving() && !matches!(mode, "safe" | "full") {
            self.stop(format!("Stopped: the Roomba switched to {mode} mode (cliff, wheel drop or carpet?)"), io);
        }
    }

    pub fn on_sensors(&mut self, s: &Sensors, io: &mut impl Io) {
        let now = io.mono();
        for (i, &fired) in s.cliff.iter().enumerate() {
            if fired {
                self.cliff_seen[i] = Some(now);
            }
        }
        let wheel_drop = s.wheel_drop_left || s.wheel_drop_right;
        if wheel_drop {
            self.cliff_seen[WHEEL_DROP] = Some(now);
        }
        let light = self.p.bump_light_min > 0
            && (0..6).any(|i| LIGHT_ANGLES[i].abs() < 1.0 && s.light[i] as i64 >= self.p.bump_light_min);
        let bumped = s.bump_left || s.bump_right || light;
        if bumped && !self.bumped && self.p.bump_marks && self.mode_drives() && !wheel_drop {
            let light_min = if self.p.bump_light_min != 0 { self.p.bump_light_min } else { 200 };
            self.mark_bump(bump_bearing(s, light_min), now, io);
            self.bumped_on_goal(io);
        }
        self.bumped = bumped;
    }

    fn mode_drives(&self) -> bool {
        matches!(self.robot_mode.as_deref(), Some("safe" | "full"))
    }

    fn bumped_on_goal(&mut self, io: &mut impl Io) {
        let Some(goal) = self.goal else { return };
        if self.state != State::Exploring || self.p.bumps_per_goal <= 0 {
            return;
        }
        self.goal_bumps += 1;
        if self.goal_bumps >= self.p.bumps_per_goal {
            io.log(
                Level::Warn,
                format!("Bumped {} times on the way to a frontier (railing or glass?): skipping it", self.goal_bumps),
            );
            self.blacklist.push((goal.x, goal.y));
            self.failed += 1;
            self.cancel_goal(io);
            self.message = "Kept bumping into something the camera can't see: trying another area".into();
        }
    }

    fn mark_bump(&mut self, bearing: f64, now: f64, io: &mut impl Io) {
        let Some((x, y, yaw)) = self.pose(io) else { return };
        let points = bump_zone(x, y, yaw, bearing, self.p.bump_mark_width);
        let cx = py_sum(points.iter().map(|p| p[0])) / 4.0;
        let cy = py_sum(points.iter().map(|p| p[1])) / 4.0;
        let ttl = self.p.bump_mark_ttl;
        self.bump_points.retain(|b| now - b.2 < ttl);
        if self.bump_points.iter().any(|&(bx, by, _)| py_hypot(cx - bx, cy - by) < self.p.bump_mark_spacing) {
            return; // already marked here
        }
        self.bump_points.push((cx, cy, now));
        let name = format!("bump_{}_{}", io.local_hms(), self.bump_points.len());
        let mut cmd = zone_add(&name, &points);
        cmd.insert("margin".into(), Value::from(self.p.bump_mark_margin));
        cmd.insert("ttl_s".into(), Value::from(self.p.bump_mark_ttl));
        io.zone_command(py_json(&Value::Object(cmd), None));
        io.log(
            Level::Info,
            format!("Bumper hit at {:+.0} deg: marked {name} for {:.0} min", bearing.to_degrees(), self.p.bump_mark_ttl / 60.0),
        );
    }

    /// The one-shot timer after the Roomba left safe/full mode: if a cliff sensor fired, keep out
    /// of that spot.
    pub fn cliff_check(&mut self, io: &mut impl Io) {
        let now = io.mono();
        let since = |t: Option<f64>| now - t.unwrap_or(-1e9);
        if since(self.cliff_seen[WHEEL_DROP]) < 3.0 {
            return; // lifted or a wheel dropped: every cliff sensor reads a drop, not a place to avoid
        }
        let fired: Vec<usize> = (0..4).filter(|&i| since(self.cliff_seen[i]) < 1.5).collect();
        if fired.is_empty() {
            return; // charger or a mode request
        }
        let names = fired.iter().map(|&i| CLIFF_ANGLES[i].0).collect::<Vec<_>>().join(", ");
        let Some((x, y, yaw)) = self.pose(io) else {
            io.log(Level::Warn, format!("Cliff ({names}) but no robot position: no keep-out zone added"));
            return;
        };
        let bearing = py_sum(fired.iter().map(|&i| CLIFF_ANGLES[i].1.sin()))
            .atan2(py_sum(fired.iter().map(|&i| CLIFF_ANGLES[i].1.cos())));
        let points = cliff_zone(x, y, yaw, bearing);
        let name = format!("cliff_{}", io.local_hms());
        io.zone_command(py_json(&Value::Object(zone_add(&name, &points)), None));
        io.log(Level::Info, format!("Cliff ({names}): added keep-out zone {name}"));
        self.message = format!("Cliff sensor stop: added keep-out zone {name}");
    }

    pub fn on_battery(&mut self, percentage: f32) {
        if !percentage.is_nan() {
            self.battery = Some(percentage as f64);
        }
    }

    pub fn on_filter(&mut self, data: &str) {
        if let Some(v) = py_json_loads(data) {
            // Python: json.loads(...).get('state', 'idle') == 'idle' (a non-object raises: ignored)
            if let Value::Object(m) = v {
                self.tracking_ok = match m.get("state") {
                    None => true,
                    Some(Value::String(s)) => s == "idle",
                    Some(_) => false,
                };
            }
        }
    }

    pub fn on_command(&mut self, data: &str, io: &mut impl Io) {
        let cmd = data.trim_matches(py_space).to_lowercase();
        match cmd.as_str() {
            "start" => self.start(io),
            "stop" => {
                if self.state.driving() {
                    self.stop("Stopped".into(), io);
                }
            }
            "home" => {
                if self.home.is_some() {
                    self.go_home("Returning to the start point".into(), false, io);
                }
            }
            _ => io.log(Level::Warn, format!("Unknown command \"{data}\" (start | stop | home)")),
        }
    }

    // ---- state changes ----

    fn start(&mut self, io: &mut impl Io) {
        if self.state.driving() {
            return;
        }
        if !self.mode_drives() {
            self.set(State::Stopped, "Enable driving (safe mode) first".into(), io);
            return;
        }
        let Some(pose) = self.pose(io) else {
            self.set(State::Stopped, "Robot position unknown (no map -> base TF yet)".into(), io);
            return;
        };
        if !io.nav_ready() {
            self.set(State::Stopped, "Navigation is not running".into(), io);
            return;
        }
        if let Some(b) = self.battery.filter(|&b| b < self.p.low_battery) {
            self.set(State::Stopped, format!("Battery at {:.0}%: charge the Roomba first", b * 100.0), io);
            return;
        }
        if self.home.is_none() || matches!(self.state, State::Done | State::Idle) {
            self.home = Some((pose.0, pose.1));
            self.blacklist.clear();
            self.visited.clear();
            self.reached = 0;
            self.failed = 0;
            self.reached_at_retry = -1;
        }
        self.no_frontier_since = None;
        self.resume_after_home = false;
        self.final_check = false;
        self.slow_turns(true, io);
        self.set(State::Exploring, "Looking for unmapped areas…".into(), io);
        io.log(Level::Info, format!("Exploration started; home at ({:.2}, {:.2})", pose.0, pose.1));
    }

    fn stop(&mut self, why: String, io: &mut impl Io) {
        self.resume_after_home = false;
        self.final_check = false;
        self.cancel_goal(io);
        self.slow_turns(false, io);
        self.set(State::Stopped, why.clone(), io);
        io.log(Level::Info, why);
    }

    fn go_home(&mut self, why: String, resume: bool, io: &mut impl Io) {
        self.resume_after_home = resume;
        let Some(home) = self.home.filter(|_| self.p.return_home) else {
            self.finish("Exploration finished".into(), io);
            return;
        };
        self.cancel_goal(io);
        self.set(State::Returning, why, io);
        self.send_goal(home.0, home.1, None, false, io);
    }

    fn finish(&mut self, why: String, io: &mut impl Io) {
        self.slow_turns(false, io);
        self.set(State::Done, why.clone(), io);
        io.log(Level::Info, why);
    }

    fn set(&mut self, state: State, message: String, io: &mut impl Io) {
        self.state = state;
        self.message = message;
        self.publish_status(io);
    }

    // ---- main loop ----

    /// The 1 s timer.
    pub fn tick(&mut self, io: &mut impl Io) {
        if self.state == State::Exploring {
            let timeout = self.p.goal_timeout;
            if let Some(b) = self.battery.filter(|&b| b < self.p.low_battery) {
                self.go_home(format!("Battery at {:.0}%: returning to the start point", b * 100.0), false, io);
            } else if !self.tracking_ok {
                self.message = "Paused: visual tracking is recovering".into();
            } else if let Some(goal) = self.goal.filter(|g| io.mono() - g.sent > timeout) {
                io.log(Level::Warn, "Frontier goal timed out; skipping it".into());
                self.blacklist.push((goal.x, goal.y));
                self.failed += 1;
                self.cancel_goal(io);
            } else if self.goal.is_none() && !self.sending {
                self.next_frontier(io);
            }
        }
        self.publish_status(io);
    }

    fn next_frontier(&mut self, io: &mut impl Io) {
        if self.grid.is_none() {
            self.message = "Waiting for the costmap…".into();
            return;
        }
        let Some(pose) = self.pose(io) else {
            self.message = "Waiting for the robot position…".into();
            return;
        };
        let grid = self.grid.as_ref().unwrap();
        let (w, h) = (grid.width as usize, grid.height as usize);
        if grid.data.len() != w * h {
            // numpy's reshape raises here (and the Python node dies); wait for a good costmap.
            io.log(Level::Error, format!("Costmap has {} cells, expected {w} x {h}", grid.data.len()));
            return;
        }
        let res = grid.resolution as f64;
        let (ox, oy) = (grid.origin_x, grid.origin_y);
        let robot = (((pose.1 - oy) / res) as i64, ((pose.0 - ox) / res) as i64);
        let fp = FrontierParams {
            free_max: self.p.free_max,
            pass_max: self.p.pass_max,
            clear_max: self.p.clear_max,
            min_cells: self.p.min_cells,
            standoff_cells: self.p.standoff / res,
            search_cells: self.p.goal_search / res,
        };
        let (goals, seed) = frontier_goals(&grid.data, w, h, robot, &fp);
        if seed.is_none() {
            self.message = "Robot is not in free space on the costmap; waiting…".into();
            return;
        }
        let mut best: Option<(f64, f64, f64, f64)> = None;
        let mut candidates = 0;
        for g in &goals {
            let (gy, gx) = (g.goal.0 as f64, g.goal.1 as f64);
            let (cy, cx) = g.centroid;
            let (x, y) = (ox + (gx + 0.5) * res, oy + (gy + 0.5) * res);
            let (fx, fy) = (ox + (cx + 0.5) * res, oy + (cy + 0.5) * res);
            if self.blacklisted(x, y) || self.is_visited(x, y) {
                continue;
            }
            let d = py_hypot(x - pose.0, y - pose.1);
            if d < self.p.min_goal_dist {
                // Already standing here and the camera still can't fill it in: don't come back.
                self.visited.push((x, y));
                continue;
            }
            candidates += 1;
            // Prefer near frontiers, but let bigger ones win over slightly nearer small ones.
            let score = d - 0.01 * g.size as f64;
            if best.is_none_or(|b| score < b.0) {
                best = Some((score, x, y, (fy - y).atan2(fx - x))); // look at the unknown
            }
        }
        self.frontier_count = candidates;
        let Some((_, x, y, yaw)) = best else {
            let now = io.mono();
            let since = *self.no_frontier_since.get_or_insert(now);
            if now - since < self.p.finish_confirm {
                self.message = "No reachable unmapped areas right now; checking again…".into();
                return;
            }
            self.no_frontier_since = None;
            if self.p.retry_failed && !self.blacklist.is_empty() && self.reached != self.reached_at_retry {
                self.reached_at_retry = self.reached;
                io.log(
                    Level::Info,
                    format!(
                        "No unmapped areas left except {} that failed earlier: trying those again",
                        self.blacklist.len()
                    ),
                );
                self.message = "Retrying areas that failed earlier…".into();
                self.blacklist.clear();
                return;
            }
            if self.final_check {
                self.final_check = false;
                let near = if self.near_home(io) { " - back at the start point" } else { "" };
                self.finish(format!("Exploration finished: no reachable unmapped areas left{near}"), io);
                return;
            }
            self.go_home("Exploration finished: no reachable unmapped areas left".into(), true, io);
            return;
        };
        self.no_frontier_since = None;
        self.message = format!("Heading to an unmapped area {:.1} m away", py_hypot(x - pose.0, y - pose.1));
        self.send_goal(x, y, Some(yaw), true, io);
    }

    fn blacklisted(&self, x: f64, y: f64) -> bool {
        self.blacklist.iter().any(|&(bx, by)| py_hypot(x - bx, y - by) < self.p.blacklist_radius)
    }

    fn is_visited(&self, x: f64, y: f64) -> bool {
        self.visited.iter().any(|&(vx, vy)| py_hypot(x - vx, y - vy) < self.p.visited_radius)
    }

    // ---- navigation ----

    fn send_goal(&mut self, x: f64, y: f64, yaw: Option<f64>, numpy: bool, io: &mut impl Io) {
        let yaw = match yaw {
            Some(yaw) => yaw,
            None => self.pose(io).map_or(0.0, |p| p.2),
        };
        self.goal_seq += 1;
        let seq = self.goal_seq;
        self.goal = Some(Goal { x, y, sent: io.mono(), numpy });
        self.goal_bumps = 0;
        self.sending = true;
        io.send_goal(GoalRequest {
            seq,
            frame: self.p.map_frame.clone(),
            x,
            y,
            qz: (yaw / 2.0).sin(),
            qw: (yaw / 2.0).cos(),
        });
    }

    /// Nav2's answer to goal `seq`.
    pub fn on_goal_response(&mut self, seq: u64, accepted: bool, io: &mut impl Io) {
        self.sending = false;
        if seq != self.goal_seq {
            // superseded or cancelled meanwhile
            if accepted {
                io.cancel_goal(seq);
            }
            return;
        }
        if !accepted {
            io.log(Level::Warn, "Navigation rejected the goal".into());
            if let Some(g) = self.goal {
                self.blacklist.push((g.x, g.y));
            }
            self.goal = None;
            return;
        }
        self.goal_handle = Some(seq);
    }

    /// The final status of the accepted goal `seq`.
    pub fn on_result(&mut self, seq: u64, status: i8, io: &mut impl Io) {
        if seq != self.goal_seq {
            return;
        }
        let goal = self.goal.take();
        self.goal_handle = None;
        if self.state == State::Returning {
            let why = self.message.clone(); // why it went home (finished, low battery, request)
            // Nav2 can report success without the robot moving (a one-pose plan while it was
            // boxed in at a wall), so check that it really is back.
            let home = status == STATUS_SUCCEEDED && self.near_home(io);
            if self.resume_after_home {
                // The map looks different from elsewhere: check once more before finishing.
                self.resume_after_home = false;
                self.final_check = true;
                let msg = if home {
                    "Back at the start point; checking for unmapped areas again…"
                } else {
                    "Could not reach the start point; checking for unmapped areas again from here…"
                };
                self.set(State::Exploring, msg.into(), io);
                return;
            }
            let msg = if home {
                format!("{why} - back at the start point")
            } else {
                format!("{why} - could not drive back to the start point")
            };
            self.finish(msg, io);
            return;
        }
        if self.state != State::Exploring {
            return;
        }
        if status == STATUS_SUCCEEDED {
            self.reached += 1;
            if let Some(g) = goal {
                // Looked from here once; if the camera couldn't fill it in, don't come back.
                self.visited.push((g.x, g.y));
            }
        } else if let Some(g) = goal {
            // Unreachable or no progress: don't try this spot again.
            self.failed += 1;
            self.blacklist.push((g.x, g.y));
        }
    }

    fn cancel_goal(&mut self, io: &mut impl Io) {
        self.goal_seq += 1;
        let handle = self.goal_handle.take();
        self.goal = None;
        if let Some(seq) = handle {
            io.cancel_goal(seq);
        }
    }

    /// On shutdown: cancel the goal being driven to, as the Python node does on Ctrl-C.
    pub fn shutdown(&mut self, io: &mut impl Io) {
        self.cancel_goal(io);
    }

    fn near_home(&self, io: &mut impl Io) -> bool {
        match (self.pose(io), self.home) {
            (Some(p), Some(h)) => py_hypot(p.0 - h.0, p.1 - h.1) <= self.p.home_tolerance,
            _ => false,
        }
    }

    fn pose(&self, io: &mut impl Io) -> Option<(f64, f64, f64)> {
        let (t, q) = io.lookup(&self.p.map_frame, &self.p.base_frame)?;
        let [x, y, z, w] = q;
        Some((t[0], t[1], (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z))))
    }

    /// Lower the controller's in-place turn speed while exploring; restore it after.
    fn slow_turns(&mut self, on: bool, io: &mut impl Io) {
        if self.p.rotate_speed <= 0.0 || !io.set_params_ready() {
            return;
        }
        if on {
            if self.saved_rotate.is_none() && io.get_params_ready() {
                io.get_rotate();
            }
            io.set_rotate(self.p.rotate_speed);
        } else if let Some(v) = self.saved_rotate {
            io.set_rotate(v);
        }
    }

    /// GetParameters' answer: the first value's (type, double_value), or the error.
    pub fn on_saved_rotate(&mut self, value: Result<Option<(u8, f64)>, String>, io: &mut impl Io) {
        match value {
            Ok(Some((t, v))) if t == PARAMETER_DOUBLE => self.saved_rotate = Some(v),
            Ok(_) => {}
            Err(e) => io.log(Level::Warn, format!("Could not read {ROTATE_PARAM}: {e}")),
        }
    }

    fn publish_status(&mut self, io: &mut impl Io) {
        if self.logged_message.as_deref() != Some(self.message.as_str()) && self.state.driving() {
            io.log(Level::Info, format!("[{}] {}", self.state.as_str(), self.message));
        }
        self.logged_message = Some(self.message.clone());
        let mut m = Map::new();
        m.insert("state".into(), self.state.as_str().into());
        m.insert("message".into(), self.message.clone().into());
        m.insert("reached".into(), self.reached.into());
        m.insert("failed".into(), self.failed.into());
        m.insert("frontiers".into(), self.frontier_count.into());
        m.insert(
            "home".into(),
            self.home.map_or(Value::Null, |(x, y)| Value::from(vec![py_round(x, 2), py_round(y, 2)])),
        );
        m.insert(
            "goal".into(),
            self.goal.map_or(Value::Null, |g| {
                let r = |v: f64| if g.numpy { np_round2(v) } else { py_round(v, 2) };
                Value::from(vec![r(g.x), r(g.y)])
            }),
        );
        io.status(py_json(&Value::Object(m), None));
    }
}

fn zone_add(name: &str, points: &[[f64; 2]]) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("op".into(), "add".into());
    m.insert("kind".into(), "keepout".into());
    m.insert("name".into(), name.into());
    m.insert("points".into(), Value::from(points.iter().map(|p| Value::from(p.to_vec())).collect::<Vec<_>>()));
    m
}

/// json.loads for status messages: like serde_json, but also accepting Python's NaN / Infinity
/// / -Infinity tokens (json.dumps writes them for non-finite floats).
pub fn py_json_loads(text: &str) -> Option<Value> {
    if let Ok(v) = serde_json::from_str(text) {
        return Some(v);
    }
    let mut out = String::with_capacity(text.len());
    let (mut in_str, mut escaped) = (false, false);
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        if in_str {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_str = false;
            }
            rest = &rest[c.len_utf8()..];
            continue;
        }
        if c == '"' {
            in_str = true;
        }
        let token = ["-Infinity", "Infinity", "NaN"].into_iter().find(|t| rest.starts_with(t));
        if let Some(t) = token {
            out.push_str("null");
            rest = &rest[t.len()..];
        } else {
            out.push(c);
            rest = &rest[c.len_utf8()..];
        }
    }
    serde_json::from_str(&out).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hypot_matches_cpython() {
        // CPython 3.12 math.hypot values where glibc's hypot (f64::hypot) is one bit off.
        assert_eq!(py_hypot(5.164604925736347, 1.8219916586263523), 5.476568053356422);
        assert_eq!(py_hypot(3.6396427326993326, 8.578912024000338), 9.319052029966818);
        assert_eq!(py_hypot(7.212750662325366, 9.292658946181227), 11.763387369585216);
        assert_eq!(py_hypot(3.0, 4.0), 5.0);
        assert_eq!(py_hypot(0.0, -2.5), 2.5);
        assert_eq!(py_hypot(1e-320, 1e-320), 1.414e-320);
        assert!(py_hypot(f64::NAN, 1.0).is_nan());
        assert_eq!(py_hypot(f64::INFINITY, f64::NAN), f64::INFINITY);
    }

    #[test]
    fn python_sum_and_rounding() {
        assert_eq!(py_sum([0.1, 0.2, 0.3]), 0.6); // plain left-to-right sum gives 0.6000000000000001
        assert_eq!(py_sum([-0.0]), 0.0);
        assert_eq!(np_round2(2.675), 2.68); // numpy; Python's round(2.675, 2) is 2.67
        assert_eq!(py_round(2.675, 2), 2.67);
        assert_eq!(format!("{:.0}", 2.5), "2");
        assert_eq!(format!("{:+.0}", -0.2f64.to_degrees() / 100.0), "-0");
    }

    #[test]
    fn distance_transform_chamfer() {
        // 5 x 5 known square, unknown border: centre is 2 steps of 0.955 from the edge.
        let mut img = vec![0u8; 7 * 7];
        for r in 1..6 {
            for c in 1..6 {
                img[r * 7 + c] = 1;
            }
        }
        let d = distance_l2_3x3(&img, 7, 7);
        assert_eq!(d[0], 0.0);
        assert_eq!(d[7 + 1], 62587.0 / 65536.0);
        assert_eq!(d[3 * 7 + 3], 3.0 * 62587.0 / 65536.0);
    }

    #[test]
    fn labels_follow_opencv_block_order() {
        // Component B's first pixel comes after A's in raster order, but B's first 2x2 block is first.
        #[rustfmt::skip]
        let img = [
            0, 0, 1, 0,
            1, 0, 0, 0,
        ];
        let (labels, n) = labels8(&img, 4, 2);
        assert_eq!(n, 2);
        assert_eq!(labels[4], 1);
        assert_eq!(labels[2], 2);
    }

    #[test]
    fn frontier_goal_in_open_room() {
        // 20 x 20 room walled on three sides, unknown from column 13: the frontier is column 12.
        let (w, h) = (20usize, 20usize);
        let mut grid = vec![0i8; w * h];
        for r in 0..h {
            for c in 0..w {
                grid[r * w + c] = if c >= 13 {
                    -1
                } else if r == 0 || r == h - 1 || c == 0 {
                    100
                } else {
                    0
                };
            }
        }
        let p = FrontierParams { free_max: 40, pass_max: 98, clear_max: 15, min_cells: 10, standoff_cells: 3.0, search_cells: 16.0 };
        let (goals, seed) = frontier_goals(&grid, w, h, (10, 3), &p);
        assert_eq!(seed, Some((10, 3)));
        assert_eq!(goals.len(), 1);
        assert_eq!(goals[0].size, 18);
        assert_eq!(goals[0].centroid, (9.5, 12.0));
        // Clear cells end at column 9 (4 chamfer steps from the unknown); rows 9 and 10 tie: first wins.
        assert_eq!(goals[0].goal, (9, 9));
    }

    #[test]
    fn json_with_nan() {
        let v = py_json_loads(r#"{"state": "idle", "rate": NaN, "s": "NaN"}"#).unwrap();
        assert_eq!(v["state"], "idle");
        assert_eq!(v["s"], "NaN");
    }
}
