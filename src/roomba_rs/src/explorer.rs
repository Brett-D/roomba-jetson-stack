//! explorer's logic: frontier exploration on Nav2's global costmap. Same behaviour as
//! roomba_perception/explorer.py (see its docstring): the same frontier groups, goal cells, scores
//! and tie-breaking (OpenCV's connected-component order, numpy's argmin/argwhere order, its
//! float32 distance transform and float rounding), the same state machine (blacklist, visited
//! spots, bump and cliff keep-out marks, retries, return home, finish checks, giving up on Nav2
//! answers that never come), status JSON, ~/markers, log lines and Nav2 / parameter-service
//! requests. Checked by tools/ab_explorer_offline.py.
//!
//! ROS-free: [`Explorer`] reacts to inputs and talks to the outside through [`Io`] (the node in
//! src/bin/explorer.rs, or the replay in examples/explorer_replay.rs).

use std::collections::{HashMap, VecDeque};

use serde_json::{Map, Value};

use crate::watchdog::{Level, ParamSource};
use crate::zones::{py_json, py_round, MarkerSpec, ADD, LINE_STRIP, SPHERE_LIST, TEXT_VIEW_FACING};

pub const ROTATE_PARAM: &str = "FollowPath.rotate_to_heading_angular_vel";
/// rcl_interfaces/ParameterType PARAMETER_DOUBLE.
pub const PARAMETER_DOUBLE: u8 = 3;
/// action_msgs/GoalStatus STATUS_SUCCEEDED.
pub const STATUS_SUCCEEDED: i8 = 4;
const PAUSED: &str = "Paused: visual tracking is recovering";
/// Look-around turns Nav2 may reject in a row (its behaviours still inactive) before it gives up.
const LOOK_REJECTS: i64 = 30;
/// Lattice (m) of the drop-off cells it remembers, in the map frame.
const DROP_CELL_M: f64 = 0.05;
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

/// Frontier cells (1): free cells next to unknown ones; outside the grid counts as unknown.
fn frontier_mask(grid: &[i8], w: usize, h: usize, free_max: i64) -> Vec<u8> {
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
    frontier
}

/// explorer.py find_frontiers' groups as cell lists ((row, col), np.argwhere order), groups of at
/// least `min_cells` in OpenCV's label order.
pub fn frontier_cells(grid: &[i8], w: usize, h: usize, free_max: i64, min_cells: i64) -> Vec<Vec<(usize, usize)>> {
    let (labels, count) = labels8(&frontier_mask(grid, w, h, free_max), w, h);
    let mut groups = vec![Vec::new(); count + 1];
    for r in 0..h {
        for c in 0..w {
            let l = labels[r * w + c] as usize;
            if l > 0 {
                groups[l].push((r, c));
            }
        }
    }
    groups.into_iter().skip(1).filter(|g| g.len() as i64 >= min_cells).collect()
}

/// explorer.py find_frontiers: free cells next to unknown ones (outside the grid counts as unknown),
/// grouped 8-connected in OpenCV's label order; groups of at least `min_cells`.
pub fn find_frontiers(grid: &[i8], w: usize, h: usize, free_max: i64, min_cells: i64, reach: &[bool]) -> Vec<Frontier> {
    let frontier = frontier_mask(grid, w, h, free_max);
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
    /// Centroid (row, col) of the frontier piece it looks at.
    pub centroid: (f64, f64),
    /// Cells in the whole frontier group.
    pub size: usize,
    /// Centroid of the whole frontier group (not in explorer.py's tuples; for the twin's report).
    pub group: (f64, f64),
}

#[derive(Clone, Debug)]
pub struct FrontierParams {
    pub free_max: i64,
    pub pass_max: i64,
    pub clear_max: i64,
    pub min_cells: i64,
    pub standoff_cells: f64,
    pub search_cells: f64,
    /// Fallback goal spots (no clear spot near a frontier): cells per cost unit; 0 = none.
    pub cost_weight: f64,
    /// Spots to look at unreachable frontiers from, within this; 0 = none.
    pub peek_cells: f64,
    /// Longer frontier groups are cut into pieces about this long, a goal spot each; 0 = never.
    pub split_cells: f64,
}

/// explorer.py frontier_pieces: a frontier group's cells (np.argwhere order) cut into
/// ceil(extent / split) pieces of equal count along its longer side.
pub fn frontier_pieces(cells: &[(usize, usize)], split: f64) -> Vec<Vec<(usize, usize)>> {
    let n = cells.len();
    if n == 0 {
        return Vec::new();
    }
    let (mut r0, mut r1, mut c0, mut c1) = (usize::MAX, 0usize, usize::MAX, 0usize);
    for &(r, c) in cells {
        (r0, r1, c0, c1) = (r0.min(r), r1.max(r), c0.min(c), c1.max(c));
    }
    let (span_r, span_c) = (r1 - r0, c1 - c0);
    let ext = span_r.max(span_c) + 1;
    if split <= 0.0 || ext as f64 <= split {
        return vec![cells.to_vec()];
    }
    let k = ((ext as f64 / split).ceil() as usize).min(n);
    let mut sorted = cells.to_vec();
    if span_r > span_c {
        sorted.sort_unstable();
    } else {
        sorted.sort_unstable_by_key(|&(r, c)| (c, r));
    }
    (0..k).map(|j| sorted[j * n / k..(j + 1) * n / k].to_vec()).collect()
}

/// Index of the cell nearest to (cy, cx) (numpy argmin: first of equals), None beyond `limit`.
fn nearest(cells: &[(usize, usize)], cy: f64, cx: f64, limit: f64) -> Option<usize> {
    let mut best = (f64::INFINITY, 0usize);
    for (i, &(rr, cc)) in cells.iter().enumerate() {
        let (a, b) = (rr as f64 - cy, cc as f64 - cx);
        let d2 = a * a + b * b;
        if d2 < best.0 || i == 0 {
            best = (d2, i);
        }
    }
    // Python float ** 2 (libm pow, as CPython calls it).
    (!cells.is_empty() && best.0 <= limit.powf(std::hint::black_box(2.0))).then_some(best.1)
}

/// Index of the cell with the lowest distance (cells) to (cy, cx) + its penalty within `limit`.
fn best_spot(cells: &[(usize, usize)], penalty: &[f64], cy: f64, cx: f64, limit: f64) -> Option<usize> {
    let l2 = limit.powf(std::hint::black_box(2.0));
    let mut best: Option<(f64, usize)> = None;
    for (i, &(rr, cc)) in cells.iter().enumerate() {
        let (a, b) = (rr as f64 - cy, cc as f64 - cx);
        let d2 = a * a + b * b;
        if d2 <= l2 {
            let score = d2.sqrt() + penalty[i];
            if best.is_none_or(|b| score < b.0) {
                best = Some((score, i));
            }
        }
    }
    best.map(|b| b.1)
}

/// `grid` with the known cells within `clear_cells` of a drop-off cell (map-frame lattice
/// DROP_CELL_M) made lethal (explorer.py drop_blocked): no frontier, goal or way through there.
pub fn block_drops(grid: &mut [i8], w: usize, h: usize, res: f64, ox: f64, oy: f64, drops: &[(i64, i64)], clear_cells: f64) {
    let k = clear_cells as i64;
    let k2 = clear_cells * clear_cells;
    for &(i, j) in drops {
        let c = (((i as f64 + 0.5) * DROP_CELL_M - ox) / res).floor() as i64;
        let r = (((j as f64 + 0.5) * DROP_CELL_M - oy) / res).floor() as i64;
        for dr in -k..=k {
            for dc in -k..=k {
                let (rr, cc) = (r + dr, c + dc);
                if rr < 0 || cc < 0 || rr >= h as i64 || cc >= w as i64 || ((dr * dr + dc * dc) as f64) > k2 {
                    continue;
                }
                let g = &mut grid[rr as usize * w + cc as usize];
                if *g >= 0 {
                    *g = 100;
                }
            }
        }
    }
}

/// The `passable` cells connected to `seed` (explorer.py reachable). 8-connected, except that a
/// diagonal step between two lethal cells (keep-out marks, walls) is no way through: Nav2's planner
/// (navfn) only spreads to the 4 neighbours, and in Isaac run 9 a ring of bump marks with diagonal
/// gaps trapped the robot while the explorer saw a way out. Past inscribed corners it still steps
/// diagonally (that inflation changes as the robot gets closer and sees better).
pub fn reachable(grid: &[i8], passable: &[bool], w: usize, h: usize, seed: (usize, usize)) -> Vec<bool> {
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
                if !passable[k] || reach[k] {
                    continue;
                }
                if dr != 0 && dc != 0 && grid[rr * w + nc as usize] >= 100 && grid[nr as usize * w + cc] >= 100 {
                    continue;
                }
                reach[k] = true;
                queue.push_back((nr as usize, nc as usize));
            }
        }
    }
    reach
}

/// explorer.py frontier_goals: reachable frontier groups (long ones in pieces) with a goal cell
/// parked in clear, known space next to each (or, in narrow places, the best reachable cell below
/// inscribed), the robot's seed cell (None if it isn't next to passable space), spots to look at
/// the frontier groups it can't reach from, and how many cells it can reach.
pub fn frontier_goals(
    grid: &[i8],
    w: usize,
    h: usize,
    robot: (i64, i64),
    p: &FrontierParams,
) -> (Vec<FrontierGoal>, Option<(usize, usize)>, Vec<FrontierGoal>, usize) {
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
    let Some(seed) = seed else { return (Vec::new(), None, Vec::new(), 0) };
    let reach = reachable(grid, &passable, w, h, seed);
    let reach_cells = reach.iter().filter(|&&r| r).count();
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
    let half_standoff = (p.standoff_cells / 2.0) as f32;
    let mut clear = Vec::new();
    let (mut spots, mut penalty) = (Vec::new(), Vec::new());
    for rr in 0..h {
        for cc in 0..w {
            let k = rr * w + cc;
            let g = grid[k] as i64;
            if !reach[k] || g < 0 {
                continue;
            }
            let d = d_unknown[(rr + 1) * w2 + cc + 1];
            if g <= p.clear_max && d >= standoff {
                clear.push((rr, cc));
            }
            if p.cost_weight > 0.0 && g <= p.pass_max && d >= half_standoff {
                spots.push((rr, cc));
                penalty.push(g as f64 * p.cost_weight);
            }
        }
    }
    let (mut goals, mut peeks) = (Vec::new(), Vec::new());
    let mean = |cells: &[(usize, usize)]| {
        let (sr, sc) = cells.iter().fold((0u64, 0u64), |a, &(r, c)| (a.0 + r as u64, a.1 + c as u64));
        (sr as f64 / cells.len() as f64, sc as f64 / cells.len() as f64)
    };
    for cells in frontier_cells(grid, w, h, p.free_max, p.min_cells) {
        let group = mean(&cells);
        // A long frontier (a room's far side seen from the doorway) gets a spot per piece: one
        // spot failing (a railing the camera misses in the way) doesn't drop all of it.
        for piece in frontier_pieces(&cells, p.split_cells) {
            let (out, limit) = if piece.iter().any(|&(r, c)| reach[r * w + c]) {
                (&mut goals, p.search_cells)
            } else if p.peek_cells > 0.0 {
                (&mut peeks, p.peek_cells) // only seen through a gap the robot can't drive through
            } else {
                continue;
            };
            let (cy, cx) = mean(&piece);
            let spot =
                nearest(&clear, cy, cx, limit).map(|i| clear[i]).or_else(|| best_spot(&spots, &penalty, cy, cx, limit).map(|i| spots[i]));
            if let Some(goal) = spot {
                out.push(FrontierGoal { goal, centroid: (cy, cx), size: cells.len(), group });
            } // else: no spot near it to look from
        }
    }
    (goals, Some(seed), peeks, reach_cells)
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
    pub cost_weight: f64,
    pub peek_search: f64,
    pub frontier_split: f64,
    pub drop_clear: f64,
    pub drop_min_hits: i64,
    pub min_goal_dist: f64,
    pub blacklist_radius: f64,
    pub visited_radius: f64,
    pub goal_timeout: f64,
    pub response_timeout: f64,
    pub return_timeout: f64,
    pub low_battery: f64,
    pub rotate_speed: f64,
    pub return_home: bool,
    pub home_tolerance: f64,
    pub home_attempts: i64,
    pub finish_confirm: f64,
    pub look_steps: i64,
    /// Radians.
    pub look_step: f64,
    pub look_timeout: f64,
    pub look_pause: f64,
    pub face_turn_min: f64,
    pub reopen_cells: i64,
    pub auto_keepout: bool,
    pub bump_marks: bool,
    pub bump_mark_ttl: f64,
    pub bump_mark_short_ttl: f64,
    pub bump_mark_margin: f64,
    pub bump_mark_confirmed_margin: f64,
    pub bump_mark_spacing: f64,
    pub bump_mark_width: f64,
    pub mark_clear: f64,
    pub mark_soft: f64,
    pub bump_light_min: i64,
    pub bumps_per_goal: i64,
    pub person_bump_dist: f64,
    pub person_bump_window: f64,
    pub trap_area: f64,
    pub trap_radius: f64,
    pub trap_wait: f64,
    pub retry_failed: bool,
    /// "rule" (default: the score below) or "learned" (a [`GoalRanker`] re-orders the same
    /// candidates; needs the `learned` build feature and `goal_ranking_model`).
    pub goal_ranking: String,
    pub goal_ranking_model: String,
    /// Learned unstuck skill (docs/learned-skills.md; off by default): when the robot is stuck on a
    /// frontier goal, call the `unstuck_skill` action, then try that goal once more; when the skill
    /// is off, busy, missing or fails, the explorer does what it does without it.
    pub unstuck_skill: bool,
    pub unstuck_time_limit: f64,
    pub unstuck_call_timeout: f64,
    pub unstuck_no_progress: f64,
    pub unstuck_clearance: f64,
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
            // No clear spot near a frontier (narrow passage): the reachable cell below inscribed
            // with the lowest distance + cost x this (cells per cost unit; 0 = skip it).
            cost_weight: p.float("goal_cost_weight", 0.05)?,
            // Frontiers it can't reach get one look from the nearest reachable spot within this.
            peek_search: p.float("peek_search_m", 3.0)?,
            // Frontier groups longer than this get a goal spot per piece of about this length.
            frontier_split: p.float("frontier_split_m", 2.0)?,
            // Drop-offs floor_obstacles reported (/perception/drops, e.g. a stair opening the camera
            // looks down into while the map shows floor) are remembered; frontier and goal cells
            // within this of one are left out, so it neither aims into the void nor reaches
            // through it (0 = ignore drops). A cell counts after drop points in this many clouds.
            drop_clear: p.float("drop_clear_m", 0.3)?,
            drop_min_hits: p.int("drop_min_hits", 2)?,
            min_goal_dist: p.float("min_goal_distance_m", 0.4)?,
            blacklist_radius: p.float("blacklist_radius_m", 0.75)?,
            visited_radius: p.float("visited_radius_m", 0.6)?,
            goal_timeout: p.float("goal_timeout_s", 60.0)?,
            // A goal request Nav2 never answers counts as rejected after this long.
            response_timeout: p.float("goal_response_timeout_s", 10.0)?,
            // A trip home with no result counts as failed after this long (0 = wait forever).
            return_timeout: p.float("return_timeout_s", 300.0)?,
            low_battery: p.float("low_battery_fraction", 0.2)?,
            rotate_speed: p.float("rotate_speed_rps", 0.7)?,
            return_home: p.boolean("return_home", true)?,
            home_tolerance: p.float("home_tolerance_m", 0.5)?,
            // Trips home when nothing is left: a failed one (a TF hiccup, a bump mark in the way) is
            // tried again, up to this many in all, before it says it could not drive back.
            home_attempts: p.int("return_home_attempts", 3)?.max(1),
            finish_confirm: p.float("finish_confirm_s", 20.0)?,
            // Look around (Nav2 spin, in steps) at the first spot reached (not on the dock: the
            // light bumpers see it while turning) and before deciding nothing is left (0 = never).
            look_steps: p.int("look_around_steps", 4)?,
            look_step: p.float("look_around_step_deg", 90.0)?.to_radians(),
            look_timeout: p.float("look_around_timeout_s", 20.0)?,
            // Pause after each look-around turn (s): visual tracking catches up while it stands.
            look_pause: p.float("look_around_pause_s", 0.0)?,
            // Nav2's goal checker ignores the final heading here (yaw_goal_tolerance 6.28), so at a
            // spot it reached it turns to face the unknown it came to see when it is off by more
            // than this (0 = never).
            face_turn_min: p.float("face_frontier_min_deg", 45.0)?.to_radians(),
            // Failed spots are skipped until the map around them opens up by this many cells (a
            // door or a person: 100+; one flickering railing post with its inflation: ~37).
            reopen_cells: p.int("blacklist_reopen_cells", 40)?,
            auto_keepout: p.boolean("auto_keepout", true)?,
            bump_marks: p.boolean("bump_marks", true)?,
            bump_mark_ttl: p.float("bump_mark_ttl_s", 600.0)?,
            // A mark not confirmed by another bump at the same spot (within bump_mark_spacing_m)
            // lasts only this long: people walking into the robot leave single hits (0 = all
            // bump_mark_ttl_s).
            bump_mark_short_ttl: p.float("bump_mark_short_ttl_s", 60.0)?,
            bump_mark_margin: p.float("bump_mark_margin_m", 0.15)?,
            // A confirmed mark (bumped twice at the same spot: a railing, glass) keeps the robot's
            // centre this far off: radius 0.17 + 0.1 m clearance (+ 1 cm) - keep-out isn't inflated.
            bump_mark_confirmed_margin: p.float("bump_mark_confirmed_margin_m", 0.28)?,
            bump_mark_spacing: p.float("bump_mark_spacing_m", 0.15)?,
            bump_mark_width: p.float("bump_mark_width_m", 0.3)?,
            // Bump and cliff marks leave the robot's footprint (plus clearance) free: footprint
            // 0.18 m + a mask/costmap cell diagonal (the grids don't line up) + slack ...
            mark_clear: p.float("mark_clear_radius_m", 0.28)?,
            // ... and carry an "avoid" band this wide beyond their margin (keep-out cells aren't
            // inflated: paths along their edge made the controller see "collision ahead").
            mark_soft: p.float("mark_soft_band_m", 0.22)?,
            bump_light_min: p.int("bump_light_signal", 0)?,
            bumps_per_goal: p.int("bumps_per_goal", 2)?,
            // A bump while a person was detected in front within this distance (in the last
            // person_bump_window_s) is a person's leg: not marked, not counted (0 = mark all).
            person_bump_dist: p.float("person_bump_dist_m", 1.2)?,
            person_bump_window: p.float("person_bump_window_s", 3.0)?,
            // Boxed in by its own bump marks (Nav2 plans 4-connected, so a ring of marks with only
            // diagonal gaps closes it in): when the area it can reach is below this while the map
            // has more free space elsewhere and bump marks lie within trap_radius_m, it removes
            // them one at a time (unconfirmed first, rows along a railing last), trap_wait_s apart
            // (0 = never).
            trap_area: p.float("trap_area_m2", 1.0)?,
            trap_radius: p.float("trap_radius_m", 1.5)?,
            trap_wait: p.float("trap_wait_s", 3.0)?,
            retry_failed: p.boolean("retry_failed_before_finish", true)?,
            goal_ranking: p.string("goal_ranking", "rule")?,
            goal_ranking_model: p.string("goal_ranking_model", "")?,
            unstuck_skill: p.boolean("unstuck_skill", false)?,
            unstuck_time_limit: p.float("unstuck_time_limit_s", 15.0)?,
            // the caller's own limit: a lost answer counts as a failure
            unstuck_call_timeout: p.float("unstuck_call_timeout_s", 25.0)?,
            // stuck = an accepted goal and the robot moved < 0.2 m in this long (0 = off)
            unstuck_no_progress: p.float("unstuck_no_progress_s", 30.0)?,
            // a goal Nav2 gave up on counts as stuck when a lethal cell is this close to the robot centre
            unstuck_clearance: p.float("unstuck_clearance_m", 0.30)?,
        })
    }
}

/// A frontier goal that passed every explorer rule, offered to a [`GoalRanker`] (map frame).
#[derive(Clone, Debug)]
pub struct RankCandidate {
    pub x: f64,
    pub y: f64,
    pub yaw: f64,
    pub frontier: (f64, f64),
    pub size_cells: usize,
    pub score: f64,
}

/// What a ranker sees besides the candidates.
pub struct RankInput<'a> {
    pub grid: &'a Grid,
    pub pose: (f64, f64, f64),
    pub home: Option<(f64, f64)>,
    pub battery: Option<f64>,
    pub elapsed_s: f64,
    pub failed: i64,
    pub blacklist: &'a [(f64, f64)],
    pub visited: &'a [(f64, f64)],
}

/// Optional learned goal ranking (`goal_ranking: learned`): picks one of the candidates the rules
/// produced (in rule order, best first). None, or an error, keeps the rule's choice.
pub trait GoalRanker: Send + Sync {
    fn choose(&mut self, input: &RankInput, candidates: &[RankCandidate]) -> anyhow::Result<Option<usize>>;
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
    /// The node clock in seconds: ROS time (sim time under use_sim_time), so timeouts about the
    /// robot's progress don't fire early when a simulation runs slower than real time.
    fn now(&mut self) -> f64;
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
    /// Whether the spin action server is available.
    fn spin_ready(&mut self) -> bool;
    /// Send a Spin goal (target_yaw, time_allowance); answer with [`Explorer::on_spin_response`]
    /// and, if accepted, [`Explorer::on_spin_result`].
    fn send_spin(&mut self, seq: u64, target_yaw: f32, sec: i32, nanosec: u32);
    /// Cancel the accepted Spin goal `seq`.
    fn cancel_spin(&mut self, seq: u64);
    /// Whether anyone listens to ~/markers (they are only built then).
    fn markers_wanted(&mut self) -> bool;
    /// ~/markers: a DELETEALL then `markers`, in `frame`, stamped by the node.
    fn markers(&mut self, frame: &str, markers: &[MarkerSpec]);
    /// ~/drops: every drop-off cell the explorer remembers (centres, in `frame`), about 1 Hz, for
    /// the global costmap's drop_layer (marked again right after a costmap clear).
    fn drops(&mut self, frame: &str, points: &[[f32; 3]]);
    /// Whether the learned unstuck skill's action server is up (`unstuck_skill: true`; only the Rust
    /// node built with the `learned` feature has a client).
    fn unstuck_ready(&mut self) -> bool {
        false
    }
    /// Call the unstuck skill (the goal it was driving to, map frame; time limit); answer with
    /// [`Explorer::on_unstuck_response`] and, if accepted, [`Explorer::on_unstuck_result`].
    fn send_unstuck(&mut self, _seq: u64, _goal: Option<(f64, f64)>, _time_limit: f64) {}
    /// Cancel the accepted unstuck call `seq`.
    fn cancel_unstuck(&mut self, _seq: u64) {}
}

/// What the last frontier search saw, for ~/markers.
#[derive(Clone, Debug)]
struct Viz {
    res: f64,
    frontiers: Vec<Vec<[f64; 3]>>,
    /// Spots considered: (x, y, colour, label).
    options: Vec<(f64, f64, [f32; 4], String)>,
    /// Spots to look at frontiers it can't reach from.
    peeks: Vec<(f64, f64)>,
}

/// A keep-out mark it sent: kind ("bump" | "cliff"), name, points, monotonic time, ttl s (0 = none).
#[derive(Clone, Debug)]
struct SentMark {
    bump: bool,
    name: String,
    points: Vec<[f64; 2]>,
    sent: f64,
    ttl: f64,
    /// The add command (bump marks; sent again with the long TTL once confirmed).
    cmd: Option<Map<String, Value>>,
    /// Bumped again at the same spot.
    confirmed: bool,
}

// ~/markers colours (exact in f32), as in explorer.py.
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
const BLUE: [f32; 4] = [0.0, 0.5, 1.0, 1.0];
const GREY: [f32; 4] = [0.5, 0.5, 0.5, 0.75];
const PURPLE: [f32; 4] = [0.75, 0.0, 1.0, 1.0];
const ORANGE: [f32; 4] = [1.0, 0.5, 0.0, 0.75];
const RED_AREA: [f32; 4] = [1.0, 0.0, 0.0, 0.25];
const CYAN: [f32; 4] = [0.0, 1.0, 1.0, 0.75];
const YELLOW: [f32; 4] = [1.0, 1.0, 0.0, 1.0];
const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const MAGENTA: [f32; 4] = [1.0, 0.0, 1.0, 0.75];
// visualization_msgs/Marker types.
const ARROW: i32 = 0;
const SPHERE: i32 = 2;
const CYLINDER: i32 = 3;
const CUBE_LIST: i32 = 6;

fn spec(ns: &str, id: usize, kind: i32, position: [f64; 3], scale: [f64; 3], color: [f32; 4], points: Vec<[f64; 3]>, text: String) -> MarkerSpec {
    MarkerSpec { ns: ns.into(), id: id as i32, kind, action: ADD, position, orientation_w: 1.0, scale, color, points, text }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GoalKind {
    Frontier,
    /// A closer look at a frontier it can't reach.
    Peek,
    Home,
}

/// A call of the learned unstuck skill in progress.
#[derive(Clone, Copy, Debug)]
struct UnstuckCall {
    seq: u64,
    sent: f64,
    accepted: bool,
    /// The goal to try again afterwards.
    retry: Option<Goal>,
    /// On failure send that goal again too (Nav2 was still working on it), else skip it.
    give_back: bool,
}

#[derive(Clone, Copy, Debug)]
struct Goal {
    x: f64,
    y: f64,
    sent: f64,
    /// A frontier goal (numpy float64 coordinates in Python, which round() differently).
    numpy: bool,
    kind: GoalKind,
    /// The heading asked for (Nav2's goal checker here ignores it).
    yaw: f64,
}

/// A spot where a goal failed: skipped until the map around it opens up.
#[derive(Clone, Debug)]
struct Failed {
    x: f64,
    y: f64,
    /// Passable cells around it when it failed ((2k + 1)^2, row-major), if there was a costmap.
    patch: Option<Vec<bool>>,
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
    blacklist: Vec<Failed>,
    visited: Vec<(f64, f64)>,
    peeked: Vec<(f64, f64)>,
    reached: i64,
    failed: i64,
    frontier_count: i64,
    saved_rotate: Option<f64>,
    /// (seq, sent monotonic) of the goal request Nav2 hasn't answered yet.
    sending: Option<(u64, f64)>,
    no_frontier_since: Option<f64>,
    resume_after_home: bool,
    /// The last trip home failed and no frontier was reached since.
    home_failures: i64,
    /// (steps left, stop as soon as a frontier shows up) while looking around.
    look: Option<(i64, bool)>,
    looked_here: bool,
    /// Look around once the first spot is reached.
    look_pending: bool,
    /// Turn (rad) to face the unknown at the spot just reached.
    turn: Option<f64>,
    /// Look-around turns rejected in a row.
    look_rejects: i64,
    /// What it said when it sent the goal (shown again after a pause).
    goal_message: String,
    /// (seq, sent monotonic, accepted) of the look-around turn in progress.
    spin: Option<(u64, f64, bool)>,
    spin_handle: Option<u64>,
    spin_seq: u64,
    // ~/markers: the last frontier search, where the goal was chosen from, the marks sent and the
    // markers last published.
    viz: Option<Viz>,
    goal_from: Option<(f64, f64)>,
    marks_sent: Vec<SentMark>,
    markers_last: Option<Vec<MarkerSpec>>,
    ranker: Option<Box<dyn GoalRanker>>,
    started: f64,
    // unstuck_skill: the call in progress, the goal it was called for (once per goal), the
    // no-progress watch (x, y, since) of the goal being driven to.
    unstuck: Option<UnstuckCall>,
    unstuck_seq: u64,
    unstuck_goal: Option<(f64, f64)>,
    progress: Option<(f64, f64, f64)>,
    unstuck_missing_logged: bool,
    /// People the person detector saw lately: (time, x, y) in the map frame.
    persons: Vec<(f64, f64, f64)>,
    /// Drop-off cells (DROP_CELL_M lattice in the map frame) -> clouds that had drop points there.
    drop_hits: HashMap<(i64, i64), i64>,
    /// No trap-breaking mark removal before this time.
    trap_next: f64,
    /// Next look-around turn not before this time (look_around_pause_s).
    look_until: f64,
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
            peeked: Vec::new(),
            reached: 0,
            failed: 0,
            frontier_count: 0,
            saved_rotate: None,
            sending: None,
            no_frontier_since: None,
            resume_after_home: false,
            home_failures: 0,
            look: None,
            looked_here: false,
            look_pending: false,
            turn: None,
            look_rejects: 0,
            goal_message: String::new(),
            spin: None,
            spin_handle: None,
            spin_seq: 0,
            viz: None,
            goal_from: None,
            marks_sent: Vec::new(),
            markers_last: None,
            ranker: None,
            started: 0.0,
            unstuck: None,
            unstuck_seq: 0,
            unstuck_goal: None,
            progress: None,
            unstuck_missing_logged: false,
            persons: Vec::new(),
            drop_hits: HashMap::new(),
            trap_next: f64::NEG_INFINITY,
            look_until: f64::NEG_INFINITY,
        };
        e.publish_status(io);
        e
    }

    /// Learned goal ranking (only used with `goal_ranking: learned`).
    pub fn set_ranker(&mut self, ranker: Box<dyn GoalRanker>) {
        self.ranker = Some(ranker);
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
        let now = io.now();
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
            let bearing = bump_bearing(s, light_min);
            if self.person_in_front(now, io) {
                io.log(Level::Info, format!("Bumper hit at {:+.0} deg with a person in front: not marked", bearing.to_degrees()));
            } else {
                self.mark_bump(bearing, now, io);
                self.bumped_on_goal(io);
            }
        }
        self.bumped = bumped;
    }

    /// People the person detector reports (/perception/person_detector/detections_3d): their
    /// positions in `frame`, kept person_bump_window_s in the map frame.
    pub fn on_persons(&mut self, frame: &str, points: &[[f64; 3]], io: &mut impl Io) {
        if self.p.person_bump_dist <= 0.0 || points.is_empty() {
            return;
        }
        let Some((t, q)) = io.lookup(&self.p.map_frame, frame) else { return };
        let now = io.now();
        let r = rotation(q);
        for p in points {
            let (x, y) = to_frame(&r, t, p[0], p[1], p[2]);
            self.persons.push((now, x, y));
        }
        let window = self.p.person_bump_window;
        self.persons.retain(|p| now - p.0 <= window);
        if self.persons.len() > 100 {
            self.persons.drain(..self.persons.len() - 100);
        }
    }

    /// Whether someone was seen in front of the robot within person_bump_dist_m lately.
    fn person_in_front(&mut self, now: f64, io: &mut impl Io) -> bool {
        if self.p.person_bump_dist <= 0.0 {
            return false;
        }
        let window = self.p.person_bump_window;
        self.persons.retain(|p| now - p.0 <= window);
        if self.persons.is_empty() {
            return false;
        }
        let Some((x, y, yaw)) = self.pose(io) else { return false };
        let (c, s) = (yaw.cos(), yaw.sin());
        self.persons
            .iter()
            .any(|&(_, px, py)| py_hypot(px - x, py - y) <= self.p.person_bump_dist && (px - x) * c + (py - y) * s > 0.0)
    }

    /// Drop points floor_obstacles reports (/perception/drops), in `frame`: remembered as drop-off
    /// cells (a cell counts once per cloud).
    pub fn on_drops(&mut self, frame: &str, points: &[[f32; 3]], io: &mut impl Io) {
        if self.p.drop_clear <= 0.0 || points.is_empty() {
            return;
        }
        let Some((t, q)) = io.lookup(&self.p.map_frame, frame) else { return };
        let r = rotation(q);
        let mut cells = std::collections::BTreeSet::new();
        for p in points {
            let (x, y) = to_frame(&r, t, p[0] as f64, p[1] as f64, p[2] as f64);
            if x.is_finite() && y.is_finite() {
                cells.insert(((x / DROP_CELL_M).floor() as i64, (y / DROP_CELL_M).floor() as i64));
            }
        }
        for c in cells {
            *self.drop_hits.entry(c).or_insert(0) += 1;
        }
    }

    /// The drop-off cells (DROP_CELL_M lattice), sorted.
    fn drop_cells(&self) -> Vec<(i64, i64)> {
        let mut out: Vec<(i64, i64)> = self.drop_hits.iter().filter(|&(_, &n)| n >= self.p.drop_min_hits).map(|(&c, _)| c).collect();
        out.sort_unstable();
        out
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
            self.blacklist(goal.x, goal.y, false);
            self.failed += 1;
            self.cancel_goal(io);
            self.message = "Kept bumping into something the camera can't see: trying another area".into();
            self.try_unstuck(None, false, "kept bumping", io);
        }
    }

    /// Keep-out command: leave the robot's footprint free (it is standing there), and keep planned
    /// paths a robot radius away from the mark (a soft band beyond its margin).
    fn clear_robot(&self, cmd: &mut Map<String, Value>, x: f64, y: f64) {
        if self.p.mark_clear > 0.0 {
            cmd.insert("clear".into(), Value::from(vec![py_round(x, 3), py_round(y, 3), self.p.mark_clear]));
        }
        if self.p.mark_soft > 0.0 {
            cmd.insert("soft".into(), Value::from(self.p.mark_soft));
        }
    }

    fn mark_bump(&mut self, bearing: f64, now: f64, io: &mut impl Io) {
        let Some((x, y, yaw)) = self.pose(io) else { return };
        let points = bump_zone(x, y, yaw, bearing, self.p.bump_mark_width);
        let cx = py_sum(points.iter().map(|p| p[0])) / 4.0;
        let cy = py_sum(points.iter().map(|p| p[1])) / 4.0;
        // A bump at the same spot again (within bump_mark_spacing_m of a mark's centre) confirms
        // the mark there: something the camera can't see (railing, glass), not someone walking by.
        // Hits around the robot at one spot (people's legs as it turns) are different spots.
        let active = self.bump_marks(now, false);
        let near = |m: &SentMark, d: f64| {
            let (mx, my) = mark_centre(&m.points);
            py_hypot(cx - mx, cy - my) < d
        };
        let same: Vec<usize> = active.iter().copied().filter(|&i| near(&self.marks_sent[i], self.p.bump_mark_spacing)).collect();
        for &i in &same {
            self.confirm_mark(i, now, (x, y), io);
        }
        let ttl = self.p.bump_mark_ttl;
        self.bump_points.retain(|b| now - b.2 < ttl);
        if !same.is_empty() {
            return; // already marked here
        }
        let confirmed = self.p.bump_mark_short_ttl <= 0.0;
        let ttl = if confirmed { self.p.bump_mark_ttl } else { self.p.bump_mark_short_ttl };
        self.bump_points.push((cx, cy, now));
        let name = format!("bump_{}_{}", io.local_hms(), self.bump_points.len());
        let mut cmd = zone_add(&name, &points);
        cmd.insert("margin".into(), Value::from(self.p.bump_mark_margin));
        cmd.insert("ttl_s".into(), Value::from(ttl));
        self.clear_robot(&mut cmd, x, y);
        io.zone_command(py_json(&Value::Object(cmd.clone()), None));
        self.sent_mark(true, &name, points, now, ttl, Some(cmd), confirmed);
        io.log(Level::Info, format!("Bumper hit at {:+.0} deg: marked {name} for {}", bearing.to_degrees(), duration_text(ttl)));
    }

    /// Bumped again at mark `i` (index in marks_sent): something solid (a railing): make it last
    /// bump_mark_ttl_s from now and keep planned paths bump_mark_confirmed_margin_m from it (the
    /// robot's free circle moves to where it stands now, `at`).
    fn confirm_mark(&mut self, i: usize, now: f64, at: (f64, f64), io: &mut impl Io) {
        let m = &mut self.marks_sent[i];
        if m.confirmed {
            return;
        }
        m.confirmed = true;
        let Some(mut cmd) = m.cmd.clone() else { return };
        let ttl = self.p.bump_mark_ttl;
        let name = m.name.clone();
        let mut del = Map::new();
        del.insert("op".into(), "delete".into());
        del.insert("name".into(), name.clone().into());
        io.zone_command(py_json(&Value::Object(del), None));
        let margin = cmd.get("margin").and_then(Value::as_f64).unwrap_or(0.0).max(self.p.bump_mark_confirmed_margin);
        cmd.insert("margin".into(), Value::from(margin));
        cmd.insert("ttl_s".into(), Value::from(ttl));
        self.clear_robot(&mut cmd, at.0, at.1);
        io.zone_command(py_json(&Value::Object(cmd.clone()), None));
        let m = &mut self.marks_sent[i];
        m.cmd = Some(cmd);
        m.sent = now;
        m.ttl = ttl;
        io.log(Level::Info, format!("Bumped at {name} again: keeping it for {}", duration_text(ttl)));
    }

    #[allow(clippy::too_many_arguments)]
    fn sent_mark(&mut self, bump: bool, name: &str, points: Vec<[f64; 2]>, sent: f64, ttl: f64, cmd: Option<Map<String, Value>>, confirmed: bool) {
        self.marks_sent.push(SentMark { bump, name: name.into(), points, sent, ttl, cmd, confirmed });
        if self.marks_sent.len() > 50 {
            self.marks_sent.remove(0);
        }
    }

    /// The one-shot timer after the Roomba left safe/full mode: if a cliff sensor fired, keep out
    /// of that spot.
    pub fn cliff_check(&mut self, io: &mut impl Io) {
        let now = io.now();
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
        let mut cmd = zone_add(&name, &points);
        self.clear_robot(&mut cmd, x, y);
        io.zone_command(py_json(&Value::Object(cmd), None));
        self.sent_mark(false, &name, points, now, 0.0, None, true);
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
            self.started = io.now();
            self.blacklist.clear();
            self.visited.clear();
            self.peeked.clear();
            self.reached = 0;
            self.failed = 0;
            self.reached_at_retry = -1;
        }
        self.no_frontier_since = None;
        self.resume_after_home = false;
        self.home_failures = 0;
        self.looked_here = false;
        // Look around once at the first spot reached, not here: it may be on the dock.
        self.look_pending = self.p.look_steps > 0;
        self.look = None;
        self.turn = None;
        self.slow_turns(true, io);
        self.set(State::Exploring, "Looking for unmapped areas…".into(), io);
        io.log(Level::Info, format!("Exploration started; home at ({:.2}, {:.2})", pose.0, pose.1));
    }

    fn stop(&mut self, why: String, io: &mut impl Io) {
        self.resume_after_home = false;
        self.cancel_unstuck(io);
        self.cancel_goal(io);
        self.cancel_spin(io);
        self.slow_turns(false, io);
        self.set(State::Stopped, why.clone(), io);
        io.log(Level::Info, why);
    }

    fn go_home(&mut self, why: String, resume: bool, io: &mut impl Io) {
        self.resume_after_home = resume;
        self.cancel_unstuck(io);
        self.cancel_spin(io);
        let Some(home) = self.home.filter(|_| self.p.return_home) else {
            self.finish("Exploration finished".into(), io);
            return;
        };
        self.cancel_goal(io);
        self.set(State::Returning, why, io);
        self.send_goal(home.0, home.1, None, GoalKind::Home, io);
    }

    fn finish(&mut self, why: String, io: &mut impl Io) {
        self.cancel_unstuck(io);
        self.cancel_spin(io);
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
        let now = io.now();
        if let Some(call) = self.unstuck.filter(|c| now - c.sent > self.p.unstuck_call_timeout) {
            io.log(Level::Warn, format!("The unstuck skill did not finish within {:.0} s", self.p.unstuck_call_timeout));
            if call.accepted {
                io.cancel_unstuck(call.seq);
            }
            self.unstuck_done(false, "no answer", io);
        }
        let spin_overdue = self.spin.is_some_and(|(_, sent, accepted)| {
            now - sent > if accepted { self.p.look_timeout + 5.0 } else { self.p.response_timeout }
        });
        if self.sending.is_some_and(|(_, sent)| now - sent > self.p.response_timeout) {
            self.no_response(io);
        } else if spin_overdue {
            let accepted = self.spin.is_some_and(|s| s.2);
            io.log(
                Level::Warn,
                if accepted {
                    "Looking around: the turn did not finish".into()
                } else {
                    "Looking around: navigation did not answer the turn request".into()
                },
            );
            self.cancel_spin(io);
            self.look_done();
        } else if self.state == State::Exploring {
            let timeout = self.p.goal_timeout;
            let stalled = if self.watch_progress(now, io) { self.goal } else { None };
            if let Some(b) = self.battery.filter(|&b| b < self.p.low_battery) {
                self.go_home(format!("Battery at {:.0}%: returning to the start point", b * 100.0), false, io);
            } else if !self.tracking_ok {
                self.message = PAUSED.into();
            } else if self.message == PAUSED {
                // Tracking is back: say what it is doing again (it kept the goal meanwhile).
                self.message = if self.goal.is_some() { self.goal_message.clone() } else { "Looking for unmapped areas…".into() };
            } else if stalled.is_some_and(|g| {
                let why = format!("no progress for {:.0} s", self.p.unstuck_no_progress);
                self.try_unstuck(Some(g), true, &why, io)
            }) {
                // the unstuck skill drives now
            } else if let Some(goal) = self.goal.filter(|g| now - g.sent > timeout) {
                io.log(Level::Warn, "Frontier goal timed out; skipping it".into());
                self.blacklist(goal.x, goal.y, true);
                self.failed += 1;
                self.cancel_goal(io);
            } else if self.goal.is_none() && self.sending.is_none() && self.spin.is_none() && self.unstuck.is_none() {
                self.next_frontier(io);
            }
        } else if self.state == State::Returning
            && self.p.return_timeout > 0.0
            && self.goal.is_some_and(|g| now - g.sent > self.p.return_timeout)
        {
            io.log(
                Level::Warn,
                format!("Driving back to the start point timed out after {:.0} s", self.p.return_timeout),
            );
            self.cancel_goal(io);
            self.returned(false, io);
        }
        self.publish_status(io);
        self.publish_markers(io);
        self.publish_drops(io);
    }

    /// ~/drops: every drop-off cell it remembers (explorer.py _publish_drops).
    fn publish_drops(&mut self, io: &mut impl Io) {
        let cells = if self.p.drop_clear > 0.0 { self.drop_cells() } else { Vec::new() };
        if cells.is_empty() {
            return;
        }
        let pts: Vec<[f32; 3]> =
            cells.iter().map(|&(i, j)| [((i as f64 + 0.5) * DROP_CELL_M) as f32, ((j as f64 + 0.5) * DROP_CELL_M) as f32, 0.0]).collect();
        io.drops(&self.p.map_frame, &pts);
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
            cost_weight: self.p.cost_weight,
            peek_cells: self.p.peek_search / res,
            split_cells: self.p.frontier_split / res,
        };
        // Drop-offs it was told about stay out of bounds even where the map shows floor.
        let drops = if self.p.drop_clear > 0.0 { self.drop_cells() } else { Vec::new() };
        let mut data = grid.data.clone();
        if !drops.is_empty() {
            block_drops(&mut data, w, h, res, ox, oy, &drops, self.p.drop_clear / res);
        }
        let (goals, seed, peeks, reach_cells) = frontier_goals(&data, w, h, robot, &fp);
        if seed.is_none() {
            self.message = "Robot is not in free space on the costmap; waiting…".into();
            return;
        }
        let free_cells = data.iter().filter(|&&g| g >= 0 && (g as i64) <= self.p.free_max).count();
        if self.break_trap(reach_cells, free_cells, res, pose, io) {
            return;
        }
        if let Some(turn) = self.turn.take() {
            self.message = "Turning to face the unmapped area…".into();
            self.send_spin(turn, io);
            return;
        }
        if self.look.is_some_and(|l| !l.1) {
            self.message = "Looking around…".into();
            self.look_step(io);
            return;
        }
        let groups = if io.markers_wanted() { Some(frontier_cells(&data, w, h, self.p.free_max, self.p.min_cells)) } else { None };
        self.reopen(io);
        let mut best: Option<(f64, f64, f64, f64)> = None;
        let mut candidates = 0;
        let mut options = Vec::new(); // for ~/markers
        let mut chosen = 0;
        let mut ranked = Vec::new(); // goal_ranking: learned
        for g in &goals {
            let (gy, gx) = (g.goal.0 as f64, g.goal.1 as f64);
            let (cy, cx) = g.centroid;
            let (x, y) = (ox + (gx + 0.5) * res, oy + (gy + 0.5) * res);
            let (fx, fy) = (ox + (cx + 0.5) * res, oy + (cy + 0.5) * res);
            if self.blacklisted(x, y) || self.is_visited(x, y) {
                let why = if self.blacklisted(x, y) { "failed earlier" } else { "looked from here" };
                options.push((x, y, GREY, why.to_string()));
                continue;
            }
            let d = py_hypot(x - pose.0, y - pose.1);
            if d < self.p.min_goal_dist {
                // Already standing here and the camera still can't fill it in: don't come back.
                self.visited.push((x, y));
                options.push((x, y, GREY, "here already".to_string()));
                continue;
            }
            candidates += 1;
            // Prefer near frontiers, but let bigger ones win over slightly nearer small ones.
            let score = d - 0.01 * g.size as f64;
            options.push((x, y, BLUE, format!("{d:.1} m, score {score:.2}")));
            if best.is_none_or(|b| score < b.0) {
                best = Some((score, x, y, (fy - y).atan2(fx - x))); // look at the unknown
                chosen = options.len() - 1;
            }
            if self.ranker.is_some() {
                ranked.push((RankCandidate { x, y, yaw: (fy - y).atan2(fx - x), frontier: (fx, fy), size_cells: g.size, score }, options.len() - 1));
            }
        }
        if let (Some(ranker), true) = (self.ranker.as_mut(), self.p.goal_ranking == "learned" && ranked.len() > 1) {
            // goal_ranking: learned - the ranker re-orders the candidates the rules above let through.
            ranked.sort_by(|a: &(RankCandidate, usize), b| a.0.score.total_cmp(&b.0.score));
            let blacklist: Vec<(f64, f64)> = self.blacklist.iter().map(|f| (f.x, f.y)).collect();
            let cands: Vec<RankCandidate> = ranked.iter().map(|r| r.0.clone()).collect();
            let input = RankInput {
                grid: self.grid.as_ref().unwrap(),
                pose,
                home: self.home,
                battery: self.battery,
                elapsed_s: io.now() - self.started,
                failed: self.failed,
                blacklist: &blacklist,
                visited: &self.visited,
            };
            match ranker.choose(&input, &cands) {
                Ok(Some(i)) if i < cands.len() => {
                    best = Some((cands[i].score, cands[i].x, cands[i].y, cands[i].yaw));
                    chosen = ranked[i].1;
                }
                Ok(_) => {}
                Err(e) => io.log(Level::Warn, format!("Learned goal ranking failed ({e}); using the rule's choice")),
            }
        }
        if let Some(groups) = groups {
            if best.is_some() {
                let o = &mut options[chosen];
                o.2 = GREEN;
                o.3 = format!("chosen: {}", o.3);
            }
            let cell = |r: usize, c: usize| [ox + (c as f64 + 0.5) * res, oy + (r as f64 + 0.5) * res, 0.0];
            let frontiers = groups.iter().map(|g| g.iter().map(|&(r, c)| cell(r, c)).collect()).collect();
            let peeks = peeks.iter().map(|g| (ox + (g.goal.1 as f64 + 0.5) * res, oy + (g.goal.0 as f64 + 0.5) * res)).collect();
            self.viz = Some(Viz { res, frontiers, options, peeks });
        }
        self.frontier_count = candidates;
        if let Some((_, x, y, yaw)) = best {
            self.look = None;
            self.no_frontier_since = None;
            self.message = format!("Heading to an unmapped area {:.1} m away", py_hypot(x - pose.0, y - pose.1));
            self.send_goal(x, y, Some(yaw), GoalKind::Frontier, io);
            return;
        }
        let now = io.now();
        let since = *self.no_frontier_since.get_or_insert(now);
        if self.look.is_none() && !self.looked_here && self.p.look_steps > 0 {
            self.look = Some((self.p.look_steps, true));
            self.message = "No unmapped areas in view: looking around…".into();
        }
        if self.look.is_some() {
            self.look_step(io);
            return;
        }
        if let Some((d, x, y, yaw)) = self.peek(&peeks, pose, res, ox, oy) {
            self.no_frontier_since = None;
            self.message = format!("Taking a closer look at an area it can't reach yet, {d:.1} m away");
            self.send_goal(x, y, Some(yaw), GoalKind::Peek, io);
            return;
        }
        if now - since < self.p.finish_confirm {
            self.message = "No reachable unmapped areas right now; checking again…".into();
            return;
        }
        self.no_frontier_since = None;
        let marks = self.bump_marks(now, true);
        if self.p.retry_failed && (!self.blacklist.is_empty() || !marks.is_empty()) && self.reached != self.reached_at_retry {
            self.reached_at_retry = self.reached;
            if !self.blacklist.is_empty() {
                io.log(
                    Level::Info,
                    format!("No unmapped areas left except {} that failed earlier: trying those again", self.blacklist.len()),
                );
            }
            self.drop_bump_marks(&marks, io);
            self.message = "Retrying areas that failed earlier…".into();
            self.blacklist.clear();
            return;
        }
        // Finish at the start point: go back there and check once more (the map looks different
        // from there), unless already there; a failed trip there is tried again up to
        // return_home_attempts in all: without the isolated bump marks (a person's; marks in a row
        // are a railing or glass the camera misses, and route the trip around it), and before the
        // last trip without any (people walking into it in a hallway leave rows too).
        let why = "Exploration finished: no reachable unmapped areas left";
        if self.near_home(io) {
            self.finish(format!("{why} - back at the start point"), io);
        } else if self.home_failures >= self.p.home_attempts {
            self.finish(format!("{why} - could not drive back to the start point"), io);
        } else {
            if self.home_failures > 0 {
                let marks = self.bump_marks(now, self.home_failures < self.p.home_attempts - 1);
                self.drop_bump_marks(&marks, io);
            }
            self.go_home(why.into(), true, io);
        }
    }

    /// Indexes (in marks_sent) of the bump marks still in force; isolated: only those with no
    /// other one within twice the mark width (a railing or glass wall gets bumped and marked along
    /// its length; someone walking into the robot leaves single marks).
    fn bump_marks(&self, now: f64, isolated: bool) -> Vec<usize> {
        let marks: Vec<usize> =
            (0..self.marks_sent.len()).filter(|&i| self.marks_sent[i].bump && (self.marks_sent[i].ttl <= 0.0 || now - self.marks_sent[i].sent < self.marks_sent[i].ttl)).collect();
        if !isolated {
            return marks;
        }
        let centres: Vec<(f64, f64)> = marks
            .iter()
            .map(|&i| {
                let ps = &self.marks_sent[i].points;
                let n = ps.len() as f64;
                (py_sum(ps.iter().map(|p| p[0])) / n, py_sum(ps.iter().map(|p| p[1])) / n)
            })
            .collect();
        let near = 2.0 * self.p.bump_mark_width;
        (0..marks.len())
            .filter(|&k| !centres.iter().enumerate().any(|(j, c)| j != k && py_hypot(centres[k].0 - c.0, centres[k].1 - c.1) < near))
            .map(|k| marks[k])
            .collect()
    }

    /// Delete these bump marks (indexes in marks_sent): someone walking into the robot leaves them
    /// too, and they can close a hallway for bump_mark_ttl_s (what is really there gets bumped and
    /// marked again). Before finishing only isolated ones; before trying the trip home again
    /// isolated ones, and all of them before the last trip.
    fn drop_bump_marks(&mut self, marks: &[usize], io: &mut impl Io) {
        if marks.is_empty() {
            return;
        }
        self.delete_marks(marks, io);
        self.bump_points.clear();
        io.log(
            Level::Info,
            format!("Removed {} bump mark(s) (a person walking into the robot leaves them too): checking again", marks.len()),
        );
    }

    /// Boxed in by its own bump marks (see trap_area_m2): removes the one that matters least
    /// (unconfirmed before confirmed, rows along a railing last, oldest first), at most one every
    /// trap_wait_s. True when it removed one.
    fn break_trap(&mut self, reach_cells: usize, free_cells: usize, res: f64, pose: (f64, f64, f64), io: &mut impl Io) -> bool {
        let now = io.now();
        if self.p.trap_area <= 0.0 || now < self.trap_next {
            return false;
        }
        let reach = reach_cells as f64 * res * res;
        let free = free_cells as f64 * res * res;
        if reach >= self.p.trap_area || free - reach < self.p.trap_area {
            return false;
        }
        let active = self.bump_marks(now, false);
        let near: Vec<usize> = active
            .iter()
            .copied()
            .filter(|&i| {
                let (mx, my) = mark_centre(&self.marks_sent[i].points);
                py_hypot(mx - pose.0, my - pose.1) <= self.p.trap_radius
            })
            .collect();
        let Some(&first) = near.first() else { return false };
        let rank = |i: usize| self.in_row(i, &active) as u8 * 2 + self.marks_sent[i].confirmed as u8;
        let mut pick = first;
        for &i in &near[1..] {
            let (a, b) = ((rank(i), self.marks_sent[i].sent), (rank(pick), self.marks_sent[pick].sent));
            if a.0 < b.0 || (a.0 == b.0 && a.1 < b.1) {
                pick = i;
            }
        }
        let name = self.marks_sent[pick].name.clone();
        self.delete_marks(&[pick], io);
        io.log(Level::Info, format!("Boxed in by its own bump marks ({reach:.1} m2 to move in): removed {name}"));
        self.message = "Boxed in by its own bump marks: clearing a way out…".into();
        self.trap_next = now + self.p.trap_wait;
        true
    }

    /// Whether bump mark `i` lies in a row of marks (a railing or glass wall marked along its
    /// length): it has neighbours within twice the mark width, all roughly on one line through it.
    fn in_row(&self, i: usize, active: &[usize]) -> bool {
        let (cx, cy) = mark_centre(&self.marks_sent[i].points);
        let near = 2.0 * self.p.bump_mark_width;
        let dirs: Vec<(f64, f64)> = active
            .iter()
            .filter(|&&j| j != i)
            .map(|&j| {
                let (x, y) = mark_centre(&self.marks_sent[j].points);
                (x - cx, y - cy)
            })
            .filter(|&(dx, dy)| py_hypot(dx, dy) < near)
            .collect();
        if dirs.is_empty() {
            return false;
        }
        dirs.iter().enumerate().all(|(k, a)| {
            dirs[k + 1..].iter().all(|b| (a.0 * b.1 - a.1 * b.0).abs() <= 0.5 * py_hypot(a.0, a.1) * py_hypot(b.0, b.1))
        })
    }

    /// Send a delete for these marks (indexes in marks_sent) and forget them.
    fn delete_marks(&mut self, marks: &[usize], io: &mut impl Io) {
        for &i in marks {
            let mut cmd = Map::new();
            cmd.insert("op".into(), "delete".into());
            cmd.insert("name".into(), self.marks_sent[i].name.clone().into());
            io.zone_command(py_json(&Value::Object(cmd), None));
        }
        let mut k = 0;
        self.marks_sent.retain(|_| {
            k += 1;
            !marks.contains(&(k - 1))
        });
    }

    /// The nearest spot to take a closer look at a frontier it can't reach, not tried yet:
    /// (distance, x, y, yaw).
    fn peek(&mut self, peeks: &[FrontierGoal], pose: (f64, f64, f64), res: f64, ox: f64, oy: f64) -> Option<(f64, f64, f64, f64)> {
        let mut best: Option<(f64, f64, f64, f64)> = None;
        for g in peeks {
            let (gy, gx) = (g.goal.0 as f64, g.goal.1 as f64);
            let (cy, cx) = g.centroid;
            let (x, y) = (ox + (gx + 0.5) * res, oy + (gy + 0.5) * res);
            let (fx, fy) = (ox + (cx + 0.5) * res, oy + (cy + 0.5) * res);
            if self.blacklisted(x, y) || self.peeked.iter().any(|&(px, py)| py_hypot(x - px, y - py) < self.p.visited_radius) {
                continue;
            }
            let d = py_hypot(x - pose.0, y - pose.1);
            if d < self.p.min_goal_dist {
                self.peeked.push((x, y)); // looking from here already (looked around)
                continue;
            }
            if best.is_none_or(|b| d < b.0) {
                best = Some((d, x, y, (fy - y).atan2(fx - x)));
            }
        }
        if let Some(b) = best {
            self.peeked.push((b.1, b.2));
        }
        best
    }

    /// The next turn of the look-around, or its end.
    fn look_step(&mut self, io: &mut impl Io) {
        let Some((left, early)) = self.look else { return };
        if left > 0 && io.now() < self.look_until {
            return; // standing still between turns (look_around_pause_s)
        }
        if left <= 0 || !io.spin_ready() {
            self.look_done();
            return;
        }
        self.look = Some((left - 1, early));
        self.send_spin(self.p.look_step, io);
    }

    /// A turn in place by ngle (rad, + = left) with Nav2's spin behaviour.
    fn send_spin(&mut self, angle: f64, io: &mut impl Io) {
        if !io.spin_ready() {
            return;
        }
        self.spin_seq += 1;
        let seq = self.spin_seq;
        self.spin = Some((seq, io.now(), false));
        let t = self.p.look_timeout;
        io.send_spin(seq, angle as f32, t as i32, ((t - (t as i32) as f64) * 1e9) as u32);
    }

    fn look_done(&mut self) {
        self.look = None;
        self.looked_here = true;
    }

    /// Nav2's answer to the look-around turn `seq`.
    pub fn on_spin_response(&mut self, seq: u64, accepted: bool, io: &mut impl Io) {
        let Some((current, sent, _)) = self.spin.filter(|s| s.0 == seq) else {
            // given up on or cancelled meanwhile
            if accepted {
                io.cancel_spin(seq);
            }
            return;
        };
        if !accepted {
            self.spin = None;
            self.look_rejects += 1;
            if self.look_rejects == 1 {
                io.log(Level::Warn, "Looking around: navigation rejected the turn".into());
            }
            if let Some((left, early)) = self.look {
                if self.look_rejects >= LOOK_REJECTS {
                    self.look_done();
                } else {
                    self.look = Some((left + 1, early)); // Nav2 still starting? that turn again next tick
                }
            }
            return;
        }
        self.look_rejects = 0;
        self.spin = Some((current, sent, true));
        self.spin_handle = Some(seq);
    }

    /// The final status of the accepted look-around turn `seq`.
    pub fn on_spin_result(&mut self, seq: u64, status: i8, io: &mut impl Io) {
        if !self.spin.is_some_and(|s| s.0 == seq) {
            return;
        }
        self.spin = None;
        self.spin_handle = None;
        if status != STATUS_SUCCEEDED {
            io.log(Level::Warn, "Looking around: the turn stopped early (obstacle?)".into());
            if self.look.is_some() {
                self.look_done();
            }
        } else if self.look.is_some() && self.p.look_pause > 0.0 {
            self.look_until = io.now() + self.p.look_pause;
        }
    }

    fn cancel_spin(&mut self, io: &mut impl Io) {
        self.look = None;
        self.turn = None;
        let handle = self.spin_handle.take();
        self.spin = None;
        if let Some(seq) = handle {
            io.cancel_spin(seq);
        }
    }

    /// Skip this spot until the map around it opens up (see [`Explorer::reopen`]); `reopen`
    /// false: only the retry before finishing (the camera can't see what was in the way).
    fn blacklist(&mut self, x: f64, y: f64, reopen: bool) {
        let patch = match &self.grid {
            Some(g) if reopen && self.p.reopen_cells > 0 && g.data.len() == g.width as usize * g.height as usize => {
                Some(passable_patch(g, self.p.blacklist_radius, self.p.pass_max, x, y))
            }
            _ => None,
        };
        self.blacklist.push(Failed { x, y, patch });
    }

    /// Failed spots where the map has opened up since (a door opened, a person walked away): try
    /// them again.
    fn reopen(&mut self, io: &mut impl Io) {
        if self.p.reopen_cells <= 0 {
            return;
        }
        let Some(grid) = &self.grid else { return };
        let (radius, pass_max, min) = (self.p.blacklist_radius, self.p.pass_max, self.p.reopen_cells);
        let opened: Vec<bool> = self
            .blacklist
            .iter()
            .map(|f| {
                f.patch.as_ref().is_some_and(|old| {
                    let now = passable_patch(grid, radius, pass_max, f.x, f.y);
                    now.len() == old.len() && now.iter().zip(old).filter(|&(&n, &o)| n && !o).count() as i64 >= min
                })
            })
            .collect();
        let mut kept = Vec::with_capacity(self.blacklist.len());
        for (f, open) in std::mem::take(&mut self.blacklist).into_iter().zip(opened) {
            if open {
                io.log(
                    Level::Info,
                    format!("The map opened up around an area that failed earlier ({:.1}, {:.1}): trying it again", f.x, f.y),
                );
            } else {
                kept.push(f);
            }
        }
        self.blacklist = kept;
    }

    fn blacklisted(&self, x: f64, y: f64) -> bool {
        self.blacklist.iter().any(|f| py_hypot(x - f.x, y - f.y) < self.p.blacklist_radius)
    }

    fn is_visited(&self, x: f64, y: f64) -> bool {
        self.visited.iter().any(|&(vx, vy)| py_hypot(x - vx, y - vy) < self.p.visited_radius)
    }

    // ---- navigation ----

    fn send_goal(&mut self, x: f64, y: f64, yaw: Option<f64>, kind: GoalKind, io: &mut impl Io) {
        let pose = self.pose(io);
        let yaw = match yaw {
            Some(yaw) => yaw,
            None => pose.map_or(0.0, |p| p.2),
        };
        self.goal_seq += 1;
        let seq = self.goal_seq;
        let sent = io.now();
        self.goal = Some(Goal { x, y, sent, numpy: kind != GoalKind::Home, kind, yaw });
        self.goal_message = self.message.clone();
        self.goal_from = pose.map(|p| (p.0, p.1));
        self.goal_bumps = 0;
        self.sending = Some((seq, sent));
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
        if self.sending.is_some_and(|(s, _)| s == seq) {
            self.sending = None;
        }
        if seq != self.goal_seq {
            // superseded, cancelled or given up on meanwhile
            if accepted {
                io.cancel_goal(seq);
            }
            return;
        }
        if !accepted {
            io.log(Level::Warn, "Navigation rejected the goal".into());
            self.refused(io);
            return;
        }
        self.goal_handle = Some(seq);
    }

    /// Nav2 never answered the goal request: stop waiting and count the goal as rejected.
    fn no_response(&mut self, io: &mut impl Io) {
        let Some((seq, _)) = self.sending.take() else { return };
        if seq != self.goal_seq {
            return; // cancelled or superseded meanwhile: nothing left to do
        }
        self.goal_seq += 1; // an answer that still comes is then for a superseded goal (cancelled)
        io.log(
            Level::Warn,
            format!("Navigation did not answer the goal request within {:.0} s; giving up on it", self.p.response_timeout),
        );
        self.refused(io);
    }

    /// The current goal was rejected (or never answered): skip that spot, or end the trip home.
    fn refused(&mut self, io: &mut impl Io) {
        let goal = self.goal.take();
        if self.state == State::Returning {
            self.returned(false, io);
        } else if let Some(g) = goal {
            self.blacklist(g.x, g.y, true);
        }
    }

    /// The final status of the accepted goal `seq`.
    pub fn on_result(&mut self, seq: u64, status: i8, io: &mut impl Io) {
        if seq != self.goal_seq {
            return;
        }
        let goal = self.goal.take();
        self.goal_handle = None;
        if self.state == State::Returning {
            // Nav2 can report success without the robot moving (a one-pose plan while it was
            // boxed in at a wall), so check that it really is back.
            let home = status == STATUS_SUCCEEDED && self.near_home(io);
            self.returned(home, io);
            return;
        }
        if self.state != State::Exploring {
            return;
        }
        let peek = goal.is_some_and(|g| g.kind == GoalKind::Peek);
        if status == STATUS_SUCCEEDED {
            self.looked_here = false; // somewhere new: look around here before giving up
            if self.look_pending {
                self.look_pending = false;
                self.look = Some((self.p.look_steps, false));
            } else if let Some(g) = goal.filter(|_| self.p.face_turn_min > 0.0) {
                if let Some((_, _, yaw)) = self.pose(io) {
                    let d = g.yaw - yaw;
                    let d = d.sin().atan2(d.cos());
                    if d.abs() > self.p.face_turn_min {
                        self.turn = Some(d); // face the unknown it came to see
                    }
                }
            }
            if peek {
                return;
            }
            self.reached += 1;
            self.home_failures = 0;
            if let Some(g) = goal {
                // Looked from here once; if the camera couldn't fill it in, don't come back.
                self.visited.push((g.x, g.y));
            }
        } else if let Some(g) = goal.filter(|_| !peek) {
            if self.p.unstuck_skill && self.boxed_in(io) && self.try_unstuck(Some(g), false, "navigation gave up next to an obstacle", io) {
                return;
            }
            // Unreachable or no progress: skip this spot (until the map around it opens up).
            self.failed += 1;
            self.blacklist(g.x, g.y, true);
        }
    }

    /// The trip home ended, at the start point (`home`) or not.
    fn returned(&mut self, home: bool, io: &mut impl Io) {
        let why = self.message.clone(); // why it went home (finished, low battery, request)
        if self.resume_after_home {
            // The map looks different from here: check once more before finishing.
            self.resume_after_home = false;
            self.home_failures = if home { 0 } else { self.home_failures + 1 };
            self.looked_here = false;
            self.no_frontier_since = None;
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
    }

    // ---- learned unstuck skill (unstuck_skill: true) ----

    /// Ask the unstuck skill to free the robot. `retry`: the goal to try again afterwards;
    /// `give_back`: on failure send it again too (Nav2 was still trying), else skip it as without
    /// the skill. False (nothing done) when the skill is off, busy, not available or was already
    /// called for this goal: the caller then carries on as without it.
    fn try_unstuck(&mut self, retry: Option<Goal>, give_back: bool, why: &str, io: &mut impl Io) -> bool {
        if !self.p.unstuck_skill || self.unstuck.is_some() || self.state != State::Exploring {
            return false;
        }
        if let (Some(g), Some(u)) = (retry, self.unstuck_goal) {
            if py_hypot(g.x - u.0, g.y - u.1) < 0.05 {
                return false;
            }
        }
        if !io.unstuck_ready() {
            if !self.unstuck_missing_logged {
                self.unstuck_missing_logged = true;
                io.log(Level::Warn, "unstuck_skill is on but its action server is not available; carrying on without it".into());
            }
            return false;
        }
        self.cancel_goal(io);
        self.unstuck_seq += 1;
        let seq = self.unstuck_seq;
        self.unstuck = Some(UnstuckCall { seq, sent: io.now(), accepted: false, retry, give_back });
        self.unstuck_goal = retry.map(|g| (g.x, g.y));
        self.progress = None;
        self.message = format!("Stuck ({why}): trying the learned unstuck skill…");
        io.log(Level::Info, format!("Stuck ({why}): calling the unstuck skill"));
        io.send_unstuck(seq, retry.map(|g| (g.x, g.y)), self.p.unstuck_time_limit);
        true
    }

    /// The unstuck skill's answer to call `seq`.
    pub fn on_unstuck_response(&mut self, seq: u64, accepted: bool, io: &mut impl Io) {
        match self.unstuck.as_mut() {
            Some(c) if c.seq == seq => {
                if accepted {
                    c.accepted = true;
                } else {
                    self.unstuck_done(false, "rejected", io);
                }
            }
            _ => {
                if accepted {
                    io.cancel_unstuck(seq); // given up on meanwhile
                }
            }
        }
    }

    /// The unstuck skill's result for call `seq`: whether the robot is free, and its message.
    pub fn on_unstuck_result(&mut self, seq: u64, free: bool, detail: &str, io: &mut impl Io) {
        if self.unstuck.is_some_and(|c| c.seq == seq) {
            self.unstuck_done(free, detail, io);
        }
    }

    fn unstuck_done(&mut self, free: bool, detail: &str, io: &mut impl Io) {
        let Some(call) = self.unstuck.take() else { return };
        if free {
            io.log(Level::Info, format!("Unstuck skill: free ({detail})"));
        } else {
            io.log(Level::Warn, format!("Unstuck skill did not free the robot ({detail})"));
        }
        if self.state != State::Exploring {
            return;
        }
        match call.retry {
            Some(g) if free || call.give_back => {
                self.message = if free {
                    "Free again: trying that area once more".into()
                } else {
                    "Unstuck skill failed: back to navigation's own recoveries".into()
                };
                self.send_goal(g.x, g.y, Some(g.yaw), g.kind, io);
            }
            Some(g) => {
                // as without the skill: skip this spot
                self.failed += 1;
                self.blacklist(g.x, g.y, true);
            }
            None => {}
        }
    }

    fn cancel_unstuck(&mut self, io: &mut impl Io) {
        if let Some(c) = self.unstuck.take() {
            if c.accepted {
                io.cancel_unstuck(c.seq);
            }
        }
    }

    /// unstuck_skill: whether the accepted goal being driven to has made no progress (0.2 m) for
    /// unstuck_no_progress_s.
    fn watch_progress(&mut self, now: f64, io: &mut impl Io) -> bool {
        if !self.p.unstuck_skill || self.p.unstuck_no_progress <= 0.0 || self.unstuck.is_some() || self.goal_handle.is_none() || !self.tracking_ok {
            self.progress = None;
            return false;
        }
        let Some((x, y, _)) = self.pose(io) else { return false };
        match self.progress {
            Some((px, py, since)) if py_hypot(x - px, y - py) < 0.2 => now - since > self.p.unstuck_no_progress,
            _ => {
                self.progress = Some((x, y, now));
                false
            }
        }
    }

    /// unstuck_skill: a lethal cell of the global costmap within unstuck_clearance_m of the robot's
    /// centre (it touches something, or a keep-out mark).
    fn boxed_in(&self, io: &mut impl Io) -> bool {
        let (Some(grid), Some(p)) = (self.grid.as_ref(), self.pose(io)) else { return false };
        let res = grid.resolution as f64;
        let (w, h) = (grid.width as i64, grid.height as i64);
        if res <= 0.0 || grid.data.len() as i64 != w * h {
            return false;
        }
        let cx = ((p.0 - grid.origin_x) / res).floor() as i64;
        let cy = ((p.1 - grid.origin_y) / res).floor() as i64;
        let r = (self.p.unstuck_clearance / res).ceil() as i64;
        for dy in -r..=r {
            for dx in -r..=r {
                let (x, y) = (cx + dx, cy + dy);
                if x < 0 || y < 0 || x >= w || y >= h || grid.data[(y * w + x) as usize] < 100 {
                    continue;
                }
                if ((dx * dx + dy * dy) as f64).sqrt() * res < self.p.unstuck_clearance {
                    return true;
                }
            }
        }
        false
    }

    fn cancel_goal(&mut self, io: &mut impl Io) {
        self.goal_seq += 1;
        let handle = self.goal_handle.take();
        self.goal = None;
        if let Some(seq) = handle {
            io.cancel_goal(seq);
        }
    }

    /// On shutdown: cancel what the robot is doing for the explorer (goal, look-around turn), as the
    /// Python node does on Ctrl-C.
    pub fn shutdown(&mut self, io: &mut impl Io) {
        self.cancel_unstuck(io);
        self.cancel_goal(io);
        self.cancel_spin(io);
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

    /// How it decides where to go, for ~/markers (explorer.py _marker_specs): the frontier groups
    /// of its last search, the spots it considered there (distance and score, the one it chose, or
    /// why it skipped them), spots to look at frontiers it can't reach from, failed spots (and when
    /// they are retried), visited spots, the keep-out marks it sent, home, and the goal with an
    /// arrow from where it was chosen.
    fn marker_specs(&self, now: f64) -> Vec<MarkerSpec> {
        let mut specs = Vec::new();
        let mut labels: Vec<(f64, f64, String)> = Vec::new();
        if let Some(v) = &self.viz {
            for (k, points) in v.frontiers.iter().enumerate() {
                specs.push(spec("frontiers", k, CUBE_LIST, [0.0; 3], [v.res, v.res, 0.01], ORANGE, points.clone(), String::new()));
            }
            for (k, (x, y, color, text)) in v.options.iter().enumerate() {
                specs.push(spec("options", k, SPHERE, [*x, *y, 0.05], [0.12; 3], *color, Vec::new(), String::new()));
                labels.push((*x, *y, text.clone()));
            }
            for (k, &(x, y)) in v.peeks.iter().enumerate() {
                specs.push(spec("peeks", k, SPHERE, [x, y, 0.05], [0.12; 3], PURPLE, Vec::new(), String::new()));
                labels.push((x, y, "can't reach: look from here".into()));
            }
        }
        let d = 2.0 * self.p.blacklist_radius;
        for (k, f) in self.blacklist.iter().enumerate() {
            specs.push(spec("failed", k, CYLINDER, [f.x, f.y, 0.01], [d, d, 0.02], RED_AREA, Vec::new(), String::new()));
            let when = if f.patch.is_some() { "failed: retry if the map opens up" } else { "failed: retry before finishing" };
            labels.push((f.x, f.y, when.into()));
        }
        if !self.visited.is_empty() {
            let points = self.visited.iter().map(|&(x, y)| [x, y, 0.05]).collect();
            specs.push(spec("visited", 0, SPHERE_LIST, [0.0; 3], [0.08; 3], GREY, points, String::new()));
        }
        let drops = self.drop_cells();
        if !drops.is_empty() {
            let points = drops.iter().map(|&(i, j)| [(i as f64 + 0.5) * DROP_CELL_M, (j as f64 + 0.5) * DROP_CELL_M, 0.01]).collect();
            specs.push(spec("drops", 0, CUBE_LIST, [0.0; 3], [DROP_CELL_M, DROP_CELL_M, 0.01], MAGENTA, points, String::new()));
        }
        let marks = self.marks_sent.iter().filter(|m| m.ttl <= 0.0 || now - m.sent < m.ttl);
        for (k, m) in marks.enumerate() {
            let points = m.points.iter().chain(m.points.first()).map(|p| [p[0], p[1], 0.03]).collect();
            specs.push(spec("marks", k, LINE_STRIP, [0.0; 3], [0.03, 0.0, 0.0], if m.bump { YELLOW } else { RED }, points, String::new()));
            labels.push((m.points[0][0], m.points[0][1], m.name.clone()));
        }
        if let Some((hx, hy)) = self.home {
            specs.push(spec("home", 0, CYLINDER, [hx, hy, 0.02], [0.3, 0.3, 0.04], CYAN, Vec::new(), String::new()));
            labels.push((hx, hy, "home".into()));
        }
        if let Some(g) = self.goal {
            if let Some((fx, fy)) = self.goal_from {
                specs.push(spec("goal", 0, ARROW, [0.0; 3], [0.03, 0.08, 0.1], GREEN, vec![[fx, fy, 0.1], [g.x, g.y, 0.1]], String::new()));
            } else {
                specs.push(spec("goal", 0, SPHERE, [g.x, g.y, 0.1], [0.2; 3], GREEN, Vec::new(), String::new()));
            }
            let text = match g.kind {
                GoalKind::Frontier => "goal: unmapped area",
                GoalKind::Peek => "goal: closer look",
                GoalKind::Home => "goal: start point",
            };
            labels.push((g.x, g.y, text.into()));
        }
        for (k, (x, y, text)) in labels.into_iter().enumerate() {
            specs.push(spec("labels", k, TEXT_VIEW_FACING, [x, y, 0.3], [0.0, 0.0, 0.1], WHITE, Vec::new(), text));
        }
        specs
    }

    /// ~/markers, when they changed and someone listens.
    fn publish_markers(&mut self, io: &mut impl Io) {
        if !io.markers_wanted() {
            self.markers_last = None;
            return;
        }
        let specs = self.marker_specs(io.now());
        if self.markers_last.as_ref() != Some(&specs) {
            io.markers(&self.p.map_frame, &specs);
            self.markers_last = Some(specs);
        }
    }
}

/// Passable cells (cost 0..=pass_max) in the square of `radius` around (x, y), row-major; outside
/// the grid: not passable (explorer.py _patch).
fn passable_patch(g: &Grid, radius: f64, pass_max: i64, x: f64, y: f64) -> Vec<bool> {
    let res = g.resolution as f64;
    let k = (radius / res) as i64;
    let (w, h) = (g.width as i64, g.height as i64);
    let (r0, c0) = (((y - g.origin_y) / res) as i64 - k, ((x - g.origin_x) / res) as i64 - k);
    let n = 2 * k + 1;
    let mut out = vec![false; (n * n) as usize];
    for i in 0..n {
        for j in 0..n {
            let (r, c) = (r0 + i, c0 + j);
            if r >= 0 && c >= 0 && r < h && c < w {
                let v = g.data[(r * w + c) as usize] as i64;
                out[(i * n + j) as usize] = v >= 0 && v <= pass_max;
            }
        }
    }
    out
}

fn zone_add(name: &str, points: &[[f64; 2]]) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("op".into(), "add".into());
    m.insert("kind".into(), "keepout".into());
    m.insert("name".into(), name.into());
    m.insert("points".into(), Value::from(points.iter().map(|p| Value::from(p.to_vec())).collect::<Vec<_>>()));
    m
}

/// A mark's centre: the mean of its corners (Python's sum).
fn mark_centre(points: &[[f64; 2]]) -> (f64, f64) {
    let n = points.len() as f64;
    (py_sum(points.iter().map(|p| p[0])) / n, py_sum(points.iter().map(|p| p[1])) / n)
}

/// Rotation matrix rows of a quaternion (x, y, z, w), as explorer.py _rotation computes them.
fn rotation(q: [f64; 4]) -> [[f64; 3]; 2] {
    let [x, y, z, w] = q;
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
    ]
}

/// (x, y) of point (px, py, pz) in the target frame of a transform (rows of `rotation`, `t`).
fn to_frame(r: &[[f64; 3]; 2], t: [f64; 3], px: f64, py: f64, pz: f64) -> (f64, f64) {
    (r[0][0] * px + r[0][1] * py + r[0][2] * pz + t[0], r[1][0] * px + r[1][1] * py + r[1][2] * pz + t[1])
}

/// "10 min" / "60 s", as the marks' log lines say how long they last.
fn duration_text(ttl: f64) -> String {
    if ttl >= 120.0 {
        format!("{:.0} min", ttl / 60.0)
    } else {
        format!("{ttl:.0} s")
    }
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
        let p = FrontierParams {
            free_max: 40,
            pass_max: 98,
            clear_max: 15,
            min_cells: 10,
            standoff_cells: 3.0,
            search_cells: 16.0,
            cost_weight: 0.05,
            peek_cells: 30.0,
            split_cells: 0.0,
        };
        let (goals, seed, peeks, _) = frontier_goals(&grid, w, h, (10, 3), &p);
        assert_eq!(seed, Some((10, 3)));
        assert_eq!(goals.len(), 1);
        assert!(peeks.is_empty());
        assert_eq!(goals[0].size, 18);
        assert_eq!(goals[0].centroid, (9.5, 12.0));
        // Clear cells end at column 9 (4 chamfer steps from the unknown); rows 9 and 10 tie: first wins.
        assert_eq!(goals[0].goal, (9, 9));
    }

    /// A 40 x 20 grid: a room (columns 1-14), a 5-cell passage (rows 8-12) through a wall at
    /// columns 15-16 into a second room that is known free up to column 30 (unknown beyond).
    fn two_rooms(passage_cost: i8) -> (Vec<i8>, usize, usize) {
        let (w, h) = (40usize, 20usize);
        let mut grid = vec![0i8; w * h];
        for r in 0..h {
            for c in 0..w {
                let wall = r == 0 || r == h - 1 || c == 0 || ((15..=16).contains(&c) && !(8..=12).contains(&r));
                grid[r * w + c] = if c > 30 {
                    -1
                } else if wall {
                    100
                } else if (13..=18).contains(&c) {
                    passage_cost // the passage and its surroundings: inflated
                } else {
                    0
                };
            }
        }
        (grid, w, h)
    }

    #[test]
    fn frontier_beyond_a_narrow_passage() {
        let p = |cost_weight: f64, peek_cells: f64| FrontierParams {
            free_max: 40,
            pass_max: 98,
            clear_max: 15,
            min_cells: 10,
            standoff_cells: 3.0,
            search_cells: 16.0,
            cost_weight,
            peek_cells,
            split_cells: 0.0,
        };
        // Passable (cost 80): reachable; the clear cells in the second room are near enough.
        let (grid, w, h) = two_rooms(80);
        let (goals, _, _, _) = frontier_goals(&grid, w, h, (10, 5), &p(0.05, 0.0));
        assert_eq!(goals.len(), 1);
        // Without clear cells near the frontier (the second room inflated too), the old rule
        // dropped it; the fallback picks the best cell below inscribed.
        let mut inflated = grid.clone();
        for r in 1..h - 1 {
            for c in 19..=29 {
                inflated[r * w + c] = 60;
            }
        }
        assert!(frontier_goals(&inflated, w, h, (10, 5), &p(0.0, 0.0)).0.is_empty());
        let (goals, _, _, _) = frontier_goals(&inflated, w, h, (10, 5), &p(0.05, 0.0));
        assert_eq!(goals.len(), 1);
        let g = goals[0].goal;
        assert!(inflated[g.0 * w + g.1] <= 98 && g.1 >= 19, "{g:?}");
        // Passage inscribed (cost 99): not reachable - no goal, but a spot to look at it from.
        let (grid, w, h) = two_rooms(99);
        let (goals, _, peeks, _) = frontier_goals(&grid, w, h, (10, 5), &p(0.05, 30.0));
        assert!(goals.is_empty());
        assert_eq!(peeks.len(), 1);
        assert!(peeks[0].goal.1 <= 12, "the spot is on the robot's side: {:?}", peeks[0].goal);
    }

    #[test]
    fn long_frontier_gets_a_spot_per_piece() {
        // A 100 x 20 room known up to row 12, unknown beyond: one frontier group 98 cells long.
        let (w, h) = (100usize, 20usize);
        let mut grid = vec![0i8; w * h];
        for r in 0..h {
            for c in 0..w {
                grid[r * w + c] = if r > 12 {
                    -1
                } else if r == 0 || c == 0 || c == w - 1 {
                    100
                } else {
                    0
                };
            }
        }
        let p = |split_cells: f64| FrontierParams {
            free_max: 40,
            pass_max: 98,
            clear_max: 15,
            min_cells: 10,
            standoff_cells: 3.0,
            search_cells: 16.0,
            cost_weight: 0.05,
            peek_cells: 0.0,
            split_cells,
        };
        let (one, _, _, _) = frontier_goals(&grid, w, h, (5, 50), &p(0.0));
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].centroid, (12.0, 49.5));
        let (goals, _, _, _) = frontier_goals(&grid, w, h, (5, 50), &p(40.0));
        // 98 cells / 40 -> 3 pieces of 32-33 cells, west to east; each keeps the group's size.
        assert_eq!(goals.len(), 3);
        assert_eq!(goals.iter().map(|g| g.centroid).collect::<Vec<_>>(), vec![(12.0, 16.5), (12.0, 49.0), (12.0, 82.0)]);
        assert!(goals.iter().all(|g| g.size == 98 && g.group == (12.0, 49.5)));
        assert!(goals[0].goal.1 < 30 && goals[2].goal.1 > 70, "{goals:?}");
        // Short groups and split 0 stay whole; an L-shape is cut along its longer side.
        assert_eq!(frontier_pieces(&[(0, 0), (0, 1)], 40.0).len(), 1);
        let l: Vec<(usize, usize)> = (0..10).map(|r| (r, 0)).chain((1..4).map(|c| (9, c))).collect();
        let pieces = frontier_pieces(&l, 4.0);
        assert_eq!(pieces.iter().map(Vec::len).collect::<Vec<_>>(), vec![4, 4, 5]);
        assert_eq!(pieces[0], vec![(0, 0), (1, 0), (2, 0), (3, 0)]);
    }

    #[test]
    fn json_with_nan() {
        let v = py_json_loads(r#"{"state": "idle", "rate": NaN, "s": "NaN"}"#).unwrap();
        assert_eq!(v["state"], "idle");
        assert_eq!(v["s"], "NaN");
    }

    /// Records what the explorer does; the robot stands still and time moves by hand.
    #[derive(Default)]
    struct TestIo {
        t: f64,
        pose: (f64, f64),
        yaw: f64,
        goals: Vec<GoalRequest>,
        cancels: Vec<u64>,
        logs: Vec<(Level, String)>,
        status: String,
        zones: Vec<String>,
        spin_server: bool,
        spins: Vec<(u64, f32)>,
        spin_cancels: Vec<u64>,
        want_markers: bool,
        markers: Vec<Vec<MarkerSpec>>,
        unstuck_server: bool,
        unstuck_calls: Vec<(u64, Option<(f64, f64)>)>,
        unstuck_cancels: Vec<u64>,
        drops: Vec<Vec<[f32; 3]>>,
    }

    impl Io for TestIo {
        fn now(&mut self) -> f64 {
            self.t
        }
        fn lookup(&mut self, _: &str, _: &str) -> Option<([f64; 3], [f64; 4])> {
            Some(([self.pose.0, self.pose.1, 0.0], [0.0, 0.0, (self.yaw / 2.0).sin(), (self.yaw / 2.0).cos()]))
        }
        fn nav_ready(&mut self) -> bool {
            true
        }
        fn send_goal(&mut self, goal: GoalRequest) {
            self.goals.push(goal);
        }
        fn cancel_goal(&mut self, seq: u64) {
            self.cancels.push(seq);
        }
        fn get_params_ready(&mut self) -> bool {
            false
        }
        fn set_params_ready(&mut self) -> bool {
            false
        }
        fn get_rotate(&mut self) {}
        fn set_rotate(&mut self, _: f64) {}
        fn status(&mut self, json: String) {
            self.status = json;
        }
        fn zone_command(&mut self, json: String) {
            self.zones.push(json);
        }
        fn log(&mut self, level: Level, msg: String) {
            self.logs.push((level, msg));
        }
        fn local_hms(&mut self) -> String {
            "000000".into()
        }
        fn start_cliff_timer(&mut self) {}
        fn spin_ready(&mut self) -> bool {
            self.spin_server
        }
        fn send_spin(&mut self, seq: u64, target_yaw: f32, _: i32, _: u32) {
            self.spins.push((seq, target_yaw));
        }
        fn cancel_spin(&mut self, seq: u64) {
            self.spin_cancels.push(seq);
        }
        fn markers_wanted(&mut self) -> bool {
            self.want_markers
        }
        fn markers(&mut self, frame: &str, markers: &[MarkerSpec]) {
            assert_eq!(frame, "map");
            self.markers.push(markers.to_vec());
        }
        fn drops(&mut self, frame: &str, points: &[[f32; 3]]) {
            assert_eq!(frame, "map");
            self.drops.push(points.to_vec());
        }
        fn unstuck_ready(&mut self) -> bool {
            self.unstuck_server
        }
        fn send_unstuck(&mut self, seq: u64, goal: Option<(f64, f64)>, _: f64) {
            self.unstuck_calls.push((seq, goal));
        }
        fn cancel_unstuck(&mut self, seq: u64) {
            self.unstuck_cancels.push(seq);
        }
    }

    impl TestIo {
        fn warnings(&self, text: &str) -> usize {
            self.logs.iter().filter(|(l, m)| *l == Level::Warn && m.contains(text)).count()
        }

        fn ticks(&mut self, e: &mut Explorer, from: u32, to: u32) {
            for t in from..=to {
                self.t = t as f64;
                e.tick(self);
            }
        }
    }

    /// Exploring the open room of frontier_goal_in_open_room (0.1 m cells, robot in cell (10, 3)),
    /// its one frontier goal sent at t = 1 s and not answered yet (no look-around).
    fn exploring(io: &mut TestIo) -> Explorer {
        exploring_with(io, |_| {})
    }

    /// `exploring` with some parameters changed.
    fn exploring_with(io: &mut TestIo, change: impl FnOnce(&mut Params)) -> Explorer {
        let mut p = Params::load(&mut crate::watchdog::Defaults).unwrap();
        p.look_steps = 0;
        change(&mut p);
        let mut e = open_room(io, p);
        io.t = 1.0;
        e.on_command("start", io);
        e.tick(io);
        assert_eq!(e.state(), State::Exploring);
        assert_eq!(io.goals.len(), 1);
        e
    }

    fn open_room(io: &mut TestIo, p: Params) -> Explorer {
        let (w, h) = (20u32, 20u32);
        let data = (0..h)
            .flat_map(|r| (0..w).map(move |c| if c >= 13 { -1 } else if r == 0 || r == h - 1 || c == 0 { 100 } else { 0 }))
            .collect();
        let mut e = Explorer::new(p, io);
        e.on_grid(Grid { width: w, height: h, resolution: 0.1, origin_x: 0.0, origin_y: 0.0, data });
        e.on_mode("full", io);
        io.pose = (0.35, 1.05);
        e
    }

    /// Exploring the open room with the learned unstuck skill on (`unstuck_skill: true`) or off.
    fn exploring_unstuck(io: &mut TestIo, on: bool) -> Explorer {
        let mut p = Params::load(&mut crate::watchdog::Defaults).unwrap();
        p.look_steps = 0;
        p.unstuck_skill = on;
        let mut e = open_room(io, p);
        io.t = 1.0;
        e.on_command("start", io);
        e.tick(io);
        assert_eq!(io.goals.len(), 1);
        e.on_goal_response(io.goals[0].seq, true, io);
        e
    }

    #[test]
    fn unstuck_skill_frees_the_robot_then_retries_the_goal_once() {
        // Off (the default): Nav2 gives up next to the wall, the spot is skipped, nothing is called.
        let mut io = TestIo { unstuck_server: true, ..Default::default() };
        let mut e = exploring_unstuck(&mut io, false);
        io.pose = (0.15, 1.05); // 0.1 m from the lethal wall cells of column 0
        e.on_result(io.goals[0].seq, 6, &mut io);
        io.ticks(&mut e, 2, 2);
        assert!(io.unstuck_calls.is_empty());
        assert!(io.status.contains("\"failed\": 1"), "{}", io.status);

        // On: the skill is called with that goal; no new goal while it drives; free -> same goal again.
        let mut io = TestIo { unstuck_server: true, ..Default::default() };
        let mut e = exploring_unstuck(&mut io, true);
        let g = io.goals[0].clone();
        io.pose = (0.15, 1.05);
        e.on_result(g.seq, 6, &mut io);
        assert_eq!(io.unstuck_calls, vec![(1, Some((g.x, g.y)))]);
        io.ticks(&mut e, 2, 4);
        assert_eq!(io.goals.len(), 1);
        e.on_unstuck_response(1, true, &mut io);
        e.on_unstuck_result(1, true, "free", &mut io);
        assert_eq!(io.goals.len(), 2);
        assert_eq!((io.goals[1].x, io.goals[1].y), (g.x, g.y));
        io.ticks(&mut e, 5, 5);
        assert!(io.status.contains("\"failed\": 0"), "{}", io.status);
        // The same goal failing again: no second call, skipped as without the skill.
        e.on_goal_response(io.goals[1].seq, true, &mut io);
        e.on_result(io.goals[1].seq, 6, &mut io);
        io.ticks(&mut e, 6, 6);
        assert_eq!(io.unstuck_calls.len(), 1);
        assert!(io.status.contains("\"failed\": 1"), "{}", io.status);

        // Nav2 gives up in open space (unreachable, not stuck): skipped, no call.
        let mut io = TestIo { unstuck_server: true, ..Default::default() };
        let mut e = exploring_unstuck(&mut io, true);
        e.on_result(io.goals[0].seq, 6, &mut io);
        assert!(io.unstuck_calls.is_empty());

        // On but the action server is missing: carries on without it (one warning).
        let mut io = TestIo::default();
        let mut e = exploring_unstuck(&mut io, true);
        io.pose = (0.15, 1.05);
        e.on_result(io.goals[0].seq, 6, &mut io);
        assert!(io.unstuck_calls.is_empty());
        assert_eq!(io.warnings("action server is not available"), 1);
    }

    #[test]
    fn unstuck_skill_on_no_progress_and_its_timeout() {
        // The robot doesn't move for 30 s on an accepted goal: the goal is cancelled and the skill
        // called; it never answers: after 25 s the goal goes back to Nav2 (classical recoveries).
        let mut io = TestIo { unstuck_server: true, ..Default::default() };
        let mut e = exploring_unstuck(&mut io, true);
        let g = io.goals[0].clone();
        io.ticks(&mut e, 2, 32);
        assert!(io.unstuck_calls.is_empty());
        io.ticks(&mut e, 33, 33);
        assert_eq!(io.unstuck_calls, vec![(1, Some((g.x, g.y)))]);
        assert_eq!(io.cancels, vec![g.seq]);
        e.on_unstuck_response(1, true, &mut io);
        io.ticks(&mut e, 34, 58);
        assert_eq!(io.goals.len(), 1);
        io.ticks(&mut e, 59, 59);
        assert_eq!(io.unstuck_cancels, vec![1]);
        assert_eq!(io.goals.len(), 2);
        assert_eq!((io.goals[1].x, io.goals[1].y), (g.x, g.y));
        assert_eq!(io.warnings("did not finish within 25 s"), 1);
        // Still stuck on it: no second call for the same goal.
        e.on_goal_response(io.goals[1].seq, true, &mut io);
        io.ticks(&mut e, 60, 100);
        assert_eq!(io.unstuck_calls.len(), 1);
    }

    /// Answers the last look-around turn: accepted, then finished with `status`.
    fn turn(io: &mut TestIo, e: &mut Explorer, status: i8) {
        let seq = io.spins.last().unwrap().0;
        e.on_spin_response(seq, true, io);
        e.on_spin_result(seq, status, io);
    }

    #[test]
    fn looks_around_at_the_first_spot_and_before_giving_up() {
        let mut io = TestIo { spin_server: true, ..Default::default() };
        let mut e = open_room(&mut io, Params::load(&mut crate::watchdog::Defaults).unwrap());
        io.t = 1.0;
        e.on_command("start", &mut io);
        // Not at the start point (it may be on the dock): straight to the frontier in view.
        io.ticks(&mut e, 1, 1);
        assert!(io.spins.is_empty());
        assert_eq!(io.goals.len(), 1);
        // Reached: four quarter turns there, even with nothing left in view.
        e.on_goal_response(io.goals[0].seq, true, &mut io);
        io.pose = (io.goals[0].x, io.goals[0].y);
        e.on_result(io.goals[0].seq, STATUS_SUCCEEDED, &mut io);
        for k in 0..4 {
            io.ticks(&mut e, 2 + k, 2 + k);
            assert_eq!(io.spins.len() as u32, k + 1);
            assert_eq!(io.spins[k as usize].1, std::f32::consts::FRAC_PI_2);
            turn(&mut io, &mut e, STATUS_SUCCEEDED);
        }
        io.ticks(&mut e, 6, 7);
        assert_eq!(io.spins.len(), 4, "looked around there once");
        assert_eq!(io.goals.len(), 1);
        // Started again where nothing is in view: it looks around before giving up, stopping at
        // the first turn that is blocked.
        io.t = 8.0;
        e.on_command("stop", &mut io);
        e.on_command("start", &mut io);
        io.ticks(&mut e, 8, 8);
        assert_eq!(io.spins.len(), 5);
        assert!(io.status.contains("looking around"), "{}", io.status);
        turn(&mut io, &mut e, 6);
        assert_eq!(io.warnings("stopped early"), 1);
        // A turn that is never answered is given up on too, and a stop cancels a turn.
        io.t = 9.0;
        e.on_command("stop", &mut io);
        e.on_command("start", &mut io);
        io.ticks(&mut e, 9, 9);
        assert_eq!(io.spins.len(), 6);
        let seq = io.spins.last().unwrap().0;
        io.ticks(&mut e, 10, 19);
        assert_eq!(io.warnings("did not answer the turn"), 0);
        io.ticks(&mut e, 20, 20);
        assert_eq!(io.warnings("did not answer the turn"), 1);
        e.on_spin_response(seq, true, &mut io);
        assert_eq!(io.spin_cancels, vec![seq], "a late acceptance is cancelled");
    }

    #[test]
    fn finishes_back_at_the_start_point() {
        let mut io = TestIo::default();
        let mut e = exploring(&mut io);
        let home = io.pose;
        // The frontier is reached; nothing is left from there: back home and check again there.
        let g = io.goals[0].clone();
        e.on_goal_response(g.seq, true, &mut io);
        io.pose = (g.x, g.y);
        e.on_result(g.seq, STATUS_SUCCEEDED, &mut io);
        io.ticks(&mut e, 2, 23);
        assert_eq!(e.state(), State::Returning);
        let trip = io.goals.last().unwrap().clone();
        assert_eq!((trip.x, trip.y), home);
        e.on_goal_response(trip.seq, true, &mut io);
        io.pose = home;
        e.on_result(trip.seq, STATUS_SUCCEEDED, &mut io);
        assert_eq!(e.state(), State::Exploring);
        io.ticks(&mut e, 24, 45);
        assert_eq!(e.state(), State::Done);
        assert!(io.status.contains("no reachable unmapped areas left - back at the start point"), "{}", io.status);
        // Already at the start point when nothing is left: no pointless trip, no second wait.
        let goals = io.goals.len();
        e.on_command("start", &mut io);
        io.ticks(&mut e, 46, 46);
        let g = io.goals.last().unwrap().clone();
        e.on_goal_response(g.seq, true, &mut io);
        e.on_result(g.seq, STATUS_SUCCEEDED, &mut io); // looked from home: the spot is visited
        io.ticks(&mut e, 47, 70);
        assert_eq!(e.state(), State::Done);
        assert_eq!(io.goals.len(), goals + 1, "only the frontier goal");
        assert!(io.status.contains("back at the start point"), "{}", io.status);
    }

    #[test]
    fn trip_home_failing_is_reported() {
        let mut io = TestIo::default();
        let mut e = exploring(&mut io);
        let g = io.goals[0].clone();
        e.on_goal_response(g.seq, true, &mut io);
        io.pose = (g.x, g.y);
        e.on_result(g.seq, STATUS_SUCCEEDED, &mut io);
        io.ticks(&mut e, 2, 23);
        // Each failed trip is followed by another check from where it is and another trip, up to
        // return_home_attempts (3) in all.
        for k in 0..3 {
            let trip = io.goals.last().unwrap().clone();
            assert_eq!((trip.x, trip.y), (0.35, 1.05), "trip {k}");
            e.on_goal_response(trip.seq, false, &mut io);
            assert_eq!(e.state(), State::Exploring, "checks again from where it is");
            io.ticks(&mut e, 24 + 22 * k, 45 + 22 * k);
        }
        assert_eq!(e.state(), State::Done);
        assert_eq!(io.goals.len(), 4, "the frontier and three trips home");
        assert!(io.status.contains("no reachable unmapped areas left - could not drive back to the start point"), "{}", io.status);
    }

    #[test]
    fn marks_leave_the_robot_free() {
        let mut io = TestIo::default();
        let mut e = exploring(&mut io);
        e.on_sensors(&Sensors { bump_left: true, ..Default::default() }, &mut io);
        assert!(io.zones[0].ends_with(r#""ttl_s": 60.0, "clear": [0.35, 1.05, 0.28], "soft": 0.22}"#), "{}", io.zones[0]);
        e.on_sensors(&Sensors { cliff: [false, true, false, false], ..Default::default() }, &mut io);
        e.on_mode("passive", &mut io);
        e.cliff_check(&mut io);
        assert!(io.zones[1].ends_with(r#""clear": [0.35, 1.05, 0.28], "soft": 0.22}"#), "{}", io.zones[1]);
    }

    #[test]
    fn failed_spot_reopens_when_the_map_opens_up() {
        let mut io = TestIo::default();
        let mut e = exploring(&mut io);
        let g = io.goals[0].clone();
        e.on_goal_response(g.seq, true, &mut io);
        // Aborted while something blocked the way (cells next to the goal lethal at that moment).
        let mut blocked = e.grid.clone().unwrap();
        for r in 3..=9 {
            for c in 5..=11 {
                blocked.data[r * 20 + c] = 100;
            }
        }
        let clear = e.grid.replace(blocked).unwrap();
        e.on_result(g.seq, 6, &mut io);
        io.ticks(&mut e, 2, 3);
        assert_eq!(io.goals.len(), 1, "skipped while blocked");
        e.on_grid(clear);
        io.ticks(&mut e, 4, 4);
        assert_eq!(io.goals.len(), 2, "tried again once the way opened up");
        assert!(io.logs.iter().any(|(_, m)| m.starts_with("The map opened up around an area that failed earlier")));
    }

    #[test]
    fn spot_skipped_for_bumps_waits_for_the_retry_before_finishing() {
        let mut io = TestIo::default();
        let mut e = exploring(&mut io);
        let g = io.goals[0].clone();
        e.on_goal_response(g.seq, true, &mut io);
        let mut blocked = e.grid.clone().unwrap();
        for r in 3..=9 {
            for c in 5..=11 {
                blocked.data[r * 20 + c] = 100;
            }
        }
        let clear = e.grid.replace(blocked).unwrap();
        for _ in 0..2 {
            e.on_sensors(&Sensors { bump_left: true, ..Default::default() }, &mut io);
            e.on_sensors(&Sensors::default(), &mut io);
        }
        assert_eq!(io.warnings("skipping it"), 1);
        // The camera can't tell whether what it bumped into is gone: the map opening up there
        // doesn't bring it back.
        e.on_grid(clear);
        io.ticks(&mut e, 2, 4);
        assert_eq!(io.goals.len(), 1);
        assert!(!io.logs.iter().any(|(_, m)| m.starts_with("The map opened up")));
    }

    #[test]
    fn markers_show_how_it_chose_the_goal() {
        let mut io = TestIo { want_markers: true, ..Default::default() };
        let mut e = exploring(&mut io);
        let m = io.markers.last().unwrap().clone();
        let count = |m: &[MarkerSpec], ns: &str| m.iter().filter(|s| s.ns == ns).count();
        assert_eq!((count(&m, "frontiers"), count(&m, "options"), count(&m, "home")), (1, 1, 1));
        assert_eq!(m.iter().find(|s| s.ns == "frontiers").unwrap().points.len(), 18, "column 12, rows 1-18");
        let arrow = m.iter().find(|s| s.ns == "goal").unwrap();
        assert_eq!(arrow.points, vec![[0.35, 1.05, 0.1], [io.goals[0].x, io.goals[0].y, 0.1]], "from where it was chosen");
        assert!(m.iter().any(|s| s.ns == "labels" && s.text.starts_with("chosen: ")), "{m:?}");
        // Unchanged: not sent again.
        let sent = io.markers.len();
        io.ticks(&mut e, 2, 3);
        assert_eq!(io.markers.len(), sent);
        // The goal fails: the spot shows as failed, and as skipped in the next search.
        e.on_goal_response(io.goals[0].seq, true, &mut io);
        e.on_result(io.goals[0].seq, 6, &mut io);
        io.ticks(&mut e, 4, 4);
        let m = io.markers.last().unwrap();
        assert_eq!(count(m, "failed"), 1);
        assert!(m.iter().any(|s| s.ns == "labels" && s.text == "failed earlier"));
        assert!(m.iter().any(|s| s.ns == "labels" && s.text == "failed: retry if the map opens up"));
        // Nobody listening: nothing built; listening again: sent again.
        io.want_markers = false;
        io.ticks(&mut e, 5, 6);
        assert_eq!(io.markers.len(), sent + 1);
        io.want_markers = true;
        io.ticks(&mut e, 7, 7);
        assert_eq!(io.markers.len(), sent + 2);
    }

    #[test]
    fn status_comes_back_after_a_tracking_pause() {
        let mut io = TestIo::default();
        let mut e = exploring(&mut io);
        e.on_filter(r#"{"state": "recovering"}"#);
        io.ticks(&mut e, 2, 2);
        assert!(io.status.contains("Paused"), "{}", io.status);
        e.on_filter(r#"{"state": "idle"}"#);
        io.ticks(&mut e, 3, 3);
        assert!(io.status.contains("Heading to an unmapped area"), "{}", io.status);
    }

    #[test]
    fn turns_to_face_the_unknown_at_the_spot() {
        let mut io = TestIo { spin_server: true, ..Default::default() };
        let mut p = Params::load(&mut crate::watchdog::Defaults).unwrap();
        p.look_steps = 0;
        let mut e = open_room(&mut io, p);
        io.t = 1.0;
        e.on_command("start", &mut io);
        io.ticks(&mut e, 1, 1);
        let g = io.goals[0].clone();
        e.on_goal_response(g.seq, true, &mut io);
        // Arrives facing away from the unknown (east): one turn of the difference.
        io.pose = (g.x, g.y);
        io.yaw = 2.5;
        e.on_result(g.seq, STATUS_SUCCEEDED, &mut io);
        io.ticks(&mut e, 2, 2);
        assert_eq!(io.spins.len(), 1);
        let want = (2.0 * g.qz.atan2(g.qw) - 2.5) as f32;
        assert!((io.spins[0].1 - want).abs() < 1e-6, "{:?} vs {want}", io.spins);
        assert!(io.status.contains("Turning to face"), "{}", io.status);
    }

    #[test]
    fn bump_marks_are_dropped_before_finishing() {
        let mut io = TestIo::default();
        let mut e = exploring(&mut io);
        let g = io.goals[0].clone();
        e.on_goal_response(g.seq, true, &mut io);
        // Someone walks into the robot on the way, and the goal fails.
        e.on_sensors(&Sensors { bump_left: true, ..Default::default() }, &mut io);
        e.on_sensors(&Sensors::default(), &mut io);
        e.on_result(g.seq, 6, &mut io);
        io.ticks(&mut e, 2, 23);
        assert_eq!(io.zones.len(), 2, "{:?}", io.zones);
        assert_eq!(io.zones[1], r#"{"op": "delete", "name": "bump_000000_1"}"#);
        assert!(io.logs.iter().any(|(_, m)| m.starts_with("Removed 1 bump mark(s) (a person walking into the robot")));
        // Bumped there again: marked again.
        e.on_sensors(&Sensors { bump_left: true, ..Default::default() }, &mut io);
        assert_eq!(io.zones.len(), 3);
    }

    #[test]
    fn railing_marks_stay_for_the_next_trip_home() {
        let mut io = TestIo::default();
        let mut e = exploring_with(&mut io, |p| p.bump_mark_short_ttl = 0.0);
        let g = io.goals[0].clone();
        e.on_goal_response(g.seq, true, &mut io);
        io.pose = (g.x, g.y);
        e.on_result(g.seq, STATUS_SUCCEEDED, &mut io);
        // Two hits side by side there: a railing the camera misses, not someone walking by.
        for dx in [0.0, 0.25] {
            io.pose = (g.x + dx, g.y);
            e.on_sensors(&Sensors { bump_left: true, ..Default::default() }, &mut io);
            e.on_sensors(&Sensors::default(), &mut io);
        }
        assert_eq!(io.zones.len(), 2);
        io.ticks(&mut e, 2, 23);
        let deletes = |io: &TestIo| io.zones.iter().filter(|z| z.contains(r#""op": "delete""#)).count();
        for k in 0..3 {
            // Kept while finishing and for the second trip home (they route it around the
            // railing), dropped before the last one.
            assert_eq!(deletes(&io), if k < 2 { 0 } else { 2 }, "trip {k}");
            let trip = io.goals.last().unwrap().clone();
            assert_eq!((trip.x, trip.y), (0.35, 1.05), "trip {k}");
            e.on_goal_response(trip.seq, false, &mut io);
            io.ticks(&mut e, 24 + 22 * k, 45 + 22 * k);
        }
        assert_eq!(e.state(), State::Done);
    }

    #[test]
    fn bump_with_a_person_in_front_is_not_marked() {
        let mut io = TestIo::default();
        let mut e = exploring(&mut io);
        let g = io.goals[0].clone();
        e.on_goal_response(g.seq, true, &mut io);
        // Someone 0.5 m ahead, seen a second before the hit: a leg, not a railing.
        e.on_persons("base_link", &[[0.5, 0.0, 0.8]], &mut io);
        io.t = 2.0;
        e.on_sensors(&Sensors { bump_left: true, ..Default::default() }, &mut io);
        e.on_sensors(&Sensors::default(), &mut io);
        assert!(io.zones.is_empty(), "{:?}", io.zones);
        assert!(io.logs.iter().any(|(_, m)| m == "Bumper hit at +29 deg with a person in front: not marked"), "{:?}", io.logs);
        // Someone behind it, or seen too long ago: marked, and counted towards skipping the goal.
        io.t = 6.0;
        e.on_persons("base_link", &[[-0.5, 0.0, 0.8]], &mut io);
        e.on_sensors(&Sensors { bump_left: true, ..Default::default() }, &mut io);
        assert_eq!(io.zones.len(), 1);
        assert!(io.cancels.is_empty(), "one bump counted, not two");
    }

    #[test]
    fn bump_mark_is_short_lived_unless_bumped_again() {
        let mut io = TestIo::default();
        let mut e = exploring(&mut io);
        e.on_sensors(&Sensors { bump_left: true, ..Default::default() }, &mut io);
        e.on_sensors(&Sensors::default(), &mut io);
        assert!(io.zones[0].contains(r#""ttl_s": 60.0"#), "{}", io.zones[0]);
        assert!(io.logs.iter().any(|(_, m)| m.ends_with("marked bump_000000_1 for 60 s")), "{:?}", io.logs);
        // The same spot again: that mark now lasts 10 min (sent again); no second mark.
        io.t = 20.0;
        e.on_sensors(&Sensors { bump_left: true, ..Default::default() }, &mut io);
        assert_eq!(io.zones.len(), 3, "{:?}", io.zones);
        assert_eq!(io.zones[1], r#"{"op": "delete", "name": "bump_000000_1"}"#);
        assert!(io.zones[2].contains(r#""ttl_s": 600.0"#), "{}", io.zones[2]);
        // ... and keeps planned paths 0.1 m off: the robot's radius + 0.1 m (+ 1 cm) of margin.
        assert!(io.zones[2].contains(r#""margin": 0.28"#), "{}", io.zones[2]);
        assert!(io.logs.iter().any(|(_, m)| m == "Bumped at bump_000000_1 again: keeping it for 10 min"));
        // A hit beside it (turned 1.2 rad at the same spot: someone's legs) is another short mark.
        e.on_sensors(&Sensors::default(), &mut io);
        io.yaw = 1.2;
        io.t = 25.0;
        e.on_sensors(&Sensors { bump_left: true, ..Default::default() }, &mut io);
        assert_eq!(io.zones.len(), 4, "{:?}", io.zones);
        assert!(io.zones[3].contains(r#""ttl_s": 60.0"#), "{}", io.zones[3]);
    }

    #[test]
    fn boxed_in_by_its_own_marks_it_removes_them() {
        let mut io = TestIo::default();
        let mut e = exploring(&mut io);
        let g = io.goals[0].clone();
        e.on_goal_response(g.seq, true, &mut io);
        e.on_result(g.seq, 6, &mut io);
        // Three hits around the robot (people's legs, as at the stair head in Isaac run 9) ...
        for (k, yaw) in [0.0, 2.0, 4.0].into_iter().enumerate() {
            io.yaw = yaw;
            io.t = 2.0 + k as f64;
            e.on_sensors(&Sensors { bump_left: true, ..Default::default() }, &mut io);
            e.on_sensors(&Sensors::default(), &mut io);
        }
        assert_eq!(io.zones.len(), 3);
        // ... and the costmap closed around it (a ring 2 cells out), the rest of the room free.
        let mut boxed = e.grid.clone().unwrap();
        for r in 0..20i64 {
            for c in 0..13i64 {
                if (r - 10).abs().max((c - 3).abs()) == 2 {
                    boxed.data[(r * 20 + c) as usize] = 100;
                }
            }
        }
        e.on_grid(boxed);
        io.ticks(&mut e, 10, 10);
        assert_eq!(io.zones.len(), 4);
        assert_eq!(io.zones[3], r#"{"op": "delete", "name": "bump_000000_1"}"#, "the oldest first");
        assert!(io.logs.iter().any(|(_, m)| m == "Boxed in by its own bump marks (0.1 m2 to move in): removed bump_000000_1"));
        io.ticks(&mut e, 11, 12);
        assert_eq!(io.zones.len(), 4, "one every trap_wait_s");
        io.ticks(&mut e, 13, 13);
        assert_eq!(io.zones[4], r#"{"op": "delete", "name": "bump_000000_2"}"#);
    }

    #[test]
    fn drop_offs_keep_frontiers_and_goals_out() {
        let mut io = TestIo::default();
        let mut p = Params::load(&mut crate::watchdog::Defaults).unwrap();
        p.look_steps = 0;
        let mut e = open_room(&mut io, p);
        // Drop points seen along x = 0.85 m (the map shows floor beyond, up to the frontier at 1.2 m),
        // relative to the robot at (0.35, 1.05).
        let pts: Vec<[f32; 3]> = (1..19).map(|r| [0.5, (r as f64 * 0.1 + 0.05 - 1.05) as f32, 0.1]).collect();
        e.on_drops("camera", &pts, &mut io);
        e.on_drops("camera", &pts, &mut io);
        assert_eq!(e.drop_cells().len(), 18);
        io.t = 1.0;
        e.on_command("start", &mut io);
        e.tick(&mut io);
        // Beyond the drop the frontier can't be reached (the spot to look at it from is where it
        // stands): no goal into the void.
        assert!(io.goals.is_empty(), "{:?}", io.goals);
        assert!(io.status.contains("No reachable unmapped areas"), "{}", io.status);
        // Without the drops it heads for the frontier past x = 0.85.
        let mut io2 = TestIo::default();
        let mut p2 = Params::load(&mut crate::watchdog::Defaults).unwrap();
        p2.look_steps = 0;
        let mut e2 = open_room(&mut io2, p2);
        io2.t = 1.0;
        e2.on_command("start", &mut io2);
        e2.tick(&mut io2);
        assert!(io2.goals[0].x > 0.85, "{:?}", io2.goals[0]);
    }

    #[test]
    fn remembered_drops_are_sent_every_second_for_the_costmap() {
        // A costmap clear (Nav2's recoveries) wipes the drop_layer: every tick, exploring or not,
        // the explorer sends each drop-off cell it remembers, so the layer is marked again.
        let mut io = TestIo::default();
        let mut p = Params::load(&mut crate::watchdog::Defaults).unwrap();
        p.look_steps = 0;
        let mut e = open_room(&mut io, p);
        io.ticks(&mut e, 1, 1);
        assert!(io.drops.is_empty(), "nothing remembered yet");
        let pts: Vec<[f32; 3]> = (1..19).map(|r| [0.5, (r as f64 * 0.1 + 0.05 - 1.05) as f32, 0.1]).collect();
        e.on_drops("camera", &pts, &mut io);
        io.ticks(&mut e, 2, 2);
        assert!(io.drops.is_empty(), "a cell counts after drop_min_hits clouds");
        e.on_drops("camera", &pts, &mut io);
        io.ticks(&mut e, 3, 5);
        assert_eq!(io.drops.len(), 3);
        assert!(io.drops.iter().all(|d| d == &io.drops[0]));
        let first = &io.drops[0];
        assert_eq!(first.len(), 18);
        assert!(first.iter().all(|q| q[0] == first[0][0] && q[2] == 0.0), "{first:?}");
    }

    #[test]
    fn look_around_pauses_between_turns() {
        let mut io = TestIo { spin_server: true, ..Default::default() };
        let mut p = Params::load(&mut crate::watchdog::Defaults).unwrap();
        p.look_pause = 1.5;
        let mut e = open_room(&mut io, p);
        io.t = 1.0;
        e.on_command("start", &mut io);
        io.ticks(&mut e, 1, 1);
        e.on_goal_response(io.goals[0].seq, true, &mut io);
        io.pose = (io.goals[0].x, io.goals[0].y);
        e.on_result(io.goals[0].seq, STATUS_SUCCEEDED, &mut io);
        io.ticks(&mut e, 2, 2);
        assert_eq!(io.spins.len(), 1);
        turn(&mut io, &mut e, STATUS_SUCCEEDED);
        io.ticks(&mut e, 3, 3);
        assert_eq!(io.spins.len(), 1, "standing still for 1.5 s after the turn");
        io.ticks(&mut e, 4, 4);
        assert_eq!(io.spins.len(), 2);
    }

    #[test]
    fn no_way_through_diagonal_gaps_between_lethal_cells() {
        // A wall with only diagonal gaps between its cells (column 8 on even rows, 9 on odd ones):
        // Nav2's planner can't get through, so the frontier beyond (column 15) is out of reach.
        let (w, h) = (20usize, 10usize);
        let house = |zigzag: i8| {
            let mut grid = vec![0i8; w * h];
            for r in 0..h {
                for c in 0..w {
                    grid[r * w + c] = if c >= 16 {
                        -1
                    } else if r == 0 || r == h - 1 || c == 0 {
                        100
                    } else if c == 8 + r % 2 {
                        zigzag
                    } else {
                        0
                    };
                }
            }
            grid
        };
        let p = FrontierParams {
            free_max: 40,
            pass_max: 98,
            clear_max: 15,
            min_cells: 5,
            standoff_cells: 3.0,
            search_cells: 16.0,
            cost_weight: 0.05,
            peek_cells: 0.0,
            split_cells: 0.0,
        };
        let (goals, _, _, reach) = frontier_goals(&house(100), w, h, (5, 2), &p);
        assert!(goals.is_empty());
        assert_eq!(reach, 60, "rows 1-8 of columns 1-7, and the four pockets at column 8");
        // Inscribed (not lethal) corners: it steps through diagonally, as it did before.
        let (goals, _, _, reach) = frontier_goals(&house(99), w, h, (5, 2), &p);
        assert!(!goals.is_empty());
        assert_eq!(reach, 112, "rows 1-8 of columns 1-15 but the zigzag");
    }

    #[test]
    fn unanswered_goal_request_is_given_up() {
        let mut io = TestIo::default();
        let mut e = exploring(&mut io);
        let first = io.goals[0].seq;
        io.ticks(&mut e, 2, 11);
        assert_eq!(io.warnings("did not answer"), 0, "waits goal_response_timeout_s");
        io.ticks(&mut e, 12, 12);
        assert_eq!(io.warnings("did not answer"), 1);
        assert!(io.status.contains(r#""goal": null"#));
        // Nav2 accepts it after all: cancelled at once, nothing else changes.
        e.on_goal_response(first, true, &mut io);
        assert_eq!(io.cancels, vec![first]);
        // The spot was blacklisted like a rejected goal; with nothing else left it's retried
        // (retry_failed_before_finish) once finish_confirm_s has passed: exploring goes on.
        io.ticks(&mut e, 13, 34);
        assert_eq!(io.goals.len(), 2);
        assert_ne!(io.goals[1].seq, first);
        assert_eq!((io.goals[1].x, io.goals[1].y), (io.goals[0].x, io.goals[0].y));
        assert_eq!(e.state(), State::Exploring);
    }

    #[test]
    fn trip_home_without_answer_or_result_fails() {
        let mut io = TestIo::default();
        let mut e = exploring(&mut io);
        e.on_command("home", &mut io);
        assert_eq!(e.state(), State::Returning);
        io.ticks(&mut e, 2, 11);
        assert_eq!(e.state(), State::Returning);
        io.ticks(&mut e, 12, 12);
        assert_eq!(e.state(), State::Done, "request never answered: the trip failed");
        assert!(io.status.contains("Returning to the start point - could not drive back to the start point"));
        // Accepted this time, but the result never comes: given up after return_timeout_s.
        e.on_command("start", &mut io);
        io.t = 20.0;
        e.on_command("home", &mut io);
        let home = io.goals.last().unwrap().seq;
        e.on_goal_response(home, true, &mut io);
        io.ticks(&mut e, 21, 320);
        assert_eq!(e.state(), State::Returning);
        io.ticks(&mut e, 321, 321);
        assert_eq!(e.state(), State::Done);
        assert_eq!(io.cancels, vec![home]);
        assert_eq!(io.warnings("timed out after 300 s"), 1);
        // Rejected: the trip fails at once (it used to stay "returning" with no goal).
        e.on_command("start", &mut io);
        e.on_command("home", &mut io);
        e.on_goal_response(io.goals.last().unwrap().seq, false, &mut io);
        assert_eq!(e.state(), State::Done);
    }
}
