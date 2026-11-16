//! Outcome measurements of a twin run (coverage over time, distance, bumps, stuck episodes, Nav2
//! goals and recoveries, explorer status, battery) and the trajectory / coverage plot.

use std::collections::BTreeMap;

use serde_json::{json, Value};

use super::mapper::Mapper;
use super::world::{Kind, Shape, World, P2};

#[derive(Clone, Debug)]
pub struct Sample {
    pub t: f64,
    pub pose: [f64; 3],
    pub coverage: f64,
    pub reachable: f64,
    pub battery: Option<f64>,
    pub explorer: String,
    /// A Nav2 goal was being pursued (accepted or executing).
    pub goal_active: bool,
}

#[derive(Clone, Debug)]
pub struct Event {
    pub t: f64,
    pub kind: String,
    pub pos: P2,
    pub detail: String,
}

#[derive(Default)]
pub struct Metrics {
    pub samples: Vec<Sample>,
    pub events: Vec<Event>,
    /// True path, a point every ~2 cm.
    pub path: Vec<P2>,
    pub counts: BTreeMap<String, u64>,
}

/// Explorer states in which the robot is meant to be moving.
pub fn driving_state(s: &str) -> bool {
    matches!(s, "exploring" | "returning")
}

impl Metrics {
    pub fn count(&mut self, key: &str) {
        *self.counts.entry(key.to_string()).or_default() += 1;
    }

    pub fn event(&mut self, t: f64, kind: &str, pos: P2, detail: impl Into<String>) {
        self.events.push(Event { t, kind: kind.into(), pos, detail: detail.into() });
    }

    pub fn track(&mut self, p: P2) {
        if self.path.last().is_none_or(|q| (q[0] - p[0]).hypot(q[1] - p[1]) >= 0.02) {
            self.path.push(p);
        }
    }

    /// Stretches of at least `window` seconds, while the explorer was driving, in which the robot
    /// stayed within `radius` of where the stretch began: (start, end, position).
    pub fn stuck_episodes(&self, window: f64, radius: f64) -> Vec<(f64, f64, P2)> {
        let mut out = Vec::new();
        let mut anchor: Option<&Sample> = None;
        let mut last_t = 0.0;
        for s in &self.samples {
            let active = driving_state(&s.explorer);
            let inside = anchor.is_some_and(|a| (s.pose[0] - a.pose[0]).hypot(s.pose[1] - a.pose[1]) <= radius);
            if active && inside {
                last_t = s.t;
                continue;
            }
            if let Some(a) = anchor {
                if last_t - a.t >= window {
                    out.push((a.t, last_t, [a.pose[0], a.pose[1]]));
                }
            }
            anchor = active.then_some(s);
            last_t = s.t;
        }
        if let Some(a) = anchor {
            if last_t - a.t >= window {
                out.push((a.t, last_t, [a.pose[0], a.pose[1]]));
            }
        }
        out
    }

    /// First time the coverage reached `frac` of the free floor.
    pub fn time_to(&self, frac: f64) -> Option<f64> {
        self.samples.iter().find(|s| s.coverage >= frac).map(|s| s.t)
    }

    /// Splits [`Metrics::stuck_episodes`] into (stuck: a Nav2 goal was active most of the time but
    /// the robot made no progress, idle: the explorer was "exploring" without a goal, e.g. waiting
    /// to confirm that nothing is left).
    pub fn stuck_and_idle(&self, window: f64, radius: f64) -> (Vec<(f64, f64, P2)>, Vec<(f64, f64, P2)>) {
        let (mut stuck, mut idle) = (Vec::new(), Vec::new());
        for e in self.stuck_episodes(window, radius) {
            let inside: Vec<&Sample> = self.samples.iter().filter(|s| s.t >= e.0 && s.t <= e.1).collect();
            let active = inside.iter().filter(|s| s.goal_active).count();
            if active * 2 >= inside.len().max(1) {
                stuck.push(e);
            } else {
                idle.push(e);
            }
        }
        (stuck, idle)
    }

    pub fn timeline_csv(&self) -> String {
        let mut out = String::from("t_s,x,y,yaw_deg,coverage,coverage_reachable,battery,explorer\n");
        for s in &self.samples {
            out.push_str(&format!(
                "{:.1},{:.3},{:.3},{:.1},{:.4},{:.4},{},{}\n",
                s.t,
                s.pose[0],
                s.pose[1],
                s.pose[2].to_degrees(),
                s.coverage,
                s.reachable,
                s.battery.map_or(String::new(), |b| format!("{b:.3}")),
                s.explorer
            ));
        }
        out
    }

    pub fn events_jsonl(&self) -> String {
        self.events
            .iter()
            .map(|e| json!({"t": round(e.t, 2), "kind": e.kind, "x": round(e.pos[0], 3), "y": round(e.pos[1], 3), "detail": e.detail}).to_string() + "\n")
            .collect()
    }

    pub fn events_of(&self, kind: &str) -> Vec<Value> {
        self.events
            .iter()
            .filter(|e| e.kind == kind)
            .map(|e| json!({"t": round(e.t, 1), "x": round(e.pos[0], 2), "y": round(e.pos[1], 2), "detail": e.detail}))
            .collect()
    }

    /// Coverage every `step` seconds: [[t, coverage], ...].
    pub fn coverage_series(&self, step: f64) -> Vec<Value> {
        let mut out = Vec::new();
        let mut next = 0.0;
        for s in &self.samples {
            if s.t >= next {
                out.push(json!([round(s.t, 0), round(s.coverage, 4)]));
                next = s.t + step;
            }
        }
        if let Some(s) = self.samples.last() {
            out.push(json!([round(s.t, 0), round(s.coverage, 4)]));
        }
        out
    }
}

pub fn round(x: f64, places: i32) -> f64 {
    let s = 10f64.powi(places);
    // + 0.0: an empty float sum is -0.0
    (x * s).round() / s + 0.0
}

// ---------------------------------------------------------------- the plot

/// Colours (B, G, R) of the plot, also listed in result.json.
pub const LEGEND: &[(&str, [u8; 3])] = &[
    ("wall (true floor plan)", [70, 70, 70]),
    ("mapped free", [255, 255, 255]),
    ("true free floor never mapped", [190, 190, 250]),
    ("mapped obstacle", [0, 0, 0]),
    ("unknown outside the house", [150, 150, 150]),
    ("trajectory (blue -> green over time)", [200, 80, 0]),
    ("start", [0, 170, 0]),
    ("end", [0, 0, 220]),
    ("bump", [0, 140, 255]),
    ("other contact (side/rear/person)", [0, 220, 220]),
    ("stuck episode", [200, 0, 200]),
    ("door (filled = closed at the end)", [30, 70, 120]),
    ("railing", [160, 30, 120]),
    ("obstacle", [0, 110, 200]),
    ("person path", [220, 200, 0]),
    ("cliff", [0, 0, 255]),
];

fn legend(name: &str) -> [u8; 3] {
    LEGEND.iter().find(|l| l.0.starts_with(name)).map_or([255, 0, 255], |l| l.1)
}

struct Canvas {
    w: usize,
    h: usize,
    scale: usize,
    origin: P2,
    res: f64,
    px: Vec<u8>,
}

impl Canvas {
    fn to_px(&self, p: P2) -> (i64, i64) {
        let x = ((p[0] - self.origin[0]) / self.res * self.scale as f64).floor() as i64;
        let y = self.h as i64 - 1 - ((p[1] - self.origin[1]) / self.res * self.scale as f64).floor() as i64;
        (x, y)
    }

    fn set(&mut self, x: i64, y: i64, c: [u8; 3]) {
        if x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h {
            let k = (y as usize * self.w + x as usize) * 4;
            self.px[k..k + 4].copy_from_slice(&[c[0], c[1], c[2], 255]);
        }
    }

    fn dot(&mut self, x: i64, y: i64, r: i64, c: [u8; 3]) {
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r * r {
                    self.set(x + dx, y + dy, c);
                }
            }
        }
    }

    fn line(&mut self, a: P2, b: P2, thick: i64, c: [u8; 3]) {
        let (x0, y0) = self.to_px(a);
        let (x1, y1) = self.to_px(b);
        for (x, y) in super::mapper::bresenham(x0, y0, x1, y1) {
            self.dot(x, y, thick, c);
        }
    }

    fn circle(&mut self, p: P2, r_m: f64, c: [u8; 3]) {
        let n = 48;
        for k in 0..n {
            let (a0, a1) = (k as f64 / n as f64 * std::f64::consts::TAU, (k + 1) as f64 / n as f64 * std::f64::consts::TAU);
            self.line([p[0] + r_m * a0.cos(), p[1] + r_m * a0.sin()], [p[0] + r_m * a1.cos(), p[1] + r_m * a1.sin()], 0, c);
        }
    }

    fn cross(&mut self, p: P2, size: i64, c: [u8; 3]) {
        let (x, y) = self.to_px(p);
        for d in -size..=size {
            self.dot(x + d, y + d, 1, c);
            self.dot(x + d, y - d, 1, c);
        }
    }

    fn shape(&mut self, s: &Shape, fill: bool, c: [u8; 3]) {
        match s {
            Shape::Circle { c: ctr, r } => self.circle(*ctr, *r, c),
            Shape::Capsule { a, b, r } => {
                let thick = ((r / self.res * self.scale as f64).round() as i64).max(1);
                self.line(*a, *b, thick, c)
            }
            Shape::Polygon(pts) => {
                for k in 0..pts.len() {
                    self.line(pts[k], pts[(k + 1) % pts.len()], 0, c);
                }
            }
        }
        if fill {
            let bb = s.bbox();
            let (x0, y1) = self.to_px([bb[0], bb[1]]);
            let (x1, y0) = self.to_px([bb[2], bb[3]]);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let p = [
                        self.origin[0] + (x as f64 + 0.5) / self.scale as f64 * self.res,
                        self.origin[1] + ((self.h as i64 - 1 - y) as f64 + 0.5) / self.scale as f64 * self.res,
                    ];
                    if s.contains(p) {
                        self.set(x, y, c);
                    }
                }
            }
        }
    }
}

/// Trajectory and coverage plot (PNG): the map against the true floor plan, the path, bumps,
/// stuck episodes and the scenario's objects. `people_paths`: each person's walking path.
pub fn plot_png(m: &Metrics, mapper: &Mapper, world: &World, people_paths: &[Vec<P2>], stuck: &[(f64, f64, P2)]) -> Option<Vec<u8>> {
    let scale = 3;
    let g = &world.grid;
    let (w, h) = (g.width * scale, g.height * scale);
    let mut cv = Canvas { w, h, scale, origin: g.origin, res: g.res, px: vec![0; w * h * 4] };
    let occ = mapper.occupancy();
    for j in 0..g.height {
        for i in 0..g.width {
            let k = j * g.width + i;
            let c = if occ[k] == 100 {
                legend("mapped obstacle")
            } else if occ[k] == 0 {
                legend("mapped free")
            } else if mapper.truth_free(k) {
                legend("true free floor never mapped")
            } else if g.solid[k] {
                legend("unknown outside")
            } else {
                legend("unknown outside")
            };
            let c = if g.solid[k] && occ[k] != 100 && occ[k] != 0 && is_wall_edge(g, i, j) { legend("wall") } else { c };
            for dy in 0..scale {
                for dx in 0..scale {
                    cv.set((i * scale + dx) as i64, (h - 1 - (j * scale + dy)) as i64, c);
                }
            }
        }
    }
    for (_, s) in &world.cliffs {
        cv.shape(s, false, legend("cliff"));
    }
    for path in people_paths {
        for w2 in path.windows(2) {
            cv.line(w2[0], w2[1], 0, legend("person path"));
        }
    }
    for b in &world.bodies {
        match b.kind {
            Kind::Door => cv.shape(&b.shape, b.active, legend("door")),
            Kind::Railing => cv.shape(&b.shape, false, legend("railing")),
            Kind::Obstacle | Kind::Dock => cv.shape(&b.shape, false, legend("obstacle")),
            Kind::Post | Kind::Person => {}
        }
    }
    let n = m.path.len().max(2) as f64;
    for (k, w2) in m.path.windows(2).enumerate() {
        let f = k as f64 / n;
        let c = [(200.0 * (1.0 - f)) as u8, (80.0 + 120.0 * f) as u8, 0];
        cv.line(w2[0], w2[1], 0, c);
    }
    for &(_, _, p) in stuck {
        cv.circle(p, 0.3, legend("stuck"));
        cv.circle(p, 0.32, legend("stuck"));
    }
    for e in &m.events {
        match e.kind.as_str() {
            "bump" => cv.cross(e.pos, 4, legend("bump")),
            "contact" => cv.cross(e.pos, 3, legend("other contact")),
            "fell" => cv.cross(e.pos, 10, legend("cliff")),
            _ => {}
        }
    }
    if let Some(p) = m.path.first() {
        let (x, y) = cv.to_px(*p);
        cv.dot(x, y, 6, legend("start"));
    }
    if let Some(p) = m.path.last() {
        let (x, y) = cv.to_px(*p);
        cv.dot(x, y, 5, legend("end"));
    }
    crate::web::image::encode_png_bgra(&cv.px, w, h)
}

fn is_wall_edge(g: &super::world::Grid, i: usize, j: usize) -> bool {
    let (i, j) = (i as i64, j as i64);
    (-1..=1).any(|dj| (-1..=1).any(|di| !g.solid_ij(i + di, j + dj)))
}

// ---------------------------------------------------------------- the explorer's view at the end

/// Nav2's global costmap (nav_msgs/OccupancyGrid layout).
#[derive(Clone, Debug, Default)]
pub struct CostGrid {
    pub width: usize,
    pub height: usize,
    pub res: f64,
    pub origin: P2,
    pub data: Vec<i8>,
}

/// The explorer's frontier parameters (its defaults, as nav.launch.py runs it), with the overrides
/// of an experiment (twin2d.launch.py explorer_params, same names as the explorer's parameters).
pub fn explorer_frontier_params(res: f64, overrides: &serde_json::Map<String, Value>) -> crate::explorer::FrontierParams {
    let f = |k: &str, d: f64| overrides.get(k).and_then(Value::as_f64).unwrap_or(d);
    crate::explorer::FrontierParams {
        free_max: f("free_cost_max", 40.0) as i64,
        pass_max: f("passable_cost_max", 98.0) as i64,
        clear_max: f("goal_cost_max", 15.0) as i64,
        min_cells: f("min_frontier_cells", 10.0) as i64,
        standoff_cells: f("goal_standoff_m", 0.3) / res,
        search_cells: f("goal_search_radius_m", 0.8) / res,
        cost_weight: f("goal_cost_weight", 0.05),
        peek_cells: f("peek_search_m", 3.0) / res,
        split_cells: f("frontier_split_m", 2.0) / res,
    }
}

/// Every frontier group on the final global costmap (any size) and whether the explorer could
/// use it, judged with its own functions and parameters (by default min_frontier_cells 10, goal
/// 0.3 m back from the frontier in cells of cost <= 15 within 0.8 m, reachable through cost <= 98).
/// Also returns a PNG of the costmap with the groups coloured by verdict.
pub fn frontier_report(cm: &CostGrid, robot: P2, overrides: &serde_json::Map<String, Value>) -> (Vec<Value>, Option<Vec<u8>>) {
    use crate::explorer::frontier_goals;
    let (w, h, res) = (cm.width, cm.height, cm.res);
    if w == 0 || h == 0 || cm.data.len() != w * h {
        return (Vec::new(), None);
    }
    let fp = explorer_frontier_params(res, overrides);
    let rc = (((robot[1] - cm.origin[1]) / res) as i64, ((robot[0] - cm.origin[0]) / res) as i64);
    let (goals, seed, peeks, _) = frontier_goals(&cm.data, w, h, rc, &fp);
    // The robot's reachable component, as frontier_goals computes it.
    let passable: Vec<bool> = cm.data.iter().map(|&g| g >= 0 && (g as i64) <= fp.pass_max).collect();
    let reach = match seed {
        Some(s) => crate::explorer::reachable(&cm.data, &passable, w, h, s),
        None => vec![false; w * h],
    };
    let xy = |r: f64, c: f64| [cm.origin[0] + (c + 0.5) * res, cm.origin[1] + (r + 0.5) * res];
    // Bottleneck cost from the robot to every known cell: the lowest possible highest cost along a
    // path through known cells (what a planner allowed up to that cost could reach; no diagonal
    // step between two lethal cells, as in the explorer's reach).
    let mut bottleneck = vec![i32::MAX; w * h];
    if let Some(s) = seed {
        let mut heap = std::collections::BinaryHeap::new();
        let k0 = s.0 * w + s.1;
        bottleneck[k0] = cm.data[k0].max(0) as i32;
        heap.push(std::cmp::Reverse((bottleneck[k0], k0)));
        while let Some(std::cmp::Reverse((b, k))) = heap.pop() {
            if b > bottleneck[k] {
                continue;
            }
            let (r, c) = ((k / w) as i64, (k % w) as i64);
            for dr in -1i64..=1 {
                for dc in -1i64..=1 {
                    let (nr, nc) = (r + dr, c + dc);
                    if nr < 0 || nc < 0 || nr >= h as i64 || nc >= w as i64 {
                        continue;
                    }
                    let n = nr as usize * w + nc as usize;
                    if cm.data[n] < 0 {
                        continue;
                    }
                    if dr != 0 && dc != 0 && cm.data[r as usize * w + nc as usize] >= 100 && cm.data[nr as usize * w + c as usize] >= 100 {
                        continue;
                    }
                    let nb = b.max(cm.data[n] as i32);
                    if nb < bottleneck[n] {
                        bottleneck[n] = nb;
                        heap.push(std::cmp::Reverse((nb, n)));
                    }
                }
            }
        }
    }
    let frontier_cell = |r: usize, c: usize| {
        let g = cm.data[r * w + c];
        let unknown = |r: isize, c: isize| r < 0 || c < 0 || r >= h as isize || c >= w as isize || cm.data[r as usize * w + c as usize] < 0;
        let (ri, ci) = (r as isize, c as isize);
        g >= 0 && g as i64 <= fp.free_max && (unknown(ri - 1, ci) || unknown(ri + 1, ci) || unknown(ri, ci - 1) || unknown(ri, ci + 1))
    };
    // Frontier groups (8-connected, as the explorer groups them) with their cells.
    let mut label = vec![0usize; w * h];
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for r0 in 0..h {
        for c0 in 0..w {
            if label[r0 * w + c0] != 0 || !frontier_cell(r0, c0) {
                continue;
            }
            groups.push(Vec::new());
            let id = groups.len();
            let mut stack = vec![(r0, c0)];
            label[r0 * w + c0] = id;
            while let Some((r, c)) = stack.pop() {
                groups[id - 1].push(r * w + c);
                for dr in -1i64..=1 {
                    for dc in -1i64..=1 {
                        let (nr, nc) = (r as i64 + dr, c as i64 + dc);
                        if nr < 0 || nc < 0 || nr >= h as i64 || nc >= w as i64 {
                            continue;
                        }
                        let (nr, nc) = (nr as usize, nc as usize);
                        if label[nr * w + nc] == 0 && frontier_cell(nr, nc) {
                            label[nr * w + nc] = id;
                            stack.push((nr, nc));
                        }
                    }
                }
            }
        }
    }
    let mut out = Vec::new();
    let mut codes = Vec::with_capacity(groups.len());
    for cells in &groups {
        let n = cells.len();
        let (sr, sc) = cells.iter().fold((0u64, 0u64), |a, &k| (a.0 + (k / w) as u64, a.1 + (k % w) as u64));
        let centroid = (sr as f64 / n as f64, sc as f64 / n as f64);
        let reachable = cells.iter().any(|&k| reach[k]);
        let entry = cells.iter().map(|&k| bottleneck[k]).min().unwrap_or(i32::MAX);
        let p = xy(centroid.0, centroid.1);
        let goal = goals.iter().find(|g| g.group == centroid);
        let (code, verdict) = if (n as i64) < fp.min_cells {
            ('S', format!("too small (< {} cells)", fp.min_cells))
        } else if !reachable {
            let how = if entry == i32::MAX {
                "no known path to it".to_string()
            } else {
                format!("the cheapest way in crosses cost {entry}; the explorer only uses cells <= {}", fp.pass_max)
            };
            let peek = match peeks.iter().find(|g| g.group == centroid) {
                Some(g) => {
                    let gp = xy(g.goal.0 as f64, g.goal.1 as f64);
                    format!("; closer look from ({:.2}, {:.2})", gp[0], gp[1])
                }
                None => String::new(),
            };
            ('R', format!("not reachable: {how}{peek}"))
        } else if let Some(g) = goal {
            let gp = xy(g.goal.0 as f64, g.goal.1 as f64);
            ('U', format!("usable: goal ({:.2}, {:.2}), {:.1} m from the robot", gp[0], gp[1], (gp[0] - robot[0]).hypot(gp[1] - robot[1])))
        } else {
            ('C', format!(
                "no spot to look from (cost <= {}, {:.2} m from unknown; or cost <= {}) within {:.2} m",
                fp.clear_max,
                fp.standoff_cells * res,
                fp.pass_max,
                fp.search_cells * res
            ))
        };
        codes.push(code);
        if n >= 3 {
            out.push(json!({"x": round(p[0], 2), "y": round(p[1], 2), "cells": n, "verdict": verdict}));
        }
    }
    // The picture: costmap greys, frontier cells by verdict, the robot.
    let scale = 3;
    let mut cv = Canvas { w: w * scale, h: h * scale, scale, origin: cm.origin, res, px: vec![0; w * h * scale * scale * 4] };
    for r in 0..h {
        for c in 0..w {
            let v = cm.data[r * w + c];
            let col = if v < 0 {
                [150, 150, 150]
            } else if v >= 99 {
                [0, 0, 0]
            } else {
                let g = (255.0 - v as f64 * 1.6) as u8;
                [g, g, if reach[r * w + c] { g } else { g.saturating_sub(40) }]
            };
            for dy in 0..scale {
                for dx in 0..scale {
                    cv.set((c * scale + dx) as i64, (cv.h - 1 - (r * scale + dy)) as i64, col);
                }
            }
        }
    }
    for r in 0..h {
        for c in 0..w {
            let id = label[r * w + c];
            if id == 0 {
                continue;
            }
            // BGR: usable green, too small orange, unreachable red, no clear spot magenta.
            let col = match codes[id - 1] {
                'U' => [0, 200, 0],
                'S' => [0, 140, 255],
                'C' => [255, 0, 255],
                _ => [0, 0, 255],
            };
            for dy in 0..scale {
                for dx in 0..scale {
                    cv.set((c * scale + dx) as i64, (cv.h - 1 - (r * scale + dy)) as i64, col);
                }
            }
        }
    }
    let (x, y) = cv.to_px(robot);
    cv.dot(x, y, 5, [200, 80, 0]);
    let png = crate::web::image::encode_png_bgra(&cv.px, cv.w, cv.h);
    (out, png)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(t: f64, x: f64, state: &str) -> Sample {
        Sample { t, pose: [x, 0.0, 0.0], coverage: t / 100.0, reachable: 0.0, battery: None, explorer: state.into(), goal_active: t < 40.0 }
    }

    #[test]
    fn stuck_episodes_need_the_explorer_driving_and_no_progress() {
        let mut m = Metrics::default();
        for t in 0..20 {
            m.samples.push(sample(t as f64, t as f64 * 0.2, "exploring")); // moving
        }
        for t in 20..60 {
            m.samples.push(sample(t as f64, 4.0 + (t % 2) as f64 * 0.05, "exploring")); // wiggling in place
        }
        for t in 60..100 {
            m.samples.push(sample(t as f64, 4.0, "done")); // finished: standing is fine
        }
        let eps = m.stuck_episodes(30.0, 0.25);
        assert_eq!(eps.len(), 1);
        assert!((eps[0].0 - 19.0).abs() <= 1.0 && eps[0].1 == 59.0, "{eps:?}");
        assert_eq!(m.time_to(0.5), Some(50.0));
        assert_eq!(m.stuck_episodes(45.0, 0.25).len(), 0);
        // A goal was active for only half of it (20-39 of 20-59): stuck, not just waiting.
        let (stuck, idle) = m.stuck_and_idle(30.0, 0.25);
        assert_eq!((stuck.len(), idle.len()), (1, 0));
        for s in m.samples.iter_mut() {
            s.goal_active = false;
        }
        let (stuck, idle) = m.stuck_and_idle(30.0, 0.25);
        assert_eq!((stuck.len(), idle.len()), (0, 1));
    }

    #[test]
    fn path_is_thinned() {
        let mut m = Metrics::default();
        for k in 0..100 {
            m.track([k as f64 * 0.001, 0.0]);
        }
        assert_eq!(m.path.len(), 5);
    }

    #[test]
    fn frontier_report_judges_groups_like_the_explorer() {
        // 3 m x 2 m: known free on the left half, unknown on the right, a lethal wall with a pocket
        // of free space behind it.
        let (w, h) = (60, 40);
        let mut data = vec![-1i8; w * h];
        for r in 0..h {
            for c in 0..30 {
                data[r * w + c] = 0;
            }
        }
        let cm = CostGrid { width: w, height: h, res: 0.05, origin: [0.0, 0.0], data: data.clone() };
        let (groups, png) = frontier_report(&cm, [0.5, 1.0], &serde_json::Map::new());
        assert!(png.is_some());
        assert_eq!(groups.len(), 1, "{groups:?}");
        assert!(groups[0]["verdict"].as_str().unwrap().starts_with("usable"), "{groups:?}");
        for r in 0..h {
            data[r * w + 20] = 100;
        }
        let cm = CostGrid { data, ..cm };
        let (groups, _) = frontier_report(&cm, [0.5, 1.0], &serde_json::Map::new());
        let pocket = groups.iter().find(|g| g["verdict"].as_str().unwrap().starts_with("not reachable")).expect("pocket");
        assert!(pocket["verdict"].as_str().unwrap().contains("crosses cost 100"), "{pocket}");
    }
}
