//! RTAB-Map's 2D occupancy grid (/rtabmap/map), built the way RTAB-Map builds it on the robot from
//! what the simulated depth camera saw: a graph node about once a second when the robot moved
//! (RGBD/LinearUpdate, AngularUpdate), each node's local grid from the decimated depth cloud
//! (Grid/DepthDecimation 4, Grid/RangeMax 2.5 m, ground below Grid/MaxGroundHeight 6 cm, obstacles
//! up to Grid/MaxObstacleHeight 1 m, radius noise filter, ray tracing from the camera to every
//! point), merged with the global grid's hit/miss log-odds (GridGlobal/ProbHit 0.7, ProbMiss 0.45,
//! clamping), as rtabmap/core/global_map/OccupancyGrid.cpp (0.23) does: one update per node and
//! cell, and the square under the robot's footprint (GridGlobal/FootprintRadius) reset to the
//! clamping minimum. So a cell seen occupied in many nodes (a closed door) needs ~18 later nodes
//! seeing through it before it is free again, as on the robot. Not modelled: graph optimisation
//! (past nodes are never moved; with odometry drift the map smears as it would without loop
//! closures).
//!
//! Also measures coverage against the true floor plan.

use std::collections::HashMap;

use super::depth::{Camera, LensPose};
use super::scenario::MappingCfg;
use super::world::{Grid, P2};

fn logit(p: f64) -> f32 {
    (p / (1.0 - p)).ln() as f32
}

pub struct Mapper {
    pub cfg: MappingCfg,
    pub width: usize,
    pub height: usize,
    pub res: f64,
    pub origin: P2,
    logodds: Vec<f32>,
    observed: Vec<bool>,
    l_hit: f32,
    l_miss: f32,
    l_min: f32,
    l_max: f32,
    /// Graph node poses (base_link in the map frame): /rtabmap/mapPath.
    pub nodes: Vec<[f64; 3]>,
    truth_free: Vec<bool>,
    truth_reach: Vec<bool>,
    n_free: usize,
    n_reach: usize,
    /// Scratch for one node: 0 untouched, 1 empty, 2 obstacle.
    scratch: Vec<u8>,
    touched: Vec<usize>,
    pub version: u64,
}

/// Coverage of the true floor plan by the map.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Coverage {
    /// Fraction of the free floor connected to the start that is mapped free.
    pub free: f64,
    /// Fraction of the places the robot's centre can reach that are mapped free.
    pub reachable: f64,
    /// Mapped free area (m^2), anywhere.
    pub mapped_m2: f64,
    /// Truly free area connected to the start (m^2).
    pub truth_m2: f64,
}

impl Mapper {
    /// `truth`: the static floor plan; `start`: where the robot starts (coverage counts the floor
    /// connected to it); `radius`: the robot's.
    pub fn new(cfg: &MappingCfg, truth: &Grid, start: P2, radius: f64) -> Self {
        let n = truth.width * truth.height;
        let truth_free = truth.component(start, |k| !truth.solid[k]);
        let dist = truth.distance_field();
        let fits = |k: usize| dist[k] as f64 >= radius - 0.5 * truth.res;
        let mut seed = start;
        let (si, sj) = truth.cell(start);
        if truth.idx(si, sj).is_none_or(|k| !fits(k)) {
            // Start squeezed against a wall (cell-centre distances are coarse): nearest cell that fits.
            let mut best = f64::INFINITY;
            for dj in -8..=8i64 {
                for di in -8..=8i64 {
                    if let Some(k) = truth.idx(si + di, sj + dj) {
                        let d = ((di * di + dj * dj) as f64).sqrt();
                        if fits(k) && d < best {
                            best = d;
                            seed = truth.cell_center(si + di, sj + dj);
                        }
                    }
                }
            }
        }
        let truth_reach = truth.component(seed, |k| fits(k));
        let n_free = truth_free.iter().filter(|&&f| f).count();
        let n_reach = truth_reach.iter().filter(|&&f| f).count();
        Mapper {
            cfg: cfg.clone(),
            width: truth.width,
            height: truth.height,
            res: truth.res,
            origin: truth.origin,
            logodds: vec![0.0; n],
            observed: vec![false; n],
            l_hit: logit(cfg.prob_hit),
            l_miss: logit(cfg.prob_miss),
            l_min: logit(cfg.clamp_min),
            l_max: logit(cfg.clamp_max),
            nodes: Vec::new(),
            truth_free,
            truth_reach,
            n_free,
            n_reach,
            scratch: vec![0; n],
            touched: Vec::new(),
            version: 0,
        }
    }

    fn cell(&self, p: P2) -> Option<usize> {
        let i = ((p[0] - self.origin[0]) / self.res).floor() as i64;
        let j = ((p[1] - self.origin[1]) / self.res).floor() as i64;
        (i >= 0 && j >= 0 && (i as usize) < self.width && (j as usize) < self.height).then(|| j as usize * self.width + i as usize)
    }

    /// Whether RTAB-Map would add a node at this pose (first node, or moved/turned enough).
    pub fn wants_node(&self, base: [f64; 3]) -> bool {
        let Some(last) = self.nodes.last() else { return true };
        let moved = (base[0] - last[0]).hypot(base[1] - last[1]);
        let turned = super::physics::wrap(base[2] - last[2]).abs();
        moved >= self.cfg.linear_update_m || turned >= self.cfg.angular_update_rad
    }

    fn mark(&mut self, k: usize, v: u8) {
        if self.scratch[k] == 0 {
            self.touched.push(k);
        }
        if v > self.scratch[k] {
            self.scratch[k] = v;
        }
    }

    /// Adds a graph node: the depth image seen from `lens` with the robot at `base` (both as the
    /// localisation believes, i.e. with any odometry error).
    pub fn add_node(&mut self, cam: &Camera, depth: &[u16], lens: LensPose, base: [f64; 3]) {
        let c = self.cfg.clone();
        let (w, h) = (cam.k.width, cam.k.height);
        let dec = c.decimation.max(1);
        let (cy, sy) = (lens.yaw.cos(), lens.yaw.sin());
        // (map x, map y, height, is obstacle)
        let mut pts: Vec<(f64, f64, f64, bool)> = Vec::new();
        for v in (0..h).step_by(dec) {
            for u in (0..w).step_by(dec) {
                let mm = depth[v * w + u];
                if mm == 0 {
                    continue;
                }
                let p = cam.point(u, v, mm as f64 / 1000.0);
                if p[0].hypot(p[1]) > c.range_max_m {
                    continue;
                }
                let z = lens.z + p[2];
                if z > c.max_obstacle_height_m {
                    continue;
                }
                pts.push((lens.x + cy * p[0] - sy * p[1], lens.y + sy * p[0] + cy * p[1], z, z > c.max_ground_height_m));
            }
        }
        if c.noise_radius_m > 0.0 && c.noise_min_neighbors > 0 {
            pts = radius_filter(&pts, c.noise_radius_m, c.noise_min_neighbors);
        }
        let Some(from) = self.cell([lens.x, lens.y]) else { return };
        let (fi, fj) = ((from % self.width) as i64, (from / self.width) as i64);
        for &(x, y, _, obstacle) in &pts {
            let Some(to) = self.cell([x, y]) else { continue };
            let (ti, tj) = ((to % self.width) as i64, (to / self.width) as i64);
            for (i, j) in bresenham(fi, fj, ti, tj) {
                if (i, j) == (ti, tj) {
                    break;
                }
                let k = j as usize * self.width + i as usize;
                self.mark(k, 1);
            }
            self.mark(to, if obstacle { 2 } else { 1 });
        }
        // RTAB-Map's footprint (GridGlobal/FootprintRadius): the square under the robot is set to
        // the clamping minimum (free, not just one miss) and no obstacle of this node lands in it.
        let r = c.footprint_radius_m;
        let mut footprint = Vec::new();
        if r >= self.res * 1.5 {
            let to_cell = |v: f64, o: f64| ((v - o) / self.res) as i64; // C++ float -> int: truncation
            let (i0, i1) = (to_cell(base[0] - r, self.origin[0]).max(0), to_cell(base[0] + r, self.origin[0]).min(self.width as i64 - 1));
            let (j0, j1) = (to_cell(base[1] - r, self.origin[1]).max(0), to_cell(base[1] + r, self.origin[1]).min(self.height as i64 - 1));
            for j in j0..j1 {
                for i in i0..i1 {
                    let k = j as usize * self.width + i as usize;
                    self.mark(k, 1);
                    footprint.push(k);
                }
            }
        }
        for &k in &self.touched {
            let delta = if self.scratch[k] == 2 { self.l_hit } else { self.l_miss };
            self.logodds[k] = (self.logodds[k] + delta).clamp(self.l_min, self.l_max);
            self.observed[k] = true;
            self.scratch[k] = 0;
        }
        for k in footprint {
            self.logodds[k] = self.l_min;
        }
        self.touched.clear();
        self.nodes.push(base);
        self.version += 1;
    }

    /// Start with the whole floor plan mapped (as from a saved map): free cells free, walls next
    /// to free space occupied.
    pub fn known_from_world(&mut self, truth: &Grid) {
        for j in 0..self.height as i64 {
            for i in 0..self.width as i64 {
                let k = j as usize * self.width + i as usize;
                if !truth.solid[k] {
                    self.logodds[k] = self.l_min;
                    self.observed[k] = true;
                } else if (-1..=1).any(|dj| (-1..=1).any(|di| !truth.solid_ij(i + di, j + dj))) {
                    self.logodds[k] = self.l_max;
                    self.observed[k] = true;
                }
            }
        }
        self.version += 1;
    }

    /// nav_msgs/OccupancyGrid data: -1 unknown, 0 free, 100 occupied.
    pub fn occupancy(&self) -> Vec<i8> {
        self.logodds.iter().zip(&self.observed).map(|(&l, &o)| if !o { -1 } else if l > 0.0 { 100 } else { 0 }).collect()
    }

    pub fn is_free(&self, k: usize) -> bool {
        self.observed[k] && self.logodds[k] <= 0.0
    }

    pub fn coverage(&self) -> Coverage {
        let mut free = 0;
        let mut reach = 0;
        let mut mapped = 0;
        for k in 0..self.logodds.len() {
            if self.is_free(k) {
                mapped += 1;
                free += self.truth_free[k] as usize;
                reach += self.truth_reach[k] as usize;
            }
        }
        let a = self.res * self.res;
        Coverage {
            free: free as f64 / self.n_free.max(1) as f64,
            reachable: reach as f64 / self.n_reach.max(1) as f64,
            mapped_m2: mapped as f64 * a,
            truth_m2: self.n_free as f64 * a,
        }
    }

    pub fn truth_free(&self, k: usize) -> bool {
        self.truth_free[k]
    }
}

/// Keeps the points with at least `min` other points within `radius` (3D).
fn radius_filter(pts: &[(f64, f64, f64, bool)], radius: f64, min: usize) -> Vec<(f64, f64, f64, bool)> {
    let key = |p: &(f64, f64, f64, bool)| ((p.0 / radius).floor() as i64, (p.1 / radius).floor() as i64, (p.2 / radius).floor() as i64);
    let mut cells: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (k, p) in pts.iter().enumerate() {
        cells.entry(key(p)).or_default().push(k);
    }
    let r2 = radius * radius;
    pts.iter()
        .enumerate()
        .filter(|(k, p)| {
            let (ci, cj, ck) = key(p);
            let mut n = 0;
            for di in -1..=1 {
                for dj in -1..=1 {
                    for dk in -1..=1 {
                        if let Some(list) = cells.get(&(ci + di, cj + dj, ck + dk)) {
                            for &o in list {
                                if o == *k {
                                    continue;
                                }
                                let q = &pts[o];
                                if (q.0 - p.0).powi(2) + (q.1 - p.1).powi(2) + (q.2 - p.2).powi(2) <= r2 {
                                    n += 1;
                                    if n >= min {
                                        return true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            false
        })
        .map(|(_, p)| *p)
        .collect()
}

/// Grid cells on the line from (i0, j0) to (i1, j1), both ends included.
pub fn bresenham(i0: i64, j0: i64, i1: i64, j1: i64) -> impl Iterator<Item = (i64, i64)> {
    let (dx, dy) = ((i1 - i0).abs(), -(j1 - j0).abs());
    let (sx, sy) = (if i0 < i1 { 1 } else { -1 }, if j0 < j1 { 1 } else { -1 });
    let (mut i, mut j, mut err) = (i0, j0, dx + dy);
    let mut done = false;
    std::iter::from_fn(move || {
        if done {
            return None;
        }
        let out = (i, j);
        if i == i1 && j == j1 {
            done = true;
        } else {
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                i += sx;
            }
            if e2 <= dx {
                err += dx;
                j += sy;
            }
        }
        Some(out)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::twin::depth::{lens_pose, RenderOpts, D1000};
    use crate::twin::rng::Rng;
    use crate::twin::world::{Body, DepthVis, Kind, Shape, World};

    fn room() -> Grid {
        // 6 m x 4 m, walls one cell thick.
        let (w, h) = (120, 80);
        let mut g = Grid::new(w, h, 0.05, [0.0, 0.0]);
        for i in 0..w {
            g.solid[i] = true;
            g.solid[(h - 1) * w + i] = true;
        }
        for j in 0..h {
            g.solid[j * w] = true;
            g.solid[j * w + w - 1] = true;
        }
        g
    }

    #[test]
    fn bresenham_lines() {
        assert_eq!(bresenham(0, 0, 3, 1).collect::<Vec<_>>(), [(0, 0), (1, 0), (2, 1), (3, 1)]);
        assert_eq!(bresenham(2, 2, 2, 2).collect::<Vec<_>>(), [(2, 2)]);
        assert_eq!(bresenham(0, 0, -2, -2).count(), 3);
    }

    #[test]
    fn a_node_maps_the_wall_ahead_and_the_floor_before_it() {
        let g = room();
        let world = World::new(g.clone());
        let cam = Camera::new(D1000);
        let mut m = Mapper::new(&MappingCfg::default(), &g, [1.0, 2.0], 0.17);
        assert!(m.coverage().free == 0.0 && m.wants_node([1.0, 2.0, 0.0]));
        // Robot 2.2 m from the +x wall (cells from x = 5.95), facing it: within Grid/RangeMax.
        let base = [3.6, 2.0, 0.0];
        let lens = lens_pose(base, [0.153, 0.0, 0.13]);
        let depth = cam.render(&world, lens, &RenderOpts::default(), &mut Rng::new(1), None);
        m.add_node(&cam, &depth, lens, base);
        let occ = m.occupancy();
        let at = |x: f64, y: f64| occ[m.cell([x, y]).unwrap()];
        assert_eq!(at(5.97, 2.0), 100, "wall cell");
        assert_eq!(at(5.0, 2.0), 0, "floor in view");
        assert_eq!(at(3.6, 2.0), 0, "footprint");
        assert_eq!(at(2.0, 2.0), -1, "behind the robot");
        assert!(!m.wants_node([3.65, 2.0, 0.0]) && m.wants_node([3.7, 2.0, 0.0]) && m.wants_node([3.6, 2.0, 0.2]));
        let c = m.coverage();
        assert!(c.free > 0.05 && c.free < 0.5 && c.reachable > c.free * 0.5, "{c:?}");
        // Known map: everything free is mapped.
        m.known_from_world(&g);
        assert!((m.coverage().free - 1.0).abs() < 1e-12);
    }

    #[test]
    fn misses_erase_a_wall_seen_once_only_after_several_passes() {
        let g = room();
        let mut world = World::new(g.clone());
        let cam = Camera::new(D1000);
        let mut m = Mapper::new(&MappingCfg::default(), &g, [1.0, 2.0], 0.17);
        let base = [3.6, 2.0, 0.0];
        let lens = lens_pose(base, [0.153, 0.0, 0.13]);
        // A person 1.5 m ahead, then gone: the cells it covered need ~4 misses (ProbMiss 0.45).
        world.bodies.push(Body {
            name: "p".into(),
            kind: Kind::Person,
            shape: Shape::Circle { c: [5.4, 2.0], r: 0.2 },
            zmin: 0.0,
            zmax: 1.7,
            depth: DepthVis::Solid,
            ir: true,
            blocks: true,
            active: true,
        });
        let depth = cam.render(&world, lens, &RenderOpts::default(), &mut Rng::new(1), None);
        m.add_node(&cam, &depth, lens, base);
        let k = m.cell([5.22, 2.0]).unwrap();
        assert_eq!(m.occupancy()[k], 100);
        world.bodies.clear();
        let depth = cam.render(&world, lens, &RenderOpts::default(), &mut Rng::new(1), None);
        let mut passes = 0;
        while m.occupancy()[k] == 100 {
            m.add_node(&cam, &depth, lens, base);
            passes += 1;
        }
        assert_eq!(passes, 5);
        // Seen in many nodes (a door closed for a while), it is saturated (ProbClampingMax 0.971):
        // ~18 nodes seeing through it before the map shows it free again, as RTAB-Map's grid.
        world.bodies.push(Body {
            name: "door".into(),
            kind: Kind::Door,
            shape: Shape::Circle { c: [5.4, 2.0], r: 0.2 },
            zmin: 0.0,
            zmax: 1.7,
            depth: DepthVis::Solid,
            ir: true,
            blocks: true,
            active: true,
        });
        let closed = cam.render(&world, lens, &RenderOpts::default(), &mut Rng::new(1), None);
        for _ in 0..10 {
            m.add_node(&cam, &closed, lens, base);
        }
        assert_eq!(m.occupancy()[k], 100);
        let mut passes = 0;
        while m.occupancy()[k] == 100 {
            m.add_node(&cam, &depth, lens, base);
            passes += 1;
        }
        assert_eq!(passes, 18);
    }

    #[test]
    fn footprint_is_reset_free_like_rtabmap() {
        let g = room();
        let world = World::new(g.clone());
        let cam = Camera::new(D1000);
        let mut m = Mapper::new(&MappingCfg::default(), &g, [1.0, 2.0], 0.17);
        let base = [3.6, 2.0, 0.0];
        let lens = lens_pose(base, [0.153, 0.0, 0.13]);
        let depth = cam.render(&world, lens, &RenderOpts::default(), &mut Rng::new(1), None);
        m.add_node(&cam, &depth, lens, base);
        let k = m.cell([3.62, 2.02]).unwrap();
        assert_eq!(m.logodds[k], m.l_min, "clamping minimum, not one miss");
    }

    #[test]
    fn radius_filter_drops_specks() {
        let mut pts: Vec<(f64, f64, f64, bool)> = (0..10).map(|k| (1.0 + k as f64 * 0.015, 1.0, 0.5, true)).collect();
        pts.push((3.0, 3.0, 0.5, true));
        let kept = radius_filter(&pts, 0.1, 5);
        assert_eq!(kept.len(), 10);
    }
}
