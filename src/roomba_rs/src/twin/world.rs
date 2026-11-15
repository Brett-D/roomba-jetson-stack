//! The twin's house: a solid/free grid (the floor plan from RTAB-Map's saved map, every cell that
//! isn't free is a full-height wall) plus bodies (doors, railings, obstacles with heights, people,
//! the dock) and cliffs, with the geometry the physics and the sensors need.

use std::sync::Arc;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

pub type P2 = [f64; 2];

/// Height of the walls and the ceiling the depth camera sees (m).
pub const WALL_HEIGHT_M: f64 = 2.5;

fn sub(a: P2, b: P2) -> P2 {
    [a[0] - b[0], a[1] - b[1]]
}
fn dot(a: P2, b: P2) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn cross(a: P2, b: P2) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}
fn norm(a: P2) -> f64 {
    a[0].hypot(a[1])
}

/// Nearest point to `p` on the segment a-b.
pub fn seg_nearest(p: P2, a: P2, b: P2) -> P2 {
    let ab = sub(b, a);
    let l2 = dot(ab, ab);
    let t = if l2 > 0.0 { (dot(sub(p, a), ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
    [a[0] + t * ab[0], a[1] + t * ab[1]]
}

/// Distance along the unit ray o + s d to the segment a-b (s >= 0), if it crosses it.
fn ray_segment(o: P2, d: P2, a: P2, b: P2) -> Option<f64> {
    let e = sub(b, a);
    let denom = cross(d, e);
    if denom.abs() < 1e-12 {
        return None;
    }
    let w = sub(a, o);
    let s = cross(w, e) / denom;
    let u = cross(w, d) / denom;
    (s >= 0.0 && (0.0..=1.0).contains(&u)).then_some(s)
}

// ---------------------------------------------------------------- the grid

#[derive(Deserialize)]
struct MapYaml {
    image: String,
    resolution: f64,
    origin: Vec<f64>,
    #[serde(default)]
    negate: i32,
    #[serde(default = "default_free_thresh")]
    free_thresh: f64,
}

fn default_free_thresh() -> f64 {
    0.196
}

#[derive(Clone, Debug)]
pub struct Grid {
    pub width: usize,
    pub height: usize,
    pub res: f64,
    pub origin: P2,
    /// Row-major from the origin (row 0 = lowest y), like nav_msgs/OccupancyGrid: true = solid.
    pub solid: Vec<bool>,
}

impl Grid {
    /// All free.
    pub fn new(width: usize, height: usize, res: f64, origin: P2) -> Self {
        Grid { width, height, res, origin, solid: vec![false; width * height] }
    }

    /// A map_server map (YAML + PGM): cells whose occupancy (255 - value) / 255 is below free_thresh
    /// are free, everything else (occupied and unknown) is solid.
    pub fn load(yaml_path: &str) -> Result<Grid> {
        let text = std::fs::read_to_string(yaml_path).with_context(|| format!("cannot read {yaml_path}"))?;
        let m: MapYaml = serde_yaml::from_str(&text).with_context(|| format!("bad map yaml {yaml_path}"))?;
        let dir = std::path::Path::new(yaml_path).parent().unwrap_or(std::path::Path::new("."));
        let image = dir.join(&m.image);
        let bytes = std::fs::read(&image).with_context(|| format!("cannot read {}", image.display()))?;
        if m.origin.len() < 2 {
            bail!("{yaml_path}: origin needs x and y");
        }
        Grid::from_pgm(&bytes, m.resolution, [m.origin[0], m.origin[1]], m.free_thresh, m.negate != 0)
    }

    pub fn from_pgm(bytes: &[u8], res: f64, origin: P2, free_thresh: f64, negate: bool) -> Result<Grid> {
        let (w, h, maxval, data) = parse_pgm(bytes)?;
        let mut g = Grid::new(w, h, res, origin);
        for row in 0..h {
            for col in 0..w {
                let v = data[row * w + col] as f64 / maxval as f64;
                let occ = if negate { v } else { 1.0 - v };
                // PGM row 0 is the top of the map (largest y).
                g.solid[(h - 1 - row) * w + col] = occ >= free_thresh;
            }
        }
        Ok(g)
    }

    pub fn cell(&self, p: P2) -> (i64, i64) {
        (((p[0] - self.origin[0]) / self.res).floor() as i64, ((p[1] - self.origin[1]) / self.res).floor() as i64)
    }

    pub fn idx(&self, i: i64, j: i64) -> Option<usize> {
        (i >= 0 && j >= 0 && (i as usize) < self.width && (j as usize) < self.height).then(|| j as usize * self.width + i as usize)
    }

    /// Outside the grid counts as solid.
    pub fn solid_ij(&self, i: i64, j: i64) -> bool {
        self.idx(i, j).map_or(true, |k| self.solid[k])
    }

    pub fn solid_at(&self, p: P2) -> bool {
        let (i, j) = self.cell(p);
        self.solid_ij(i, j)
    }

    pub fn cell_center(&self, i: i64, j: i64) -> P2 {
        [self.origin[0] + (i as f64 + 0.5) * self.res, self.origin[1] + (j as f64 + 0.5) * self.res]
    }

    /// Distance along the unit ray o + s d to the first solid cell (0 if `o` is in one), at most `max`.
    pub fn ray(&self, o: P2, d: P2, max: f64) -> f64 {
        let (mut i, mut j) = self.cell(o);
        if self.solid_ij(i, j) {
            return 0.0;
        }
        let gx = (o[0] - self.origin[0]) / self.res;
        let gy = (o[1] - self.origin[1]) / self.res;
        let (si, sj) = (if d[0] > 0.0 { 1 } else { -1 }, if d[1] > 0.0 { 1 } else { -1 });
        let t_next = |g: f64, c: i64, dc: f64| {
            if dc > 0.0 {
                ((c as f64 + 1.0) - g) * self.res / dc
            } else if dc < 0.0 {
                (g - c as f64) * self.res / -dc
            } else {
                f64::INFINITY
            }
        };
        let mut tx = t_next(gx, i, d[0]);
        let mut ty = t_next(gy, j, d[1]);
        let dx = if d[0] != 0.0 { self.res / d[0].abs() } else { f64::INFINITY };
        let dy = if d[1] != 0.0 { self.res / d[1].abs() } else { f64::INFINITY };
        loop {
            let t;
            if tx < ty {
                t = tx;
                tx += dx;
                i += si;
            } else {
                t = ty;
                ty += dy;
                j += sj;
            }
            if t > max {
                return max;
            }
            if self.solid_ij(i, j) {
                return t;
            }
        }
    }

    /// Nearest solid point within `max` of `p`: (distance, point); distance 0 inside a solid cell.
    pub fn nearest_solid(&self, p: P2, max: f64) -> Option<(f64, P2)> {
        let (i0, j0) = self.cell([p[0] - max, p[1] - max]);
        let (i1, j1) = self.cell([p[0] + max, p[1] + max]);
        let mut best: Option<(f64, P2)> = None;
        for j in j0..=j1 {
            for i in i0..=i1 {
                if !self.solid_ij(i, j) {
                    continue;
                }
                let lo = [self.origin[0] + i as f64 * self.res, self.origin[1] + j as f64 * self.res];
                let q = [p[0].clamp(lo[0], lo[0] + self.res), p[1].clamp(lo[1], lo[1] + self.res)];
                let d = norm(sub(p, q));
                if d <= max && best.is_none_or(|b| d < b.0) {
                    best = Some((d, q));
                }
            }
        }
        best
    }

    /// Free cells 4-connected to the cell of `start` (empty if it is solid).
    pub fn component(&self, start: P2, passable: impl Fn(usize) -> bool) -> Vec<bool> {
        let mut seen = vec![false; self.solid.len()];
        let (i, j) = self.cell(start);
        let Some(s) = self.idx(i, j) else { return seen };
        if !passable(s) {
            return seen;
        }
        let mut stack = vec![s];
        seen[s] = true;
        while let Some(k) = stack.pop() {
            let (ci, cj) = ((k % self.width) as i64, (k / self.width) as i64);
            for (ni, nj) in [(ci + 1, cj), (ci - 1, cj), (ci, cj + 1), (ci, cj - 1)] {
                if let Some(n) = self.idx(ni, nj) {
                    if !seen[n] && passable(n) {
                        seen[n] = true;
                        stack.push(n);
                    }
                }
            }
        }
        seen
    }

    /// Distance (m) from each cell centre to the nearest solid cell centre (two-pass chamfer, 8-neighbour).
    pub fn distance_field(&self) -> Vec<f32> {
        let (w, h) = (self.width, self.height);
        let inf = f32::MAX / 4.0;
        let mut d: Vec<f32> = self.solid.iter().map(|&s| if s { 0.0 } else { inf }).collect();
        let (a, b) = (1.0f32, std::f32::consts::SQRT_2);
        for j in 0..h {
            for i in 0..w {
                let k = j * w + i;
                let mut v = d[k];
                if i > 0 {
                    v = v.min(d[k - 1] + a);
                }
                if j > 0 {
                    v = v.min(d[k - w] + a);
                    if i > 0 {
                        v = v.min(d[k - w - 1] + b);
                    }
                    if i + 1 < w {
                        v = v.min(d[k - w + 1] + b);
                    }
                }
                d[k] = v;
            }
        }
        for j in (0..h).rev() {
            for i in (0..w).rev() {
                let k = j * w + i;
                let mut v = d[k];
                if i + 1 < w {
                    v = v.min(d[k + 1] + a);
                }
                if j + 1 < h {
                    v = v.min(d[k + w] + a);
                    if i + 1 < w {
                        v = v.min(d[k + w + 1] + b);
                    }
                    if i > 0 {
                        v = v.min(d[k + w - 1] + b);
                    }
                }
                d[k] = v;
            }
        }
        // Cells at the border: the outside counts as solid.
        for j in 0..h {
            for i in 0..w {
                let edge = (i.min(w - 1 - i)).min(j.min(h - 1 - j)) as f32 + 1.0;
                let k = j * w + i;
                d[k] = d[k].min(edge) * self.res as f32;
            }
        }
        d
    }
}

fn parse_pgm(bytes: &[u8]) -> Result<(usize, usize, u32, Vec<u8>)> {
    let mut pos = 0;
    let mut tokens = Vec::new();
    while tokens.len() < 4 {
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos < bytes.len() && bytes[pos] == b'#' {
            while pos < bytes.len() && bytes[pos] != b'\n' {
                pos += 1;
            }
            continue;
        }
        let start = pos;
        while pos < bytes.len() && !bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if start == pos {
            bail!("truncated PGM header");
        }
        tokens.push(String::from_utf8_lossy(&bytes[start..pos]).to_string());
    }
    pos += 1; // the single whitespace after maxval
    if tokens[0] != "P5" {
        bail!("only binary PGM (P5) maps are supported, not {}", tokens[0]);
    }
    let w: usize = tokens[1].parse()?;
    let h: usize = tokens[2].parse()?;
    let maxval: u32 = tokens[3].parse()?;
    if maxval == 0 || maxval > 255 {
        bail!("PGM maxval {maxval} not supported (8-bit only)");
    }
    if bytes.len() < pos + w * h {
        bail!("PGM data truncated: {} of {} bytes", bytes.len().saturating_sub(pos), w * h);
    }
    Ok((w, h, maxval, bytes[pos..pos + w * h].to_vec()))
}

// ---------------------------------------------------------------- shapes and bodies

#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    Circle { c: P2, r: f64 },
    /// Simple polygon, either winding.
    Polygon(Vec<P2>),
    /// A thick segment (railings).
    Capsule { a: P2, b: P2, r: f64 },
}

impl Shape {
    pub fn contains(&self, p: P2) -> bool {
        match self {
            Shape::Circle { c, r } => norm(sub(p, *c)) <= *r,
            Shape::Capsule { a, b, r } => norm(sub(p, seg_nearest(p, *a, *b))) <= *r,
            Shape::Polygon(pts) => {
                let mut inside = false;
                let n = pts.len();
                for k in 0..n {
                    let (a, b) = (pts[k], pts[(k + 1) % n]);
                    if (a[1] > p[1]) != (b[1] > p[1]) {
                        let x = a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]);
                        if p[0] < x {
                            inside = !inside;
                        }
                    }
                }
                inside
            }
        }
    }

    /// Distance from `p` to the shape and the nearest point of it (`p` itself, distance 0, inside).
    pub fn nearest(&self, p: P2) -> (f64, P2) {
        match self {
            Shape::Circle { c, r } => {
                let v = sub(p, *c);
                let d = norm(v);
                if d <= *r {
                    (0.0, p)
                } else {
                    (d - r, [c[0] + v[0] * r / d, c[1] + v[1] * r / d])
                }
            }
            Shape::Capsule { a, b, r } => {
                let q = seg_nearest(p, *a, *b);
                let v = sub(p, q);
                let d = norm(v);
                if d <= *r {
                    (0.0, p)
                } else {
                    (d - r, [q[0] + v[0] * r / d, q[1] + v[1] * r / d])
                }
            }
            Shape::Polygon(pts) => {
                if self.contains(p) {
                    return (0.0, p);
                }
                let n = pts.len();
                let mut best = (f64::INFINITY, p);
                for k in 0..n {
                    let q = seg_nearest(p, pts[k], pts[(k + 1) % n]);
                    let d = norm(sub(p, q));
                    if d < best.0 {
                        best = (d, q);
                    }
                }
                best
            }
        }
    }

    /// (entry, exit) distances of the unit ray o + s d through the shape: the first interval, with
    /// entry 0 when `o` is inside. None when the ray misses it.
    pub fn ray(&self, o: P2, d: P2) -> Option<(f64, f64)> {
        match self {
            Shape::Circle { c, r } => {
                let oc = sub(o, *c);
                let b = dot(oc, d);
                let cc = dot(oc, oc) - r * r;
                let disc = b * b - cc;
                if disc < 0.0 {
                    return None;
                }
                let sq = disc.sqrt();
                let (s0, s1) = (-b - sq, -b + sq);
                (s1 >= 0.0).then_some((s0.max(0.0), s1))
            }
            Shape::Polygon(pts) => {
                let n = pts.len();
                let mut hits: Vec<f64> = (0..n).filter_map(|k| ray_segment(o, d, pts[k], pts[(k + 1) % n])).collect();
                hits.sort_by(|a, b| a.total_cmp(b));
                if self.contains(o) {
                    Some((0.0, hits.first().copied().unwrap_or(f64::INFINITY)))
                } else {
                    match hits.len() {
                        0 => None,
                        1 => Some((hits[0], hits[0])),
                        _ => Some((hits[0], hits[1])),
                    }
                }
            }
            Shape::Capsule { a, b, r } => {
                // Union of the two end discs and the rectangle between them.
                let ab = sub(*b, *a);
                let l = norm(ab);
                let mut parts = vec![Shape::Circle { c: *a, r: *r }, Shape::Circle { c: *b, r: *r }];
                if l > 1e-9 {
                    let nrm = [-ab[1] / l * r, ab[0] / l * r];
                    parts.push(Shape::Polygon(vec![
                        [a[0] + nrm[0], a[1] + nrm[1]],
                        [b[0] + nrm[0], b[1] + nrm[1]],
                        [b[0] - nrm[0], b[1] - nrm[1]],
                        [a[0] - nrm[0], a[1] - nrm[1]],
                    ]));
                }
                let ivs: Vec<(f64, f64)> = parts.iter().filter_map(|s| s.ray(o, d)).collect();
                let entry = ivs.iter().map(|v| v.0).fold(f64::INFINITY, f64::min);
                let exit = ivs.iter().map(|v| v.1).fold(f64::NEG_INFINITY, f64::max);
                entry.is_finite().then_some((entry, exit))
            }
        }
    }

    pub fn bbox(&self) -> [f64; 4] {
        match self {
            Shape::Circle { c, r } => [c[0] - r, c[1] - r, c[0] + r, c[1] + r],
            Shape::Capsule { a, b, r } => [a[0].min(b[0]) - r, a[1].min(b[1]) - r, a[0].max(b[0]) + r, a[1].max(b[1]) + r],
            Shape::Polygon(pts) => pts.iter().fold([f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY], |b, p| {
                [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])]
            }),
        }
    }

    pub fn centroid(&self) -> P2 {
        let b = self.bbox();
        [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Door,
    /// A railing's physical extent (the robot can't pass; the depth camera doesn't see it).
    Railing,
    /// A railing post as the depth camera sees it (sometimes).
    Post,
    Obstacle,
    Person,
    Dock,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Door => "door",
            Kind::Railing => "railing",
            Kind::Post => "post",
            Kind::Obstacle => "obstacle",
            Kind::Person => "person",
            Kind::Dock => "dock",
        }
    }
}

/// How the depth camera sees a body.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DepthVis {
    Solid,
    Invisible,
    /// Seen in a frame with this probability (thin posts: stereo depth rarely returns them).
    Sparse(f64),
}

#[derive(Clone, Debug)]
pub struct Body {
    pub name: String,
    pub kind: Kind,
    pub shape: Shape,
    /// Height range above the floor (m).
    pub zmin: f64,
    pub zmax: f64,
    pub depth: DepthVis,
    /// Seen by the light bumpers (IR, ~1-8 cm above the floor).
    pub ir: bool,
    /// Stops the robot.
    pub blocks: bool,
    /// Present now (an open door or a person who left is inactive).
    pub active: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum What {
    Wall,
    Body(usize),
}

#[derive(Clone, Copy, Debug)]
pub struct Contact {
    pub dist: f64,
    pub point: P2,
    pub what: What,
}

#[derive(Clone, Debug)]
pub struct World {
    pub grid: Arc<Grid>,
    pub bodies: Vec<Body>,
    /// (name, area) where the floor drops away (stairs).
    pub cliffs: Vec<(String, Shape)>,
}

impl World {
    pub fn new(grid: Grid) -> Self {
        World { grid: Arc::new(grid), bodies: Vec::new(), cliffs: Vec::new() }
    }

    fn touches_band(b: &Body, zlo: f64, zhi: f64) -> bool {
        b.active && b.zmin < zhi && b.zmax > zlo
    }

    /// Nearest obstacle within `max` of `p` that a body spanning heights zlo..zhi would hit
    /// (walls, and the blocking bodies present now in that height band).
    pub fn nearest(&self, p: P2, max: f64, zlo: f64, zhi: f64) -> Option<Contact> {
        let mut best = self.grid.nearest_solid(p, max).map(|(dist, point)| Contact { dist, point, what: What::Wall });
        for (k, b) in self.bodies.iter().enumerate() {
            if !b.blocks || !Self::touches_band(b, zlo, zhi) {
                continue;
            }
            let bb = b.shape.bbox();
            if p[0] < bb[0] - max || p[0] > bb[2] + max || p[1] < bb[1] - max || p[1] > bb[3] + max {
                continue;
            }
            let (dist, point) = b.shape.nearest(p);
            if dist <= max && best.is_none_or(|c| dist < c.dist) {
                best = Some(Contact { dist, point, what: What::Body(k) });
            }
        }
        best
    }

    /// The light bumpers' and wall sensor's view: walls and the IR-visible bodies in the height
    /// band zlo..zhi along the unit ray, up to `max`.
    pub fn ir_ray(&self, o: P2, d: P2, max: f64, zlo: f64, zhi: f64) -> (f64, Option<What>) {
        let w = self.grid.ray(o, d, max);
        let mut best = (w, (w < max).then_some(What::Wall));
        for (k, b) in self.bodies.iter().enumerate() {
            if !b.ir || !Self::touches_band(b, zlo, zhi) {
                continue;
            }
            if let Some((s, _)) = b.shape.ray(o, d) {
                if s < best.0 {
                    best = (s, Some(What::Body(k)));
                }
            }
        }
        best
    }

    pub fn cliff_at(&self, p: P2) -> Option<usize> {
        self.cliffs.iter().position(|(_, s)| s.contains(p))
    }

    pub fn what_name(&self, w: What) -> String {
        match w {
            What::Wall => "wall".into(),
            What::Body(k) => format!("{}:{}", self.bodies[k].kind.as_str(), self.bodies[k].name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn room() -> Grid {
        // 2 m x 2 m at 0.05 m, walls one cell thick around the border.
        let mut g = Grid::new(40, 40, 0.05, [0.0, 0.0]);
        for k in 0..40 {
            for (i, j) in [(k, 0), (k, 39), (0, k), (39, k)] {
                g.solid[j * 40 + i] = true;
            }
        }
        g
    }

    #[test]
    fn pgm_rows_are_flipped_and_unknown_is_solid() {
        // 3x2 image: top row free, unknown, occupied; bottom row free, free, free.
        let mut pgm = b"P5\n# comment\n3 2\n255\n".to_vec();
        pgm.extend_from_slice(&[254, 205, 0, 254, 254, 254]);
        let g = Grid::from_pgm(&pgm, 0.05, [1.0, 2.0], 0.196, false).unwrap();
        assert_eq!(g.solid, vec![false, false, false, false, true, true]);
        assert!(!g.solid_at([1.01, 2.01]) && g.solid_at([1.11, 2.06]) && g.solid_at([0.99, 2.01]));
    }

    #[test]
    fn grid_ray_and_nearest_solid() {
        let g = room();
        // From the centre towards +x the wall cell starts at x = 1.95.
        assert!((g.ray([1.0, 1.0], [1.0, 0.0], 5.0) - 0.95).abs() < 1e-9);
        assert!((g.ray([1.0, 1.0], [-1.0, 0.0], 5.0) - 0.95).abs() < 1e-9);
        let d = std::f64::consts::FRAC_1_SQRT_2;
        assert!((g.ray([1.0, 1.0], [d, d], 5.0) - 0.95 * 2f64.sqrt()).abs() < 1e-6);
        assert_eq!(g.ray([1.0, 1.0], [1.0, 0.0], 0.5), 0.5);
        let (dist, q) = g.nearest_solid([1.8, 1.0], 0.3).unwrap();
        assert!((dist - 0.15).abs() < 1e-9 && (q[0] - 1.95).abs() < 1e-9);
        assert!(g.nearest_solid([1.0, 1.0], 0.3).is_none());
        let df = g.distance_field();
        assert!((df[20 * 40 + 20] - 0.95).abs() < 0.06, "{}", df[20 * 40 + 20]);
        let comp = g.component([1.0, 1.0], |k| !g.solid[k]);
        assert_eq!(comp.iter().filter(|&&c| c).count(), 38 * 38);
    }

    #[test]
    fn shape_rays_and_distances() {
        let c = Shape::Circle { c: [2.0, 0.0], r: 0.5 };
        let (a, b) = c.ray([0.0, 0.0], [1.0, 0.0]).unwrap();
        assert!((a - 1.5).abs() < 1e-9 && (b - 2.5).abs() < 1e-9);
        assert!(c.ray([0.0, 0.0], [0.0, 1.0]).is_none());
        assert_eq!(c.ray([2.0, 0.0], [1.0, 0.0]).unwrap().0, 0.0);
        let sq = Shape::Polygon(vec![[1.0, -1.0], [2.0, -1.0], [2.0, 1.0], [1.0, 1.0]]);
        assert_eq!(sq.ray([0.0, 0.0], [1.0, 0.0]), Some((1.0, 2.0)));
        assert!(sq.contains([1.5, 0.0]) && !sq.contains([0.5, 0.0]));
        assert!((sq.nearest([0.0, 0.0]).0 - 1.0).abs() < 1e-12);
        let cap = Shape::Capsule { a: [1.0, -1.0], b: [1.0, 1.0], r: 0.1 };
        let (a, _) = cap.ray([0.0, 0.5], [1.0, 0.0]).unwrap();
        assert!((a - 0.9).abs() < 1e-9);
        assert!((cap.nearest([0.0, 2.0]).0 - (2f64.sqrt() - 0.1)).abs() < 1e-9);
    }

    #[test]
    fn bodies_block_by_height_band_and_presence() {
        let mut w = World::new(room());
        let body = |kind, zmin, zmax, ir| Body {
            name: "b".into(),
            kind,
            shape: Shape::Circle { c: [1.0, 1.4], r: 0.1 },
            zmin,
            zmax,
            depth: DepthVis::Solid,
            ir,
            blocks: true,
            active: true,
        };
        w.bodies.push(body(Kind::Obstacle, 0.5, 0.9, true)); // table top: above the robot
        assert!(w.nearest([1.0, 1.0], 0.4, 0.02, 0.16).is_none());
        w.bodies.push(body(Kind::Obstacle, 0.0, 0.04, true)); // threshold the robot hits
        let c = w.nearest([1.0, 1.0], 0.4, 0.02, 0.16).unwrap();
        assert!((c.dist - 0.3).abs() < 1e-9 && c.what == What::Body(1));
        w.bodies[1].active = false;
        assert!(w.nearest([1.0, 1.0], 0.4, 0.02, 0.16).is_none());
        w.bodies[1].active = true;
        let (s, what) = w.ir_ray([1.0, 1.0], [0.0, 1.0], 0.5, 0.01, 0.08);
        assert!((s - 0.3).abs() < 1e-9 && what == Some(What::Body(1)));
        w.cliffs.push(("stairs".into(), Shape::Polygon(vec![[0.2, 0.2], [0.6, 0.2], [0.6, 0.6], [0.2, 0.6]])));
        assert_eq!(w.cliff_at([0.3, 0.3]), Some(0));
        assert_eq!(w.cliff_at([1.0, 1.0]), None);
    }
}
