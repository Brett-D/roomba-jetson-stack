//! Learned frontier ranking for the explorer: features from the global costmap and the explorer's
//! own candidates, and the ONNX ranker (twin/learned/explore/model.py) that scores them.
//!
//! The ranker only re-orders the candidates the explorer already accepted (frontier_goals + its
//! blacklist / visited / minimum-distance filters), so every existing explorer rule still applies.
//! Inputs, as built by ExploreEnv.observe (twin/learned/explore/env2d.py):
//!   map   [1, 3, 64, 64]  robot-centred, map-aligned crop, 0.2 m cells: fraction of costmap cells
//!                         free / lethal / unknown (outside the grid: unknown)
//!   glob  [1, 10]         battery, time / 1800 s, cos / sin yaw, home offset / 10 m (x, y),
//!                         candidates / 16, failed goals / 10, mapped free area / 200 m^2,
//!                         blacklist size / 10
//!   feats [1, 16, 16]     per candidate (rule order, best first): offset / 10 m (x, y),
//!                         distance / 10, path cost / 10, frontier length / 5 m, unknown area in a
//!                         3.1 m square around the frontier / 10 m^2, cos / sin of the turn to it,
//!                         distance goal-home / 10, blacklisted / visited spots within 1.5 m / 3,
//!                         rule score / 10, rank / 16, frontier centre - goal (x, y, m),
//!                         (path cost - distance) / 5
//!   mask  [1, 16]         1 = valid candidate
//! Output: logits [1, 16]; the explorer drives to the argmax.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::path::Path;

use anyhow::Result;

use super::onnx::OnnxModel;

pub const K: usize = 16;
pub const CROP: usize = 64;
pub const CROP_RES: f64 = 0.2;
pub const N_GLOB: usize = 10;
pub const N_FEAT: usize = 16;
/// Cells within this distance of a lethal cell cost twice as much on paths (Nav2-like).
const MAX_TIME_S: f64 = 1800.0;

/// One explorer candidate (map frame, metres).
#[derive(Clone, Debug)]
pub struct Candidate {
    pub goal: (f64, f64),
    pub frontier: (f64, f64),
    /// Frontier length (m): cells x resolution.
    pub size_m: f64,
    /// The explorer's score (distance - 0.01 x cells at 0.05 m).
    pub score: f64,
}

/// Explorer state the features need.
#[derive(Clone, Debug, Default)]
pub struct Context {
    pub pose: (f64, f64, f64),
    pub home: (f64, f64),
    pub battery: f64,
    pub elapsed_s: f64,
    pub failed: usize,
    pub blacklist: Vec<(f64, f64)>,
    pub visited: Vec<(f64, f64)>,
}

/// The explorer's global costmap (OccupancyGrid values: -1 unknown, 0..100).
#[derive(Clone, Copy, Debug)]
pub struct Costmap<'a> {
    pub width: usize,
    pub height: usize,
    pub resolution: f64,
    pub origin_x: f64,
    pub origin_y: f64,
    pub data: &'a [i8],
}

impl Costmap<'_> {
    fn cell(&self, x: f64, y: f64) -> Option<(usize, usize)> {
        let c = ((x - self.origin_x) / self.resolution).floor();
        let r = ((y - self.origin_y) / self.resolution).floor();
        (c >= 0.0 && r >= 0.0 && (c as usize) < self.width && (r as usize) < self.height).then(|| (r as usize, c as usize))
    }
}

#[derive(PartialEq)]
struct Item(f32, usize);
impl Eq for Item {}
impl PartialOrd for Item {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Item {
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal).then(o.1.cmp(&self.1))
    }
}

/// Path cost (cells) from `src` over passable cells (0..98: not inscribed), 8-connected, cells
/// with any inflation cost (1..98) cost 2x: ExploreEnv.distance_field.
pub fn path_costs(g: &Costmap, src: (usize, usize)) -> Vec<f32> {
    let (w, h) = (g.width, g.height);
    let mut d = vec![f32::INFINITY; w * h];
    let cost = |k: usize| -> Option<f32> {
        let v = g.data[k];
        if !(0..=98).contains(&v) {
            None
        } else if v >= 1 {
            Some(2.0)
        } else {
            Some(1.0)
        }
    };
    let s = src.0 * w + src.1;
    d[s] = 0.0;
    let mut heap = BinaryHeap::from([Item(0.0, s)]);
    let sq2 = std::f32::consts::SQRT_2;
    while let Some(Item(dk, k)) = heap.pop() {
        if dk > d[k] {
            continue;
        }
        let (r, c) = ((k / w) as i64, (k % w) as i64);
        for (dr, dc, step) in [(-1, -1, sq2), (-1, 0, 1.0), (-1, 1, sq2), (0, -1, 1.0), (0, 1, 1.0), (1, -1, sq2), (1, 0, 1.0), (1, 1, sq2)] {
            let (nr, nc) = (r + dr, c + dc);
            if nr < 0 || nc < 0 || nr >= h as i64 || nc >= w as i64 {
                continue;
            }
            let n = nr as usize * w + nc as usize;
            let Some(cn) = cost(n) else { continue };
            let nd = dk + step * cn;
            if nd < d[n] {
                d[n] = nd;
                heap.push(Item(nd, n));
            }
        }
    }
    d
}

/// Builds the ranker's four inputs. `cands` must be in the explorer's rule order (best first);
/// only the first K are used.
pub fn features(g: &Costmap, cands: &[Candidate], ctx: &Context) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let res = g.resolution;
    let (px, py, yaw) = ctx.pose;
    // map crop: fractions per 0.2 m cell of free / lethal / unknown costmap cells
    let mut map = vec![0f32; 3 * CROP * CROP];
    let mut counts = vec![0f32; CROP * CROP];
    let half = CROP as f64 * CROP_RES / 2.0;
    let x0 = px - half;
    let y0 = py - half;
    let (c0, c1) = (((x0 - g.origin_x) / res).floor() as i64, ((x0 + 2.0 * half - g.origin_x) / res).ceil() as i64);
    let (r0, r1) = (((y0 - g.origin_y) / res).floor() as i64, ((y0 + 2.0 * half - g.origin_y) / res).ceil() as i64);
    for r in r0.max(0)..r1.min(g.height as i64) {
        let cy = g.origin_y + (r as f64 + 0.5) * res;
        let i = ((cy - y0) / CROP_RES).floor();
        if i < 0.0 || i >= CROP as f64 {
            continue;
        }
        for c in c0.max(0)..c1.min(g.width as i64) {
            let cx = g.origin_x + (c as f64 + 0.5) * res;
            let j = ((cx - x0) / CROP_RES).floor();
            if j < 0.0 || j >= CROP as f64 {
                continue;
            }
            let k = i as usize * CROP + j as usize;
            let v = g.data[r as usize * g.width + c as usize];
            let ch = if v < 0 { 2 } else if v >= 100 { 1 } else { 0 };
            map[ch * CROP * CROP + k] += 1.0;
            counts[k] += 1.0;
        }
    }
    let full = (CROP_RES / res).powi(2) as f32;
    for k in 0..CROP * CROP {
        // cells outside the grid count as unknown
        let n = counts[k];
        let missing = (full - n).max(0.0);
        map[2 * CROP * CROP + k] += missing;
        let total = n + missing;
        if total > 0.0 {
            for ch in 0..3 {
                map[ch * CROP * CROP + k] /= total;
            }
        } else {
            map[2 * CROP * CROP + k] = 1.0;
        }
    }
    // path costs from the robot's cell (the nearest passable cell within 0.3 m)
    let mut dist = vec![f32::INFINITY; g.width * g.height];
    if let Some((r, c)) = g.cell(px, py) {
        let rad = (0.3 / res).round() as i64;
        let mut best: Option<(i64, usize, usize)> = None;
        for dr in -rad..=rad {
            for dc in -rad..=rad {
                let (rr, cc) = (r as i64 + dr, c as i64 + dc);
                if rr < 0 || cc < 0 || rr >= g.height as i64 || cc >= g.width as i64 {
                    continue;
                }
                let v = g.data[rr as usize * g.width + cc as usize];
                if (0..=98).contains(&v) && best.is_none_or(|b| dr * dr + dc * dc < b.0) {
                    best = Some((dr * dr + dc * dc, rr as usize, cc as usize));
                }
            }
        }
        if let Some((_, rr, cc)) = best {
            dist = path_costs(g, (rr, cc));
        }
    }
    let unknown_at = |x: f64, y: f64| g.cell(x, y).map_or(true, |(r, c)| g.data[r * g.width + c] < 0);
    let n = cands.len().min(K);
    let mut feats = vec![0f32; K * N_FEAT];
    let mut mask = vec![0f32; K];
    let near = |list: &[(f64, f64)], x: f64, y: f64| list.iter().filter(|p| (p.0 - x).hypot(p.1 - y) < 1.5).count() as f64;
    let half_box = 1.55;
    for (i, cand) in cands.iter().take(n).enumerate() {
        let (gx, gy) = cand.goal;
        let (rx, ry) = (gx - px, gy - py);
        let euc = rx.hypot(ry);
        let ang = ry.atan2(rx) - yaw;
        let path = g.cell(gx, gy).map_or(f64::INFINITY, |(r, c)| dist[r * g.width + c] as f64 * res);
        let path = if path.is_finite() { path } else { euc * 1.5 };
        // unknown area (m^2) in a 3.1 m square around the frontier centre
        let (fx, fy) = cand.frontier;
        let mut unk = 0usize;
        let steps = (2.0 * half_box / res).round() as i64;
        for a in 0..steps {
            for b in 0..steps {
                let x = fx - half_box + (b as f64 + 0.5) * res;
                let y = fy - half_box + (a as f64 + 0.5) * res;
                if unknown_at(x, y) {
                    unk += 1;
                }
            }
        }
        let gain = unk as f64 * res * res;
        let home_d = (gx - ctx.home.0).hypot(gy - ctx.home.1);
        let f = [
            rx / 10.0,
            ry / 10.0,
            euc / 10.0,
            path / 10.0,
            cand.size_m / 5.0,
            gain / 10.0,
            ang.cos(),
            ang.sin(),
            home_d / 10.0,
            near(&ctx.blacklist, gx, gy) / 3.0,
            near(&ctx.visited, gx, gy) / 3.0,
            cand.score / 10.0,
            i as f64 / K as f64,
            fx - gx,
            fy - gy,
            (path - euc) / 5.0,
        ];
        for (k, v) in f.iter().enumerate() {
            feats[i * N_FEAT + k] = *v as f32;
        }
        mask[i] = 1.0;
    }
    let known_free = g.data.iter().filter(|&&v| (0..100).contains(&v)).count() as f64 * res * res;
    let glob = vec![
        ctx.battery as f32,
        (ctx.elapsed_s / MAX_TIME_S) as f32,
        yaw.cos() as f32,
        yaw.sin() as f32,
        ((ctx.home.0 - px) / 10.0) as f32,
        ((ctx.home.1 - py) / 10.0) as f32,
        n as f32 / K as f32,
        ctx.failed as f32 / 10.0,
        (known_free / 200.0) as f32,
        ctx.blacklist.len() as f32 / 10.0,
    ];
    (map, glob, feats, mask)
}

pub struct FrontierRanker {
    model: OnnxModel,
}

/// The explorer's `goal_ranking: learned` hook.
impl crate::explorer::GoalRanker for FrontierRanker {
    fn choose(&mut self, input: &crate::explorer::RankInput, candidates: &[crate::explorer::RankCandidate]) -> Result<Option<usize>> {
        let g = input.grid;
        let cm = Costmap {
            width: g.width as usize,
            height: g.height as usize,
            resolution: g.resolution as f64,
            origin_x: g.origin_x,
            origin_y: g.origin_y,
            data: &g.data,
        };
        let cands: Vec<Candidate> = candidates
            .iter()
            .map(|c| Candidate { goal: (c.x, c.y), frontier: c.frontier, size_m: c.size_cells as f64 * cm.resolution, score: c.score })
            .collect();
        let ctx = Context {
            pose: input.pose,
            home: input.home.unwrap_or((input.pose.0, input.pose.1)),
            battery: input.battery.unwrap_or(1.0),
            elapsed_s: input.elapsed_s,
            failed: input.failed.max(0) as usize,
            blacklist: input.blacklist.to_vec(),
            visited: input.visited.to_vec(),
        };
        self.choose_among(&cm, &cands, &ctx)
    }
}

impl FrontierRanker {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Ok(FrontierRanker { model: OnnxModel::load(path, &[&[1, 3, CROP, CROP], &[1, N_GLOB], &[1, K, N_FEAT], &[1, K]])? })
    }

    pub fn logits(&self, map: &[f32], glob: &[f32], feats: &[f32], mask: &[f32]) -> Result<Vec<f32>> {
        self.model.run(&[map, glob, feats, mask])
    }

    /// Index (into `cands`) of the candidate to drive to.
    pub fn choose_among(&self, g: &Costmap, cands: &[Candidate], ctx: &Context) -> Result<Option<usize>> {
        if cands.is_empty() {
            return Ok(None);
        }
        let (map, glob, feats, mask) = features(g, cands, ctx);
        let l = self.logits(&map, &glob, &feats, &mask)?;
        let n = cands.len().min(K);
        Ok((0..n).max_by(|&a, &b| l[a].partial_cmp(&l[b]).unwrap_or(Ordering::Equal).then(b.cmp(&a))))
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
    fn path_costs_go_around_walls_and_pay_for_inflation() {
        // 10 x 10 free grid, a wall at column 5 except the top row; inflation next to it
        let (w, h) = (10, 10);
        let mut d = vec![0i8; w * h];
        for r in 1..h {
            d[r * w + 5] = 100;
            d[r * w + 4] = 50;
        }
        let g = Costmap { width: w, height: h, resolution: 0.1, origin_x: 0.0, origin_y: 0.0, data: &d };
        let c = path_costs(&g, (9, 0));
        assert!(c[9 * w + 5].is_infinite());
        let around = c[9 * w + 9];
        assert!(around > 15.0 && around.is_finite(), "{around}");
        assert_eq!(c[9 * w + 1], 1.0);
        assert_eq!(c[9 * w + 4], 3.0 + 2.0);
    }

    #[test]
    fn features_shapes_and_unknown_padding() {
        let d = vec![0i8; 100 * 100];
        let g = Costmap { width: 100, height: 100, resolution: 0.05, origin_x: 0.0, origin_y: 0.0, data: &d };
        let ctx = Context { pose: (2.5, 2.5, 0.0), home: (2.5, 2.5), battery: 1.0, ..Default::default() };
        let c = vec![Candidate { goal: (3.5, 2.5), frontier: (3.8, 2.5), size_m: 1.0, score: 0.8 }];
        let (map, glob, feats, mask) = features(&g, &c, &ctx);
        assert_eq!((map.len(), glob.len(), feats.len(), mask.len()), (3 * 64 * 64, 10, 16 * 16, 16));
        // the robot's crop cell is free; the crop's corner (6.4 m away) is off the 5 m grid: unknown
        assert!((map[32 * 64 + 32] - 1.0).abs() < 1e-6);
        assert!((map[2 * 4096] - 1.0).abs() < 1e-6);
        assert_eq!(mask[0], 1.0);
        assert!((feats[2] - 0.1).abs() < 1e-6);
        assert!((feats[3] - 0.1).abs() < 1e-3, "{}", feats[3]);
    }

    /// The exported ONNX ranker reproduces the PyTorch model on recorded inputs.
    #[test]
    fn onnx_matches_pytorch() {
        let r = FrontierRanker::load(models().join("explore_ranker.onnx")).expect("load explore_ranker.onnx");
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(models().join("explore_vectors.json")).unwrap()).unwrap();
        let f = |x: &serde_json::Value| -> Vec<f32> { x.as_array().unwrap().iter().map(|a| a.as_f64().unwrap() as f32).collect() };
        let cases = v.as_array().unwrap();
        assert!(!cases.is_empty());
        let mut worst = 0f32;
        for case in cases {
            let l = r.logits(&f(&case["map"]), &f(&case["glob"]), &f(&case["feats"]), &f(&case["mask"])).unwrap();
            let want = f(&case["logits"]);
            for (a, b) in l.iter().zip(&want) {
                if *b > -1e8 {
                    worst = worst.max((a - b).abs());
                }
            }
            assert_eq!(l.len(), want.len());
        }
        assert!(worst <= 1e-5, "max |rust - pytorch| = {worst}");
    }
}
