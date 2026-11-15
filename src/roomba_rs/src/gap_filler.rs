//! map_gap_filler's processing: joins dotted obstacles in RTAB-Map's occupancy grid (a railing's
//! posts) so the planner can't route between them. Same operations as
//! roomba_perception/map_gap_filler.py (OpenCV/numpy reproduced exactly in [`crate::cv`]), so the
//! output grid is identical cell for cell (tools/ab_map_gap_filler_offline.py).

use std::collections::HashSet;

use crate::cv::{argsort, close_rect, components8, round_half_even, Canvas};

#[derive(Clone, Debug)]
pub struct Params {
    pub gap_m: f64,
    pub occupied: i64,
    pub join_fences: bool,
    pub spacing_m: f64,
    pub tolerance_m: f64,
    pub min_posts: i64,
    pub max_dot_cells: i64,
    /// Robot radius: no fence link may cross within this of the robot's path.
    pub path_clear_radius_m: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            gap_m: 0.10,
            occupied: 65,
            join_fences: true,
            spacing_m: 0.45,
            tolerance_m: 0.06,
            min_posts: 3,
            max_dot_cells: 6,
            path_clear_radius_m: 0.17,
        }
    }
}

/// The parts of a nav_msgs/OccupancyGrid used here.
pub struct GridInfo {
    pub width: usize,
    pub height: usize,
    /// As in the message (float32); used as a double like rclpy does.
    pub resolution: f32,
    pub origin_x: f64,
    pub origin_y: f64,
}

/// fill_gaps(): closes gaps of up to gap_m between occupied cells (sets them to 100). Returns the
/// new grid and how many cells were added.
pub fn fill_gaps(grid: &[i8], info: &GridInfo, gap_m: f64, occupied: i64) -> (Vec<i8>, usize) {
    let (w, h) = (info.width, info.height);
    let occ: Vec<u8> = grid.iter().map(|&v| (v as i64 >= occupied) as u8).collect();
    let r = ((gap_m / info.resolution as f64 / 2.0).ceil() as i64).max(1) as usize;
    let closed = close_rect(&occ, w, h, r);
    let mut out = grid.to_vec();
    let mut added = 0;
    for i in 0..grid.len() {
        if closed[i] > 0 && occ[i] == 0 {
            out[i] = 100;
            added += 1;
        }
    }
    (out, added)
}

/// fence_links(): pairs of dot centres (cells, (x, y)) to join: runs of >= min_posts small occupied
/// blobs on a straight line, consecutive ones at most spacing_m apart, not crossing `keep_clear`.
pub fn fence_links(grid: &[i8], info: &GridInfo, p: &Params, keep_clear: Option<&[u8]>) -> Vec<((i64, i64), (i64, i64))> {
    let (w, h) = (info.width, info.height);
    let occ: Vec<u8> = grid.iter().map(|&v| (v as i64 >= p.occupied) as u8).collect();
    let dots: Vec<[f64; 2]> = components8(&occ, w, h)
        .into_iter()
        .filter(|c| c.area as i64 <= p.max_dot_cells)
        .map(|c| [c.cx, c.cy])
        .collect();
    if (dots.len() as i64) < p.min_posts {
        return Vec::new();
    }
    let res = info.resolution as f64;
    let (max_link, tol) = (p.spacing_m / res, p.tolerance_m / res);
    let reach = max_link * p.min_posts as f64;
    let n = dots.len();
    let mut links: HashSet<(usize, usize)> = HashSet::new();
    let mut along = vec![0.0; n];
    let mut across = vec![0.0; n];
    for i in 0..n {
        for j in i + 1..n {
            let (dx, dy) = (dots[j][0] - dots[i][0], dots[j][1] - dots[i][1]);
            let d = dx.hypot(dy);
            if !(d >= 1.5 && d <= max_link) {
                continue;
            }
            let u = [dx / d, dy / d];
            let a = dots[i];
            // `off @ u` with the reference BLAS dgemv numpy calls: second product fused.
            for k in 0..n {
                let off = [dots[k][0] - a[0], dots[k][1] - a[1]];
                along[k] = off[1].mul_add(u[1], off[0] * u[0]);
                across[k] = off[1].mul_add(u[0], off[0] * -u[1]);
            }
            let (lo_bound, hi_bound) = (-d - reach, 2.0 * d + reach);
            let on: Vec<usize> =
                (0..n).filter(|&k| across[k].abs() <= tol && along[k] >= lo_bound && along[k] <= hi_bound).collect();
            if (on.len() as i64) < p.min_posts {
                continue;
            }
            let order = argsort(&on.iter().map(|&k| along[k]).collect::<Vec<_>>());
            let on: Vec<usize> = order.iter().map(|&o| on[o]).collect();
            let pos: Vec<f64> = on.iter().map(|&k| along[k]).collect();
            let (Some(lo), Some(hi)) = (on.iter().position(|&k| k == i), on.iter().position(|&k| k == j)) else {
                continue;
            };
            let (mut lo, mut hi) = (lo as isize, hi as isize);
            if lo <= hi && pos[lo as usize..=hi as usize].windows(2).any(|s| s[1] - s[0] > max_link) {
                continue;
            }
            while lo > 0 && pos[lo as usize] - pos[lo as usize - 1] <= max_link {
                lo -= 1;
            }
            while hi < on.len() as isize - 1 && pos[hi as usize + 1] - pos[hi as usize] <= max_link {
                hi += 1;
            }
            if ((hi - lo + 1) as i64) < p.min_posts {
                continue;
            }
            for k in lo..hi {
                let (a, b) = (on[k as usize], on[k as usize + 1]);
                links.insert((a.min(b), a.max(b)));
            }
        }
    }
    let cell = |d: [f64; 2]| (round_half_even(d[0]) as i64, round_half_even(d[1]) as i64);
    let mut out = Vec::new();
    for (i, j) in links {
        let (pi, pj) = (cell(dots[i]), cell(dots[j]));
        if let Some(clear) = keep_clear {
            let crosses = crate::cv::line_points(w as i64, h as i64, pi, pj)
                .iter()
                .any(|&(x, y)| clear[y as usize * w + x as usize] != 0);
            if crosses {
                continue;
            }
        }
        out.push((pi, pj));
    }
    out
}

/// join_fences(): the grid with fence_links() drawn as occupied cells, and the number of links.
pub fn join_fences(grid: &[i8], info: &GridInfo, p: &Params, keep_clear: Option<&[u8]>) -> (Vec<i8>, usize) {
    let links = fence_links(grid, info, p, keep_clear);
    if links.is_empty() {
        return (grid.to_vec(), 0);
    }
    let mut mask = Canvas::new(info.width, info.height);
    for &(a, b) in &links {
        mask.line(a, b, 1);
    }
    let out = grid.iter().zip(&mask.data).map(|(&g, &m)| if m > 0 && g < 65 { 100 } else { g }).collect();
    (out, links.len())
}

/// Cells within path_clear_radius_m of the robot's path (cv2.polylines with a thick pen), or None
/// without a path.
pub fn driven_mask(path: &[(f64, f64)], info: &GridInfo, clear_radius_m: f64) -> Option<Vec<u8>> {
    if path.is_empty() {
        return None;
    }
    let res = info.resolution as f64;
    let r = (round_half_even(clear_radius_m / res) as i64).max(1);
    let pts: Vec<(i64, i64)> = path
        .iter()
        .map(|&(x, y)| {
            (round_half_even((x - info.origin_x) / res) as i32 as i64, round_half_even((y - info.origin_y) / res) as i32 as i64)
        })
        .collect();
    let mut mask = Canvas::new(info.width, info.height);
    mask.polyline_thick(&pts, 2 * r + 1, 1);
    Some(mask.data)
}

/// MapGapFiller's state and per-message processing (without ROS).
pub struct GapFiller {
    pub params: Params,
    path: Vec<(f64, f64)>,
    logged: bool,
    fences: i64,
}

/// What one map produces.
pub struct Filled {
    pub data: Vec<i8>,
    /// Log lines: (throttled to one per 30 s, text).
    pub logs: Vec<(bool, String)>,
}

impl GapFiller {
    pub fn new(params: Params) -> Self {
        GapFiller { params, path: Vec::new(), logged: false, fences: -1 }
    }

    /// /rtabmap/mapPath: the robot's path in the map frame.
    pub fn on_path(&mut self, poses: Vec<(f64, f64)>) {
        self.path = poses;
    }

    /// /rtabmap/map. None: publish the input unchanged (empty map).
    pub fn on_map(&mut self, info: &GridInfo, data: &[i8]) -> Option<Filled> {
        if info.width == 0 || info.height == 0 {
            return None;
        }
        let p = &self.params;
        let (mut out, added) = fill_gaps(data, info, p.gap_m, p.occupied);
        let mut logs = Vec::new();
        if p.join_fences {
            // Detect on the raw grid (the closing would merge dots into blobs), then add both.
            let keep_clear = driven_mask(&self.path, info, p.path_clear_radius_m);
            let (fenced, links) = join_fences(data, info, p, keep_clear.as_deref());
            for (o, f) in out.iter_mut().zip(fenced) {
                *o = (*o).max(f);
            }
            if links as i64 != self.fences {
                self.fences = links as i64;
                logs.push((true, format!("{links} fence link(s) joining lined-up dots")));
            }
        }
        if !self.logged {
            self.logged = true;
            logs.push((
                false,
                format!("Filling gaps up to {:.2} m between obstacles ({added} cells added to the first map)", self.params.gap_m),
            ));
        }
        Some(Filled { data: out, logs })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(w: usize, h: usize) -> GridInfo {
        GridInfo { width: w, height: h, resolution: 0.05, origin_x: 0.0, origin_y: 0.0 }
    }

    /// A row of single-cell posts 7 cells (0.35 m) apart along y = 10.
    fn posts(w: usize, h: usize) -> Vec<i8> {
        let mut g = vec![0i8; w * h];
        for x in (5..40).step_by(7) {
            g[10 * w + x] = 100;
        }
        g
    }

    #[test]
    fn joins_lined_up_posts() {
        let (w, h) = (50, 20);
        let (out, links) = join_fences(&posts(w, h), &info(w, h), &Params::default(), None);
        assert_eq!(links, 4);
        assert!((5..=33).all(|x| out[10 * w + x] == 100));
        assert_eq!(out[10 * w + 4], 0);
    }

    #[test]
    fn never_across_the_robots_path() {
        let (w, h) = (50, 20);
        // The robot drove across the row at x = 1.0 m (cell 20): posts 12-19 and 19-26 stay apart.
        let mut f = GapFiller::new(Params::default());
        f.on_path(vec![(1.0, 0.0), (1.0, 0.95)]);
        let out = f.on_map(&info(w, h), &posts(w, h)).unwrap();
        assert_eq!(out.data[10 * w + 22], 0);
        assert_eq!(out.data[10 * w + 15], 0);
        assert_eq!(out.data[10 * w + 8], 100);
        assert_eq!(out.logs[0].1, "2 fence link(s) joining lined-up dots");
    }

    #[test]
    fn closing_fills_one_cell_gaps_and_keeps_doorways() {
        let (w, h) = (20, 5);
        let mut g = vec![0i8; w * h];
        for x in [2, 4, 10, 16] {
            g[2 * w + x] = 100;
        }
        let (out, added) = fill_gaps(&g, &info(w, h), 0.10, 65);
        assert_eq!(added, 1);
        assert_eq!(out[2 * w + 3], 100);
        assert_eq!(out[2 * w + 12], 0);
    }

    #[test]
    fn empty_map_passes_through() {
        assert!(GapFiller::new(Params::default()).on_map(&info(0, 0), &[]).is_none());
    }
}
