//! The unstuck policy's view of the local costmap: a robot-centred 24 x 24 patch of 0.1 m cells,
//! 1 where the costmap is lethal. Row 0 is the farthest ahead, column 0 the leftmost; each cell is
//! the max over 2 x 2 sample points (so 1-cell obstacles of the 0.05 m costmap are not missed).
//! Samples farther than the rolling window (+-1.5 m on the costmap axes) or outside the grid are 0.
//! Same as UnstuckEnv.costmap_patch / patch_offsets in twin/learned/unstuck/unstuck_env.py.

pub const PATCH: usize = 24;
pub const PATCH_RES: f64 = 0.1;
pub const WINDOW: f64 = 1.5;
/// nav_msgs/OccupancyGrid value of a lethal cell (costmap_2d publishes LETHAL_OBSTACLE 254 as 100;
/// keep-out zones are lethal too). Inscribed (99) and inflation costs are not obstacles here.
pub const LETHAL: i8 = 100;

/// An OccupancyGrid in its frame (the local costmap: odom), row-major, row = y.
#[derive(Clone, Copy, Debug)]
pub struct GridView<'a> {
    pub width: usize,
    pub height: usize,
    pub resolution: f64,
    pub origin_x: f64,
    pub origin_y: f64,
    pub data: &'a [i8],
}

impl GridView<'_> {
    /// Cell value at a point, None outside the grid.
    pub fn at(&self, x: f64, y: f64) -> Option<i8> {
        let c = ((x - self.origin_x) / self.resolution).floor();
        let r = ((y - self.origin_y) / self.resolution).floor();
        if c < 0.0 || r < 0.0 || c >= self.width as f64 || r >= self.height as f64 {
            return None;
        }
        self.data.get(r as usize * self.width + c as usize).copied()
    }
}

/// Sample offsets (forward, left) in the robot frame, 4 per patch cell, in cell order.
pub fn offsets() -> Vec<[f64; 2]> {
    let q = PATCH_RES / 4.0;
    let mut out = Vec::with_capacity(PATCH * PATCH * 4);
    for i in 0..PATCH {
        let u = (PATCH as f64 / 2.0 - 0.5 - i as f64) * PATCH_RES;
        for j in 0..PATCH {
            let v = (PATCH as f64 / 2.0 - 0.5 - j as f64) * PATCH_RES;
            for (du, dv) in [(q, q), (q, -q), (-q, q), (-q, -q)] {
                out.push([u + du, v + dv]);
            }
        }
    }
    out
}

/// The lethal patch for a robot at (x, y, yaw) in the grid's frame.
pub fn lethal_patch(g: &GridView, x: f64, y: f64, yaw: f64) -> Vec<f32> {
    let (c, s) = (yaw.cos(), yaw.sin());
    let mut out = vec![0f32; PATCH * PATCH];
    for (k, o) in offsets().iter().enumerate() {
        let wx = c * o[0] - s * o[1];
        let wy = s * o[0] + c * o[1];
        if wx.abs() > WINDOW || wy.abs() > WINDOW {
            continue;
        }
        if g.at(x + wx, y + wy).is_some_and(|v| v >= LETHAL) {
            out[k / 4] = 1.0;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hits(p: &[f32]) -> Vec<(usize, usize)> {
        (0..PATCH * PATCH).filter(|&k| p[k] > 0.0).map(|k| (k / PATCH, k % PATCH)).collect()
    }

    #[test]
    fn obstacle_ahead_lands_in_the_top_rows_and_left_in_the_left_columns() {
        let (w, h) = (60, 60);
        let mut d = vec![0i8; w * h];
        // a lethal cell 0.6 m ahead (+x) of a robot at the origin facing +x, one 0.5 m to its left
        d[30 * w + 42] = 100;
        d[40 * w + 30] = 100;
        let g = GridView { width: w, height: h, resolution: 0.05, origin_x: -1.5, origin_y: -1.5, data: &d };
        let h0 = hits(&lethal_patch(&g, 0.0, 0.0, 0.0));
        // ahead: u = 0.6 -> rows 5/6, v = 0 -> cols 11/12; left: v = 0.5 -> cols 6/7, rows 11/12
        assert!(h0.iter().any(|&(r, c)| (5..=6).contains(&r) && (11..=12).contains(&c)), "{h0:?}");
        assert!(h0.iter().any(|&(r, c)| (11..=12).contains(&r) && (6..=7).contains(&c)), "{h0:?}");
        // turned left by 90 deg, the +x obstacle is on the robot's right
        let h1 = hits(&lethal_patch(&g, 0.0, 0.0, std::f64::consts::FRAC_PI_2));
        assert!(h1.iter().any(|&(r, c)| (11..=12).contains(&r) && (17..=18).contains(&c)), "{h1:?}");
    }

    #[test]
    fn inscribed_and_unknown_are_not_lethal() {
        let d = vec![99i8; 400];
        let g = GridView { width: 20, height: 20, resolution: 0.05, origin_x: -0.5, origin_y: -0.5, data: &d };
        assert!(lethal_patch(&g, 0.0, 0.0, 0.3).iter().all(|&v| v == 0.0));
        let d = vec![-1i8; 400];
        let g = GridView { data: &d, ..g };
        assert!(lethal_patch(&g, 0.0, 0.0, 0.3).iter().all(|&v| v == 0.0));
    }
}
