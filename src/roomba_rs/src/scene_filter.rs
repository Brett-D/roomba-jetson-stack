//! scene_filter.py's helpers for deciding whether a detection should be ignored (used by the person
//! detector): user-drawn exclusion zones and line of sight on RTAB-Map's 2D occupancy grid. All
//! geometry is 2D in the map frame. (apply_transform is [`crate::tf_buffer::Transform::apply`],
//! quat_rotate is [`crate::geometry::quat_rotate`].) Same float operations as the Python, so the
//! results are identical (tools/ab_scene_filter.py).

use std::collections::HashSet;

use serde_json::Value;

use crate::zones::py_str;

/// Ray casting point-in-polygon test (edges on the boundary as the Python computes them).
pub fn point_in_polygon(x: f64, y: f64, pts: &[[f64; 2]]) -> bool {
    let mut inside = false;
    let n = pts.len();
    if n == 0 {
        return false;
    }
    let mut j = n - 1;
    for i in 0..n {
        let ([xi, yi], [xj, yj]) = (pts[i], pts[j]);
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

pub struct Zone {
    pub name: String,
    pub points: Vec<[f64; 2]>,
}

/// User-drawn ignore zones from /perception/ignore_zones ({"frame": "map", "zones": [...]}).
/// Navigation keep-out zones share the message but are not ignore zones.
pub struct ZoneSet {
    pub frame: String,
    pub zones: Vec<Zone>,
}

impl Default for ZoneSet {
    fn default() -> Self {
        ZoneSet { frame: "map".into(), zones: Vec::new() }
    }
}

impl ZoneSet {
    pub fn load_json(&mut self, text: &str) -> Result<(), String> {
        let data: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let data = data.as_object().ok_or("not a JSON object")?;
        let frame = match data.get("frame") {
            None => "map".to_string(),
            Some(Value::String(s)) => s.clone(),
            Some(other) => py_str(other),
        };
        let mut zones = Vec::new();
        for z in data.get("zones").and_then(Value::as_array).into_iter().flatten() {
            let Some(z) = z.as_object() else { continue };
            let pts = z.get("points").and_then(Value::as_array).cloned().unwrap_or_default();
            if pts.len() < 3 || z.get("kind").is_some_and(|k| k == "keepout") {
                continue;
            }
            let points = pts
                .iter()
                .filter_map(|p| Some([p.get(0)?.as_f64()?, p.get(1)?.as_f64()?]))
                .collect();
            let name = match z.get("name") {
                None => "zone".to_string(),
                Some(Value::String(s)) => s.clone(),
                Some(other) => py_str(other),
            };
            zones.push(Zone { name, points });
        }
        self.frame = frame;
        self.zones = zones;
        Ok(())
    }

    /// Name of the first zone containing (x, y).
    pub fn containing(&self, x: f64, y: f64) -> Option<&str> {
        self.zones.iter().find(|z| point_in_polygon(x, y, &z.points)).map(|z| z.name.as_str())
    }
}

/// Whether occupied cells of an occupancy grid block a 2D ray.
pub struct GridLineOfSight {
    pub occupied_threshold: i64,
    pub min_blocking_cells: i64,
    grid: Vec<i8>,
    width: usize,
    height: usize,
    res: f64,
    ox: f64,
    oy: f64,
    yaw: f64,
    pub frame: Option<String>,
}

impl GridLineOfSight {
    pub fn new(occupied_threshold: i64, min_blocking_cells: i64) -> Self {
        GridLineOfSight {
            occupied_threshold,
            min_blocking_cells,
            grid: Vec::new(),
            width: 0,
            height: 0,
            res: 0.0,
            ox: 0.0,
            oy: 0.0,
            yaw: 0.0,
            frame: None,
        }
    }

    /// A new map (nav_msgs/OccupancyGrid fields; `orientation` = (x, y, z, w)).
    pub fn update(&mut self, width: usize, height: usize, resolution: f32, origin: [f64; 2], orientation: [f64; 4], frame: &str, data: &[i8]) {
        let [x, y, z, w] = orientation;
        self.grid = data.to_vec();
        self.width = width;
        self.height = height;
        self.res = resolution as f64;
        self.ox = origin[0];
        self.oy = origin[1];
        self.yaw = (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z));
        self.frame = Some(frame.to_string());
    }

    pub fn ready(&self) -> bool {
        self.frame.is_some()
    }

    fn cell(&self, x: f64, y: f64) -> (i64, i64) {
        let (dx, dy) = (x - self.ox, y - self.oy);
        let (c, s) = ((-self.yaw).cos(), (-self.yaw).sin());
        let (gx, gy) = (c * dx - s * dy, s * dx + c * dy);
        ((gx / self.res).floor() as i64, (gy / self.res).floor() as i64)
    }

    /// Number of distinct occupied cells crossed between start and end, excluding skip_start_m near
    /// the camera and skip_end_m around the target (its own body). Python defaults: 0.15, 0.35.
    pub fn blocked(&self, start: [f64; 2], end: [f64; 2], skip_start_m: f64, skip_end_m: f64) -> usize {
        if !self.ready() {
            return 0;
        }
        let ([sx, sy], [ex, ey]) = (start, end);
        let length = (ex - sx).hypot(ey - sy);
        if length <= skip_start_m + skip_end_m {
            return 0;
        }
        let step = self.res * 0.5;
        let n = ((length - skip_start_m - skip_end_m) / step) as i64 + 1;
        let (ux, uy) = ((ex - sx) / length, (ey - sy) / length);
        let mut hits = HashSet::new();
        for i in 0..n {
            let d = skip_start_m + i as f64 * step;
            let (cx, cy) = self.cell(sx + ux * d, sy + uy * d);
            if 0 <= cx && (cx as usize) < self.width && 0 <= cy && (cy as usize) < self.height {
                let v = self.grid[cy as usize * self.width + cx as usize];
                if v as i64 >= self.occupied_threshold {
                    hits.insert((cx, cy));
                }
            }
        }
        hits.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zones_skip_keepout_and_small() {
        let mut z = ZoneSet::default();
        z.load_json(
            r#"{"frame": "map", "zones": [{"name": "nav", "kind": "keepout", "points": [[0,0],[2,0],[2,2]]},
                {"name": "two", "points": [[0,0],[1,0]]}, {"name": "tv", "points": [[0,0],[2,0],[2,2],[0,2]]}]}"#,
        )
        .unwrap();
        assert_eq!(z.containing(1.0, 0.5), Some("tv"));
        assert_eq!(z.containing(3.0, 0.5), None);
    }

    #[test]
    fn wall_blocks_the_ray() {
        let mut g = GridLineOfSight::new(65, 2);
        let mut data = vec![0i8; 40 * 40];
        for y in 0..40 {
            data[y * 40 + 20] = 100;
        }
        g.update(40, 40, 0.05, [0.0, 0.0], [0.0, 0.0, 0.0, 1.0], "map", &data);
        assert!(g.blocked([0.2, 1.0], [1.9, 1.0], 0.15, 0.35) >= 1);
        assert_eq!(g.blocked([0.2, 1.0], [0.2, 1.9], 0.15, 0.35), 0);
    }
}
