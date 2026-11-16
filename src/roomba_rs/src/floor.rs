//! Depth image -> floor / obstacle split, ported from roomba_perception/floor_obstacles.py.
//! Pure functions only; the ROS wiring lives in src/bin/floor_obstacles.rs.

use std::collections::HashMap;

use crate::geometry::{cross, dot, normalized, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct Intrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

/// 16-bit depth image (millimetres), as in sensor_msgs/Image 16UC1.
pub struct DepthImage<'a> {
    pub data: &'a [u8],
    pub width: usize,
    pub height: usize,
    pub step: usize,
    pub big_endian: bool,
}

/// Every `decimation`-th pixel in both directions with depth in (min_range, max_range), as 3D
/// points in the optical frame (x right, y down, z forward), row by row.
pub fn depth_points(img: &DepthImage, decimation: usize, k: Intrinsics, min_range: f64, max_range: f64) -> Vec<Vec3> {
    let d = decimation.max(1);
    let mut pts = Vec::with_capacity((img.width / d + 1) * (img.height / d + 1));
    for v in (0..img.height).step_by(d) {
        let row = &img.data[v * img.step..];
        for u in (0..img.width).step_by(d) {
            let b = [row[2 * u], row[2 * u + 1]];
            let raw = if img.big_endian { u16::from_be_bytes(b) } else { u16::from_le_bytes(b) };
            // float32 like the numpy version: depth.astype(np.float32) * 0.001
            let z = (raw as f32 * 0.001f32) as f64;
            if z > min_range && z < max_range {
                pts.push([(u as f64 - k.cx) * z / k.fx, (v as f64 - k.cy) * z / k.fy, z]);
            }
        }
    }
    pts
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    let n = v.len();
    if n == 0 {
        return None;
    }
    let mid = n / 2;
    let (_, &mut hi, _) = v.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    if n % 2 == 1 {
        return Some(hi);
    }
    let lo = v[..mid].iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Some((lo + hi) / 2.0)
}

/// Height of the floor below the camera (negative): the lowest dense 2 cm height band, refined by
/// the median of the points around it. `h` = point heights relative to the camera.
pub fn estimate_floor(h: &[f64]) -> Option<f64> {
    let below: Vec<f64> = h.iter().copied().filter(|&x| x < -0.08).collect();
    if below.len() < 50 {
        return None;
    }
    // np.arange(-3.0, -0.06, 0.02) edges and np.histogram's binning (last bin closed).
    let n_edges = ((-0.06f64 - -3.0f64) / 0.02).ceil() as usize;
    let edges: Vec<f64> = (0..n_edges).map(|i| -3.0 + i as f64 * 0.02).collect();
    let mut counts = vec![0usize; n_edges - 1];
    let last = edges[n_edges - 1];
    for &x in &below {
        if x < edges[0] || x > last {
            continue;
        }
        let mut i = edges.partition_point(|&e| e <= x) - 1;
        if i == n_edges - 1 {
            i -= 1;
        }
        counts[i] += 1;
    }
    let need = 40usize.max((0.03 * h.len() as f64) as usize);
    let k = counts.iter().position(|&c| c >= need)?;
    let lo = edges[k];
    median(below.into_iter().filter(|&x| x >= lo - 0.03 && x <= lo + 0.05).collect())
}

/// Levelled forward (optical z with the vertical part removed) and sideways directions.
pub fn floor_axes(up: Vec3) -> (Vec3, Vec3) {
    let fwd = normalized([-up[0] * up[2], -up[1] * up[2], 1.0 - up[2] * up[2]]);
    (fwd, cross(up, fwd))
}

/// Speckle filter: keep a point only if at least `min_points` points fall in its 3x3 block of
/// `cell` m floor cells (stray depth noise is 1-2 points; a wall or a thin post stacks many).
pub fn dense_mask(pts: &[Vec3], up: Vec3, cell: f64, min_points: usize) -> Vec<bool> {
    let (fwd, side) = floor_axes(up);
    let keys: Vec<(i64, i64)> = pts
        .iter()
        .map(|&p| ((dot(p, fwd) / cell).floor() as i64, (dot(p, side) / cell).floor() as i64))
        .collect();
    let mut counts: HashMap<(i64, i64), usize> = HashMap::with_capacity(keys.len());
    for &k in &keys {
        *counts.entry(k).or_insert(0) += 1;
    }
    let mut total: HashMap<(i64, i64), usize> = HashMap::with_capacity(counts.len());
    for &(i, j) in counts.keys() {
        let mut t = 0;
        for di in -1..=1 {
            for dj in -1..=1 {
                t += counts.get(&(i + di, j + dj)).copied().unwrap_or(0);
            }
        }
        total.insert((i, j), t);
    }
    keys.iter().map(|k| total[k] >= min_points).collect()
}

/// floor_obstacles.py drop_points: negative obstacles. Points more than `below` m under the floor
/// plane and closer than max_range m (depth noise grows with range) were seen past a drop (through a stair railing's balusters, over a landing's edge, to the
/// floor below), which the obstacle band above the floor never reports, so the drop was mapped as
/// floor. The floor ends where each such point's line of sight crosses the floor plane: a point is
/// reported there, `lift` m above the floor. Only where at least `min_points` of them fall in their
/// 3x3 block of `cell` m floor cells (`fwd`, `side`: floor_axes) - stray deep pixels are noise.
/// `h`: the points' heights relative to the camera, `floor_h`: the floor's (negative). Exactly the
/// numpy arithmetic (elementwise, no fused multiply-add), as tools/ab_floor_drops_offline.py checks.
#[allow(clippy::too_many_arguments)]
pub fn drop_points(
    pts: &[Vec3],
    h: &[f64],
    floor_h: f64,
    up: Vec3,
    below: f64,
    max_range: f64,
    lift: f64,
    fwd: Vec3,
    side: Vec3,
    cell: f64,
    min_points: usize,
) -> Vec<Vec3> {
    let limit = floor_h - below;
    let lifted = [lift * up[0], lift * up[1], lift * up[2]];
    let q: Vec<Vec3> = pts
        .iter()
        .zip(h)
        .filter(|&(p, &hp)| hp < limit && p[2] < max_range)
        .map(|(p, &hp)| {
            let t = floor_h / hp;
            [p[0] * t + lifted[0], p[1] * t + lifted[1], p[2] * t + lifted[2]]
        })
        .collect();
    let key = |p: &Vec3| {
        (
            ((p[0] * fwd[0] + p[1] * fwd[1] + p[2] * fwd[2]) / cell).floor() as i64,
            ((p[0] * side[0] + p[1] * side[1] + p[2] * side[2]) / cell).floor() as i64,
        )
    };
    let keys: Vec<(i64, i64)> = q.iter().map(key).collect();
    let mut counts: HashMap<(i64, i64), usize> = HashMap::with_capacity(keys.len());
    for &k in &keys {
        *counts.entry(k).or_insert(0) += 1;
    }
    let near = |(i, j): (i64, i64)| -> usize {
        (-1..=1).flat_map(|di| (-1..=1).map(move |dj| (i + di, j + dj))).map(|c| counts.get(&c).copied().unwrap_or(0)).sum()
    };
    q.into_iter().zip(keys).filter(|&(_, k)| near(k) >= min_points).map(|(p, _)| p).collect()
}

/// Short wall `distance` m ahead of the camera, 0.2 m wide, used when depth sees nothing.
pub fn blind_points(up: Vec3, distance: f64) -> Vec<Vec3> {
    let (fwd, side) = floor_axes(up);
    [-0.1, -0.05, 0.0, 0.05, 0.1]
        .iter()
        .map(|&o| [distance * fwd[0] + o * side[0], distance * fwd[1] + o * side[1], distance * fwd[2] + o * side[2]])
        .collect()
}

/// PointCloud2 payload: x, y, z as little-endian float32, 12 bytes per point.
pub fn cloud_bytes<'a>(pts: impl Iterator<Item = &'a Vec3>) -> Vec<u8> {
    let mut out = Vec::new();
    for p in pts {
        for c in p {
            out.extend_from_slice(&(*c as f32).to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const UP: Vec3 = [0.0, -1.0, 0.0]; // level camera: optical -y is up

    #[test]
    fn depth_points_decimates_and_filters_range() {
        // 4x3 image, step 8 bytes; depths in mm. Decimation 2 keeps rows 0, 2 and columns 0, 2.
        let mm: [u16; 12] = [500, 0, 1000, 5000, 9, 9, 9, 9, 250, 600, 700, 800];
        let data: Vec<u8> = mm.iter().flat_map(|v| v.to_le_bytes()).collect();
        let img = DepthImage { data: &data, width: 4, height: 3, step: 8, big_endian: false };
        let k = Intrinsics { fx: 100.0, fy: 100.0, cx: 2.0, cy: 1.0 };
        let p = depth_points(&img, 2, k, 0.3, 4.0);
        // (u,v) = (0,0) 500, (2,0) 1000, (0,2) 250 (too close), (2,2) 700.
        assert_eq!(p.len(), 3);
        assert!((p[0][0] - (0.0 - 2.0) * 0.5 / 100.0).abs() < 1e-6 && (p[0][2] - 0.5).abs() < 1e-6);
        assert!((p[2][1] - (2.0 - 1.0) * 0.7 / 100.0).abs() < 1e-6 && (p[2][2] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn floor_is_lowest_dense_band() {
        let mut h = vec![-0.11; 400];                    // floor 11 cm below the camera
        h.extend(std::iter::repeat(-0.5).take(30));      // a few points lower (too sparse)
        h.extend(std::iter::repeat(0.2).take(500));      // walls above
        let f = estimate_floor(&h).unwrap();
        assert!((f + 0.11).abs() < 1e-9, "{f}");
        assert_eq!(estimate_floor(&[0.1; 100]), None);
    }

    #[test]
    fn speckle_filter_drops_isolated_points() {
        let mut pts: Vec<Vec3> = (0..10).map(|i| [0.02, -0.05 * i as f64 / 10.0, 1.0]).collect(); // a post
        pts.push([0.8, 0.0, 2.0]);                                                            // a speck
        let keep = dense_mask(&pts, UP, 0.05, 4);
        assert!(keep[..10].iter().all(|&k| k));
        assert!(!keep[10]);
    }

    #[test]
    fn drop_points_mark_where_the_floor_ends() {
        // Level camera 0.11 m above the floor; past 1 m the floor drops away (a stairwell): its far
        // wall is seen 0.5 m below the floor at 1.6 m, a 10 x 6 patch of points.
        let mut pts: Vec<Vec3> = Vec::new();
        for i in 0..10 {
            for k in 0..6 {
                pts.push([-0.1 + 0.02 * i as f64, 0.61, 1.6 + 0.01 * k as f64]);
            }
        }
        pts.push([2.0, 0.61, 1.6]); // a stray one elsewhere
        pts.push([0.0, 0.11, 1.0]); // floor
        pts.push([0.0, 0.61, 2.5]); // too far
        let h: Vec<f64> = pts.iter().map(|&p| dot(p, UP)).collect();
        let (fwd, side) = floor_axes(UP);
        let d = drop_points(&pts, &h, -0.11, UP, 0.12, 2.0, 0.1, fwd, side, 0.05, 6);
        assert_eq!(d.len(), 60);
        // On the line of sight, where it crosses the floor plane (1.6 * 0.11 / 0.61 = 0.29 m ahead),
        // lifted 0.1 m above the floor.
        assert!((d[0][2] - 1.6 * 0.11 / 0.61).abs() < 1e-9 && (dot(d[0], UP) - (-0.11 + 0.1)).abs() < 1e-9, "{:?}", d[0]);
        assert!(drop_points(&pts, &h, -0.11, UP, 0.12, 2.0, 0.1, fwd, side, 0.05, 61).is_empty());
    }

    #[test]
    fn blind_wall_is_ahead_and_level() {
        let p = blind_points(UP, 0.15);
        assert_eq!(p.len(), 5);
        assert!(p.iter().all(|q| (q[2] - 0.15).abs() < 1e-12 && q[1].abs() < 1e-12));
        assert!((p[0][0] - p[4][0]).abs() > 0.19);
    }
}