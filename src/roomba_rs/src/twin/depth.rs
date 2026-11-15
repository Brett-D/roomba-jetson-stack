//! The simulated MYNT D1000 depth camera and IMU.
//!
//! Depth: 16UC1 millimetres, registered to the left camera (camera_left_optical_frame), the real
//! intrinsics (640 x 360, f 349.2), valid 0.3-7 m, 0 elsewhere. Rendered by 2.5D ray casting: for
//! each image column a 2D ray across the floor plan finds the walls (full height, up to the 2.5 m
//! ceiling) and the bodies it passes (doors, people, obstacles with a height range, railing posts
//! when the stereo happens to see them); each pixel's ray then hits the first of those whose
//! height range it crosses, else the floor or the ceiling. Over a cliff (stairs going down) the
//! floor is CLIFF_DROP_M lower, so the camera sees down into it, not an obstacle. The camera is
//! assumed level (cam_pitch 0, as mounted).
//!
//! IMU: angular velocity and specific force of the robot's planar motion in camera_imu_frame,
//! with gravity, a constant gyro bias and white noise like the real one.

use crate::geometry::{quat_conj, quat_mul, quat_rotate, Quat, Vec3};

use super::rng::Rng;
use super::world::{DepthVis, World, P2, WALL_HEIGHT_M};

/// How far the floor drops inside a cliff area (stairs going down), as the depth camera sees it.
pub const CLIFF_DROP_M: f64 = 1.0;

/// Pinhole intrinsics (rectified, no distortion).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Intrinsics {
    pub width: usize,
    pub height: usize,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

/// The D1000 depth stream as recorded on the robot (/camera/depth/camera_info).
pub const D1000: Intrinsics =
    Intrinsics { width: 640, height: 360, fx: 349.20001220703125, fy: 349.20001220703125, cx: 318.03509521484375, cy: 174.52203369140625 };

/// Static transforms the camera driver publishes (from the recording): camera_link ->
/// camera_left_optical_frame and camera_right_optical_frame, camera_left_optical_frame -> camera_imu_frame.
pub const LEFT_OPTICAL_Y: f64 = 0.059188953399658205;
pub const Q_LINK_OPTICAL: Quat = [-0.5, 0.5, -0.5, 0.5];
pub const IMU_IN_OPTICAL: Vec3 = [0.0477928465820604, -0.0020221539578519996, -0.0012811425097398002];
pub const Q_OPTICAL_IMU: Quat = [0.9999874724368888, 0.0021644046093707487, 0.0034723934920541906, -0.002883193612270431];

#[derive(Clone, Debug)]
pub struct RenderOpts {
    pub min_range: f64,
    pub max_range: f64,
    /// Reported depth = true depth x scale.
    pub scale: f64,
    pub noise_a: f64,
    pub noise_b: f64,
    pub dropout: f64,
}

impl Default for RenderOpts {
    fn default() -> Self {
        RenderOpts { min_range: 0.3, max_range: 7.0, scale: 1.0, noise_a: 0.0, noise_b: 0.0, dropout: 0.0 }
    }
}

/// What a pixel saw (for the preview image).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
    Nothing,
    Floor,
    Ceiling,
    Wall,
    Body(u8),
}

pub struct Camera {
    pub k: Intrinsics,
    /// Per column: x_n = (u - cx) / fx, the column's azimuth (rad, left positive) and 1/sqrt(1 + x_n^2).
    azimuth: Vec<f64>,
    inv: Vec<f64>,
    /// Per row: y_n = (v - cy) / fy.
    yn: Vec<f64>,
}

/// Where the camera is: the left lens in the map frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LensPose {
    pub x: f64,
    pub y: f64,
    /// Height above the floor (m).
    pub z: f64,
    pub yaw: f64,
}

/// The left lens for a robot pose and the mount (base_link -> camera_link translation, level).
pub fn lens_pose(robot: [f64; 3], mount: Vec3) -> LensPose {
    let (c, s) = (robot[2].cos(), robot[2].sin());
    let (lx, ly) = (mount[0], mount[1] + LEFT_OPTICAL_Y);
    LensPose { x: robot[0] + c * lx - s * ly, y: robot[1] + s * lx + c * ly, z: mount[2], yaw: robot[2] }
}

impl Camera {
    pub fn new(k: Intrinsics) -> Self {
        let mut azimuth = Vec::with_capacity(k.width);
        let mut inv = Vec::with_capacity(k.width);
        for u in 0..k.width {
            let xn = (u as f64 - k.cx) / k.fx;
            azimuth.push((-xn).atan());
            inv.push(1.0 / (1.0 + xn * xn).sqrt());
        }
        let yn = (0..k.height).map(|v| (v as f64 - k.cy) / k.fy).collect();
        Camera { k, azimuth, inv, yn }
    }

    /// The 3D point of pixel (u, v) at optical depth z_m, in the lens's level frame
    /// (x forward, y left, z up relative to the lens).
    pub fn point(&self, u: usize, v: usize, z_m: f64) -> Vec3 {
        let xn = (u as f64 - self.k.cx) / self.k.fx;
        [z_m, -xn * z_m, -self.yn[v] * z_m]
    }

    /// Renders one depth image (row-major u16 millimetres). `preview`: also fills a grayscale view.
    pub fn render(&self, world: &World, lens: LensPose, opts: &RenderOpts, rng: &mut Rng, mut preview: Option<&mut Vec<u8>>) -> Vec<u16> {
        let (w, h) = (self.k.width, self.k.height);
        let mut out = vec![0u16; w * h];
        if let Some(p) = preview.as_deref_mut() {
            p.clear();
            p.resize(w * h, 0);
        }
        // Bodies the camera can see in this frame (sparse ones by chance).
        let visible: Vec<usize> = world
            .bodies
            .iter()
            .enumerate()
            .filter(|(_, b)| {
                b.active
                    && match b.depth {
                        DepthVis::Solid => true,
                        DepthVis::Invisible => false,
                        DepthVis::Sparse(p) => rng.uniform() < p,
                    }
            })
            .map(|(k, _)| k)
            .collect();
        let max_s = opts.max_range / opts.scale.max(1e-3) * 1.5 + 0.5;
        let o: P2 = [lens.x, lens.y];
        let a_cos: Vec<f64> = self.azimuth.iter().map(|a| (lens.yaw + a).cos()).collect();
        let a_sin: Vec<f64> = self.azimuth.iter().map(|a| (lens.yaw + a).sin()).collect();
        let mut hits: Vec<(f64, f64, f64, f64, usize)> = Vec::new();
        for u in 0..w {
            let d = [a_cos[u], a_sin[u]];
            let s_wall = world.grid.ray(o, d, max_s);
            hits.clear();
            for &k in &visible {
                let b = &world.bodies[k];
                if let Some((s0, s1)) = b.shape.ray(o, d) {
                    if s0 < s_wall {
                        hits.push((s0, s1.min(s_wall), b.zmin, b.zmax, k));
                    }
                }
            }
            hits.sort_by(|x, y| x.0.total_cmp(&y.0));
            let inv = self.inv[u];
            for v in 0..h {
                // Height along the ray: z(s) = lens.z + slope * s (s = horizontal distance).
                let slope = -self.yn[v] * inv;
                let mut best = f64::INFINITY;
                let mut surf = Surface::Nothing;
                if slope < 0.0 {
                    best = lens.z / -slope;
                    surf = Surface::Floor;
                    // Over a drop-off (stairs going down) the floor is far lower: the camera sees
                    // down into it rather than an obstacle.
                    if !world.cliffs.is_empty() {
                        let d = [a_cos[u], a_sin[u]];
                        let p = [o[0] + best * d[0], o[1] + best * d[1]];
                        if world.cliffs.iter().any(|(_, s)| s.contains(p)) {
                            best = (lens.z + CLIFF_DROP_M) / -slope;
                        }
                    }
                } else if slope > 0.0 {
                    best = (WALL_HEIGHT_M - lens.z) / slope;
                    surf = Surface::Ceiling;
                }
                if s_wall < best && s_wall < max_s {
                    best = s_wall;
                    surf = Surface::Wall;
                }
                for &(s0, s1, zmin, zmax, k) in &hits {
                    if s0 >= best {
                        break;
                    }
                    // First s in [s0, s1] where zmin <= z(s) <= zmax.
                    let enter = if slope == 0.0 {
                        (zmin..=zmax).contains(&lens.z).then_some(s0)
                    } else {
                        let (a1, a2) = ((zmin - lens.z) / slope, (zmax - lens.z) / slope);
                        let lo = a1.min(a2).max(s0);
                        let hi = a1.max(a2).min(s1);
                        (lo <= hi).then_some(lo)
                    };
                    if let Some(s) = enter {
                        if s < best {
                            best = s;
                            surf = Surface::Body(world.bodies[k].kind as u8);
                        }
                        break;
                    }
                }
                if !best.is_finite() {
                    continue;
                }
                // Optical depth Z = s * inv, as the camera reports it.
                let mut z = best * inv * opts.scale;
                if opts.noise_a > 0.0 || opts.noise_b > 0.0 {
                    z += rng.gauss(opts.noise_a + opts.noise_b * z * z);
                }
                if opts.dropout > 0.0 && rng.uniform() < opts.dropout {
                    continue;
                }
                if z < opts.min_range || z > opts.max_range {
                    continue;
                }
                out[v * w + u] = (z * 1000.0).round() as u16;
                if let Some(p) = preview.as_deref_mut() {
                    p[v * w + u] = shade(surf, z, u, v);
                }
            }
        }
        out
    }
}

/// Grayscale for the preview view: floor darker with a faint texture, walls by distance, bodies flat.
fn shade(surf: Surface, z: f64, u: usize, v: usize) -> u8 {
    let fade = |base: f64| (base - 18.0 * z).clamp(25.0, 255.0) as u8;
    match surf {
        Surface::Nothing => 0,
        Surface::Floor => fade(120.0 + if (u / 16 + v / 16) % 2 == 0 { 10.0 } else { 0.0 }),
        Surface::Ceiling => fade(200.0),
        Surface::Wall => fade(230.0),
        Surface::Body(k) => 60 + 25 * k,
    }
}

// ---------------------------------------------------------------- IMU

/// Rotation of camera_imu_frame in base_link (camera level on the mount).
pub fn q_base_imu(cam_pitch: f64) -> Quat {
    let q_mount = [0.0, (cam_pitch / 2.0).sin(), 0.0, (cam_pitch / 2.0).cos()];
    quat_mul(quat_mul(q_mount, Q_LINK_OPTICAL), Q_OPTICAL_IMU)
}

/// One IMU sample in camera_imu_frame: (angular velocity rad/s, linear acceleration m/s^2) for a
/// robot turning at `w` with forward acceleration `accel` at speed `v` (base_link, planar).
pub fn imu_sample(q_bi: Quat, v: f64, w: f64, accel: f64, bias: Vec3, gyro_noise: f64, accel_noise: f64, rng: &mut Rng) -> (Vec3, Vec3) {
    let qi = quat_conj(q_bi);
    let gyro = quat_rotate(qi, [0.0, 0.0, w]);
    let acc = quat_rotate(qi, [accel, v * w, 9.80665]);
    (
        [gyro[0] + bias[0] + rng.gauss(gyro_noise), gyro[1] + bias[1] + rng.gauss(gyro_noise), gyro[2] + bias[2] + rng.gauss(gyro_noise)],
        [acc[0] + rng.gauss(accel_noise), acc[1] + rng.gauss(accel_noise), acc[2] + rng.gauss(accel_noise)],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::twin::world::{Body, DepthVis, Grid, Kind, Shape};

    /// A 10 m x 10 m room, the lens 3 m from the +x wall.
    fn room() -> World {
        let (n, res) = (200, 0.05);
        let mut g = Grid::new(n, n, res, [-5.0, -5.0]);
        for k in 0..n {
            for (i, j) in [(k, 0), (k, n - 1), (0, k), (n - 1, k)] {
                g.solid[j * n + i] = true;
            }
        }
        // wall cells start at x = 4.95
        World::new(g)
    }

    fn lens() -> LensPose {
        LensPose { x: 1.95, y: 0.0, z: 0.13, yaw: 0.0 }
    }

    fn at(img: &[u16], u: usize, v: usize) -> u16 {
        img[v * D1000.width + u]
    }

    #[test]
    fn wall_floor_and_range_limits() {
        let cam = Camera::new(D1000);
        let world = room();
        let mut rng = Rng::new(1);
        let img = cam.render(&world, lens(), &RenderOpts::default(), &mut rng, None);
        // Centre pixel (on the optical axis, ~level): the wall 3 m ahead.
        let (uc, vc) = (318, 174);
        assert!((at(&img, uc, vc) as i32 - 3000).abs() <= 2, "{}", at(&img, uc, vc));
        // Lower rows: the floor, Z = 0.13 / y_n (the bottom ~35 rows see it closer than 0.3 m: 0).
        let yn = (320.0 - D1000.cy) / D1000.fy;
        assert!((at(&img, uc, 320) as f64 - 130.0 / yn).abs() <= 1.0, "{}", at(&img, uc, 320));
        assert_eq!(at(&img, uc, 359), 0);
        // A wall 0.2 m away reads 0 everywhere.
        let near = LensPose { x: 4.75, ..lens() };
        let img = cam.render(&world, near, &RenderOpts::default(), &mut rng, None);
        assert!(img.iter().all(|&d| d == 0));
        // Scale: the real camera's 15 % short depth.
        let opts = RenderOpts { scale: 0.85, ..Default::default() };
        let img = cam.render(&world, lens(), &opts, &mut rng, None);
        assert!((at(&img, uc, vc) as i32 - 2550).abs() <= 2);
    }

    #[test]
    fn low_obstacles_overhangs_people_and_invisible_railings() {
        let cam = Camera::new(D1000);
        let mut world = room();
        let body = |kind, shape, zmin, zmax, depth| Body { name: "b".into(), kind, shape, zmin, zmax, depth, ir: true, blocks: true, active: true };
        // A 5 cm high box 1 m ahead: the bottom rows see it, the centre row passes over it.
        world.bodies.push(body(Kind::Obstacle, Shape::Polygon(vec![[2.95, -0.3], [3.25, -0.3], [3.25, 0.3], [2.95, 0.3]]), 0.0, 0.05, DepthVis::Solid));
        let mut rng = Rng::new(2);
        let img = cam.render(&world, lens(), &RenderOpts::default(), &mut rng, None);
        let (uc, vc) = (318, 174);
        assert!((at(&img, uc, vc) as i32 - 3000).abs() <= 2);
        // Row whose ray reaches z = 0.03 at s = 1.0: y_n = 0.1 -> v = cy + 0.1 fy.
        let v = (D1000.cy + 0.1 * D1000.fy).round() as usize;
        assert!((at(&img, uc, v) as i32 - 1000).abs() <= 5, "{}", at(&img, uc, v));
        // An invisible railing in front of it changes nothing; a solid person hides everything.
        world.bodies.push(body(Kind::Railing, Shape::Capsule { a: [2.5, -1.0], b: [2.5, 1.0], r: 0.02 }, 0.0, 0.9, DepthVis::Invisible));
        let img2 = cam.render(&world, lens(), &RenderOpts::default(), &mut rng, None);
        assert_eq!(img, img2);
        world.bodies.push(body(Kind::Person, Shape::Circle { c: [2.55, 0.0], r: 0.2 }, 0.0, 1.7, DepthVis::Solid));
        let img3 = cam.render(&world, lens(), &RenderOpts::default(), &mut rng, None);
        // 0.6 m to the person's centre, minus the radius: 0.4 m.
        assert!((at(&img3, uc, vc) as i32 - 400).abs() <= 3, "{}", at(&img3, uc, vc));
        // A table top (0.6-0.75 m) 1 m ahead: rays from the 0.13 m lens going up hit its underside.
        world.bodies.truncate(1);
        world.bodies.push(body(Kind::Obstacle, Shape::Polygon(vec![[2.95, -0.5], [3.95, -0.5], [3.95, 0.5], [2.95, 0.5]]), 0.6, 0.75, DepthVis::Solid));
        let img4 = cam.render(&world, lens(), &RenderOpts::default(), &mut rng, None);
        // Top row (slope ~0.5): at 1 m it is 0.63 m up, inside the top: its front face.
        assert!((at(&img4, uc, 0) as i32 - 1000).abs() <= 10, "{}", at(&img4, uc, 0));
        // Slope ~0.3: still under the top at its front edge, it hits the underside (z = 0.6) at ~1.57 m.
        assert!((at(&img4, uc, 70) as i32 - 1570).abs() <= 10, "{}", at(&img4, uc, 70));
    }

    #[test]
    fn the_camera_looks_down_into_stairs() {
        let cam = Camera::new(D1000);
        let mut world = room();
        let mut rng = Rng::new(4);
        let (uc, v) = (318, 250); // this row meets the floor ~0.6 m ahead
        let img = cam.render(&world, lens(), &RenderOpts::default(), &mut rng, None);
        let floor = 130.0 / ((v as f64 - D1000.cy) / D1000.fy);
        assert!((at(&img, uc, v) as f64 - floor).abs() <= 1.0);
        // Stairs down from 0.4 to 1 m ahead: that ray now goes past the edge, down to the far wall.
        world.cliffs.push(("stairs".into(), Shape::Polygon(vec![[2.35, -1.0], [2.95, -1.0], [2.95, 1.0], [2.35, 1.0]])));
        let img = cam.render(&world, lens(), &RenderOpts::default(), &mut rng, None);
        assert!((at(&img, uc, v) as i32 - 3000).abs() <= 2, "{}", at(&img, uc, v));
    }

    #[test]
    fn imu_reads_gravity_up_and_yaw_rate_about_the_vertical() {
        let q = q_base_imu(0.0);
        let mut rng = Rng::new(3);
        let (g, a) = imu_sample(q, 0.0, 0.5, 0.0, [0.0; 3], 0.0, 0.0, &mut rng);
        // In camera_imu_frame "up" is about +y (the recording read +9.76 on y at rest).
        assert!((a[1] - 9.80665).abs() < 0.01 && a[0].abs() < 0.1 && a[2].abs() < 0.1, "{a:?}");
        assert!((g[1] - 0.5).abs() < 0.01, "{g:?}");
    }
}
