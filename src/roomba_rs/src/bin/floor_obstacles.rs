//! floor_obstacles (Rust): splits the depth camera's view into floor and obstacles for Nav2.
//! Drop-in replacement for roomba_perception/floor_obstacles.py: same topics, parameters,
//! status JSON and behaviour (see that file's docstring), at a fraction of the CPU.
//!
//! Subscribes: /camera/depth/image_rect_raw, /camera/depth/camera_info, /camera/imu/data, /tf_static
//! Publishes:  /perception/obstacles, /perception/obstacles_mapping, /perception/floor_points
//!             (PointCloud2, depth optical frame), /perception/floor_status (String JSON)

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use rclrs::*;
use ros_env::{geometry_msgs, sensor_msgs, std_msgs, tf2_msgs};

use roomba_rs::floor::{
    blind_points, cloud_bytes, dense_mask, depth_points, drop_points, estimate_floor, floor_axes, DepthImage, Intrinsics,
};
use roomba_rs::geometry::{dot, normalized, quat_conj, quat_rotate, Quat, Vec3};
use roomba_rs::static_tf::StaticTree;

struct Params {
    decimation: MandatoryParameter<i64>,
    rate_hz: MandatoryParameter<f64>,
    min_range: MandatoryParameter<f64>,
    max_range: MandatoryParameter<f64>,
    min_h: MandatoryParameter<f64>,
    max_h: MandatoryParameter<f64>,
    floor_band: MandatoryParameter<f64>,
    fallback_height: MandatoryParameter<f64>,
    fixed_height: MandatoryParameter<f64>,
    floor_memory: MandatoryParameter<f64>,
    imu_frame: MandatoryParameter<Arc<str>>,
    pause_turn_rate: MandatoryParameter<f64>,
    min_cell_points: MandatoryParameter<i64>,
    cell_size: MandatoryParameter<f64>,
    blind_distance: MandatoryParameter<f64>,
    drop_depth: MandatoryParameter<f64>,
    drop_max_range: MandatoryParameter<f64>,
    drop_lift: MandatoryParameter<f64>,
    drop_min_points: MandatoryParameter<i64>,
}

#[derive(Default)]
struct State {
    info: Option<Intrinsics>,
    imu_q: Option<Quat>,
    r_opt_imu: Option<Quat>,
    tree: StaticTree,
    floor_h: Option<f64>,
    floor_seen: Option<Instant>,
    last_proc: Option<Instant>,
    last_fast_turn: Option<Instant>,
    last_log: Option<Instant>,
    speckles: usize,
    paused_frames: u64,
    blind_frames: u64,
}

struct Outputs {
    obs: Publisher<sensor_msgs::msg::PointCloud2>,
    obs_map: Publisher<sensor_msgs::msg::PointCloud2>,
    floor: Publisher<sensor_msgs::msg::PointCloud2>,
    status: Publisher<std_msgs::msg::String>,
}

fn cloud(header: &std_msgs::msg::Header, pts: &[Vec3], every: usize) -> sensor_msgs::msg::PointCloud2 {
    let data = cloud_bytes(pts.iter().step_by(every.max(1)));
    let n = (data.len() / 12) as u32;
    let field = |name: &str, offset: u32| sensor_msgs::msg::PointField {
        name: name.into(),
        offset,
        datatype: sensor_msgs::msg::PointField::FLOAT32,
        count: 1,
    };
    sensor_msgs::msg::PointCloud2 {
        header: header.clone(),
        height: 1,
        width: n,
        fields: vec![field("x", 0), field("y", 4), field("z", 8)],
        is_bigendian: false,
        point_step: 12,
        row_step: 12 * n,
        data,
        is_dense: true,
    }
}

/// Python-style float for the status JSON (keeps a ".0" on whole numbers).
fn jf(x: f64) -> String {
    if x.is_finite() && x.fract() == 0.0 { format!("{x:.1}") } else { format!("{x}") }
}

fn round_to(x: f64, places: i32) -> f64 {
    let s = 10f64.powi(places);
    (x * s).round() / s
}

impl State {
    /// World "up" in the depth optical frame, or None until the IMU and static TF are known.
    fn up_in_optical(&mut self, frame: &str, imu_frame: &str) -> Option<Vec3> {
        let q = self.imu_q?;
        if self.r_opt_imu.is_none() {
            self.r_opt_imu = self.tree.rotation(frame, imu_frame);
        }
        let up_imu = quat_rotate(quat_conj(q), [0.0, 0.0, 1.0]);
        Some(normalized(quat_rotate(self.r_opt_imu?, up_imu)))
    }

    fn on_depth(&mut self, msg: sensor_msgs::msg::Image, p: &Params, out: &Outputs, node: &Node) {
        let now = Instant::now();
        let Some(k) = self.info else { return };
        if let Some(last) = self.last_proc {
            if now.duration_since(last).as_secs_f64() < 1.0 / p.rate_hz.get() {
                return;
            }
        }
        self.last_proc = Some(now);
        let Some(up) = self.up_in_optical(&msg.header.frame_id, &p.imu_frame.get()) else { return };
        if msg.encoding != "16UC1" && msg.encoding != "mono16" {
            log!(node.warn().throttle(Duration::from_secs(10)), "unsupported depth encoding {}", msg.encoding);
            return;
        }

        let img = DepthImage {
            data: &msg.data,
            width: msg.width as usize,
            height: msg.height as usize,
            step: msg.step as usize,
            big_endian: msg.is_bigendian != 0,
        };
        let min_range = p.min_range.get();
        let pts = depth_points(&img, p.decimation.get().max(1) as usize, k, min_range, p.max_range.get());
        if pts.len() < 100 {
            self.blind_frames += 1;
            out.obs.publish(cloud(&msg.header, &blind_points(up, p.blind_distance.get()), 1)).ok();
            let status = format!("{{\"blind\": true, \"blind_frames\": {}}}", self.blind_frames);
            out.status.publish(std_msgs::msg::String { data: status }).ok();
            log!(
                node.warn().throttle(Duration::from_secs(5)),
                "depth blind (closer than {min_range:.2} m?): blocking forward motion only"
            );
            return;
        }
        let h: Vec<f64> = pts.iter().map(|&q| dot(q, up)).collect();

        let est = estimate_floor(&h);
        let fixed = p.fixed_height.get();
        if fixed > 0.0 {
            self.floor_h = Some(-fixed);
        } else if let Some(e) = est {
            self.floor_h = Some(match self.floor_h {
                Some(f) => 0.7 * f + 0.3 * e,
                None => e,
            });
            self.floor_seen = Some(now);
        } else if self.floor_h.is_none()
            || self.floor_seen.map_or(true, |t| now.duration_since(t).as_secs_f64() > p.floor_memory.get())
        {
            self.floor_h = Some(-p.fallback_height.get());
        }
        let floor_h = self.floor_h.unwrap_or(-p.fallback_height.get());

        let (min_h, max_h, band) = (p.min_h.get(), p.max_h.get(), p.floor_band.get());
        let mut obstacles = Vec::new();
        let mut floor = Vec::new();
        for (q, hh) in pts.iter().zip(&h) {
            let rel = hh - floor_h;
            if rel > min_h && rel < max_h {
                obstacles.push(*q);
            }
            if rel.abs() < band {
                floor.push(*q);
            }
        }
        let min_cells = p.min_cell_points.get();
        if min_cells > 1 && !obstacles.is_empty() {
            let keep = dense_mask(&obstacles, up, p.cell_size.get(), min_cells as usize);
            self.speckles = keep.iter().filter(|&&k| !k).count();
            obstacles = obstacles.into_iter().zip(keep).filter_map(|(q, k)| k.then_some(q)).collect();
        }
        let mut drops = 0;
        if p.drop_depth.get() > 0.0 {
            let (fwd, side) = floor_axes(up);
            let d = drop_points(
                &pts,
                &h,
                floor_h,
                up,
                p.drop_depth.get(),
                p.drop_max_range.get(),
                p.drop_lift.get(),
                fwd,
                side,
                p.cell_size.get(),
                p.drop_min_points.get().max(1) as usize,
            );
            drops = d.len();
            obstacles.extend(d);
        }

        let obs_cloud = cloud(&msg.header, &obstacles, 1);
        // Depth frames arrive ~80 ms late, so the costmap copies also pause just after a turn.
        if self.last_fast_turn.map_or(false, |t| now.duration_since(t).as_secs_f64() < 0.3) {
            self.paused_frames += 1;
            out.obs.publish(obs_cloud).ok();
        } else {
            out.obs.publish(obs_cloud.clone()).ok();
            out.obs_map.publish(obs_cloud).ok();
            out.floor.publish(cloud(&msg.header, &floor, 2)).ok();
        }

        let tilt = (-up[1]).clamp(-1.0, 1.0).acos().to_degrees();
        let status = format!(
            "{{\"camera_height_m\": {}, \"measured_height_m\": {}, \"fixed_height\": {}, \"floor_visible\": {}, \
             \"tilt_deg\": {}, \"obstacle_points\": {}, \"speckles_removed\": {}, \"floor_points\": {}, \
             \"paused_frames\": {}, \"blind_frames\": {}, \"drop_points\": {}}}",
            jf(round_to(-floor_h, 2)),
            est.map_or("null".to_string(), |e| jf(round_to(-e, 3))),
            fixed > 0.0,
            est.is_some(),
            jf(round_to(tilt, 1)),
            obstacles.len(),
            self.speckles,
            floor.len(),
            self.paused_frames,
            self.blind_frames,
            drops,
        );
        if self.last_log.map_or(true, |t| now.duration_since(t).as_secs_f64() > 5.0) {
            self.last_log = Some(now);
            log!(node.info(), "floor: {status}");
        }
        out.status.publish(std_msgs::msg::String { data: status }).ok();
    }
}

fn main() -> Result<()> {
    roomba_rs::default_signals();
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();
    let node = executor.create_node("floor_obstacles")?;

    let f = |name: &str, default: f64| node.declare_parameter(name).default(default).mandatory();
    let i = |name: &str, default: i64| node.declare_parameter(name).default(default).mandatory();
    let params = Arc::new(Params {
        decimation: i("decimation", 4)?,
        rate_hz: f("rate_hz", 6.0)?,
        min_range: f("min_range_m", 0.3)?,
        max_range: f("max_range_m", 4.0)?,
        min_h: f("min_obstacle_height_m", 0.06)?,
        max_h: f("max_obstacle_height_m", 1.0)?,
        floor_band: f("floor_band_m", 0.03)?,
        fallback_height: f("fallback_camera_height_m", 0.4)?,
        fixed_height: f("fixed_camera_height_m", 0.0)?,
        floor_memory: f("floor_memory_s", 30.0)?,
        imu_frame: node.declare_parameter("imu_frame").default("camera_imu_frame".into()).mandatory()?,
        pause_turn_rate: f("pause_turn_rate_rps", 0.0)?,
        min_cell_points: i("min_cell_points", 1)?,
        cell_size: f("cell_size_m", 0.05)?,
        blind_distance: f("blind_obstacle_distance_m", 0.15)?,
        // Drop-offs (negative obstacles): points this far below the floor plane (0 = off), within
        // drop_max_range_m, put obstacle points drop_point_height_m above the floor where their line
        // of sight crosses the floor plane, where at least drop_min_points of them gather.
        drop_depth: f("drop_min_depth_m", 0.0)?,
        drop_max_range: f("drop_max_range_m", 2.0)?,
        drop_lift: f("drop_point_height_m", 0.1)?,
        drop_min_points: i("drop_min_points", 6)?,
    });
    let out = Outputs {
        obs: node.create_publisher("/perception/obstacles".keep_last(2))?,
        obs_map: node.create_publisher("/perception/obstacles_mapping".keep_last(2))?,
        floor: node.create_publisher("/perception/floor_points".keep_last(2))?,
        status: node.create_publisher("/perception/floor_status".keep_last(2))?,
    };

    let worker = node.create_worker::<State>(State::default());
    let _info = worker.create_subscription::<sensor_msgs::msg::CameraInfo, _>(
        "/camera/depth/camera_info".keep_last(2),
        |s: &mut State, m: sensor_msgs::msg::CameraInfo| {
            s.info = Some(Intrinsics { fx: m.p[0], fy: m.p[5], cx: m.p[2], cy: m.p[6] });
        },
    )?;
    let imu_params = Arc::clone(&params);
    let _imu = worker.create_subscription::<sensor_msgs::msg::Imu, _>(
        "/camera/imu/data".keep_last(20),
        move |s: &mut State, m: sensor_msgs::msg::Imu| {
            let o = m.orientation;
            s.imu_q = Some([o.x, o.y, o.z, o.w]);
            let w = m.angular_velocity;
            let rate = imu_params.pause_turn_rate.get();
            if rate > 0.0 && (w.x * w.x + w.y * w.y + w.z * w.z).sqrt() > rate {
                s.last_fast_turn = Some(Instant::now());
            }
        },
    )?;
    let _tf = worker.create_subscription::<tf2_msgs::msg::TFMessage, _>(
        "/tf_static".qos(QoSProfile::topics_default().keep_last(100).reliable().transient_local()),
        |s: &mut State, m: tf2_msgs::msg::TFMessage| {
            for t in m.transforms {
                let r: geometry_msgs::msg::Quaternion = t.transform.rotation;
                s.tree.insert(&t.header.frame_id, &t.child_frame_id, [r.x, r.y, r.z, r.w]);
            }
            s.r_opt_imu = None; // recompute with the new transforms
        },
    )?;
    let depth_node = node.clone();
    let _depth = worker.create_subscription::<sensor_msgs::msg::Image, _>(
        "/camera/depth/image_rect_raw".keep_last(2),
        move |s: &mut State, m: sensor_msgs::msg::Image| s.on_depth(m, &params, &out, &depth_node),
    )?;

    log!(node.info(), "floor_obstacles (Rust) running");
    roomba_rs::node_params::spin(&mut executor)?;
    Ok(())
}