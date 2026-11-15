//! ekf_inputs (Rust): prepares the wheel odometry and the camera gyro for robot_localization.
//! Drop-in replacement for roomba_perception/ekf_inputs.py: same topics, parameters, noise model
//! and status JSON (see that file's docstring for the details).
//!
//! /wheel/odom -> /ekf/wheel_odom, /camera/imu/data_raw -> /ekf/imu, /ekf/inputs_status (1 Hz)

use std::sync::Arc;
use std::time::{Duration, Instant};

/// Monotonic seconds since the node started (EkfCore takes the clock as a number).
fn mono(start: Instant) -> f64 {
    start.elapsed().as_secs_f64()
}

use anyhow::Result;
use rclrs::*;
use ros_env::{nav_msgs, sensor_msgs, std_msgs};

use roomba_rs::ekf::{npz_gyro_mean, parse_imu_noise, unknown_pose_cov, BiasEstimator, CalValue, EkfCore, GyroModel, WheelParams, V3};

fn stamp(h: &std_msgs::msg::Header) -> f64 {
    h.stamp.sec as f64 + h.stamp.nanosec as f64 * 1e-9
}

fn expand(path: &str) -> std::path::PathBuf {
    match path.strip_prefix("~/") {
        Some(rest) => std::path::Path::new(&std::env::var("HOME").unwrap_or_default()).join(rest),
        None => path.into(),
    }
}

struct State {
    core: EkfCore,
    start: Instant,
    pub_wheel: Publisher<nav_msgs::msg::Odometry>,
    pub_imu: Publisher<sensor_msgs::msg::Imu>,
    pub_status: Publisher<std_msgs::msg::String>,
}
fn jf(x: f64) -> String {
    if x.is_finite() && x.fract() == 0.0 { format!("{x:.1}") } else { format!("{x}") }
}

fn round_to(x: f64, places: i32) -> f64 {
    let s = 10f64.powi(places);
    (x * s).round() / s
}

impl State {
    fn on_wheel(&mut self, msg: nav_msgs::msg::Odometry) {
        let tw = &msg.twist.twist;
        let (vx, wz, cov) = self.core.on_wheel(stamp(&msg.header), tw.linear.x, tw.angular.z, mono(self.start));
        let mut out = nav_msgs::msg::Odometry::default();
        out.child_frame_id = if msg.child_frame_id.is_empty() { "base_link".into() } else { msg.child_frame_id };
        out.header = msg.header;
        out.twist.twist.linear.x = vx;
        out.twist.twist.angular.z = wz;
        out.twist.covariance = cov;
        // Pose is not fused (integrated wheel pose drifts without bound); keep it marked unknown.
        out.pose.covariance = unknown_pose_cov();
        self.pub_wheel.publish(out).ok();
    }

    fn on_imu(&mut self, msg: sensor_msgs::msg::Imu) {
        let w = &msg.angular_velocity;
        let Some((mean, sd)) = self.core.on_imu(stamp(&msg.header), [w.x, w.y, w.z], mono(self.start)) else { return };
        let v = sd * sd;
        let mut out = sensor_msgs::msg::Imu::default();
        out.header = msg.header;
        out.orientation_covariance[0] = -1.0;
        out.linear_acceleration_covariance[0] = -1.0;
        out.angular_velocity.x = mean[0];
        out.angular_velocity.y = mean[1];
        out.angular_velocity.z = mean[2];
        out.angular_velocity_covariance = [v, 0.0, 0.0, 0.0, v, 0.0, 0.0, 0.0, v];
        self.pub_imu.publish(out).ok();
    }
    fn status(&self) {
        let b = self.core.bias.bias;
        let age = self.core.wheel_age(mono(self.start));
        let data = format!(
            "{{\"gyro_bias_dps\": [{}, {}, {}], \"bias_ready\": {}, \"bias_samples\": {}, \"gyro_sd_rad_s\": {}, \
             \"calibrated\": {}, \"wheel_age_s\": {}}}",
            jf(round_to(b[0].to_degrees(), 4)),
            jf(round_to(b[1].to_degrees(), 4)),
            jf(round_to(b[2].to_degrees(), 4)),
            self.core.bias.ready,
            self.core.bias.samples,
            self.core.last_sd.map_or("null".into(), |s| jf(round_to(s, 6))),
            self.core.model.noise_n.is_some(),
            age.map_or("null".into(), |a| jf(round_to(a, 2))),
        );
        self.pub_status.publish(std_msgs::msg::String { data }).ok();
    }
}

fn main() -> Result<()> {
    roomba_rs::default_signals();
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();
    let node = executor.create_node("ekf_inputs")?;
    // rclrs undeclares a parameter when its handle is dropped; keep them so `ros2 param` sees them.
    let handles = std::cell::RefCell::new(Vec::<Box<dyn std::any::Any>>::new());
    let keep = |p: Box<dyn std::any::Any>| handles.borrow_mut().push(p);
    let f = |name: &str, d: f64| -> Result<f64> {
        let p = node.declare_parameter(name).default(d).mandatory()?;
        let v = p.get();
        keep(Box::new(p));
        Ok(v)
    };
    let s = |name: &str, d: &str| -> Result<String> {
        let p = node.declare_parameter::<Arc<str>>(name).default(d.into()).mandatory()?;
        let v = p.get().to_string();
        keep(Box::new(p));
        Ok(v)
    };

    let wheel = WheelParams {
        lin_scale: f("wheel_linear_scale", 0.952)?,
        ang_scale: f("wheel_angular_scale", 0.71)?,
        vx_sd0: f("wheel_vx_sd", 0.01)?,
        vx_sd_rel: f("wheel_vx_sd_ratio", 0.1)?,
        vx_sd_rel_rev: f("wheel_vx_sd_ratio_reverse", 0.5)?,
        vy_sd: f("wheel_vy_sd", 0.01)?,
        wz_sd0: f("wheel_wz_sd", 0.01)?,
        wz_sd_rel: f("wheel_wz_sd_ratio", 0.4)?,
    };
    let fallback_sd = f("gyro_sd", 0.03)?;
    let motion_ratio = f("gyro_motion_sd_ratio", 0.02)?;
    let turn_on_sd = f("bias_turn_on_sd", 0.001)?;
    let average_p = node.declare_parameter("imu_average").default(4i64).mandatory()?;
    let average = average_p.get().max(1) as usize;
    keep(Box::new(average_p));
    let still_s = f("still_time_s", 1.0)?;
    let bias_tau = f("bias_time_constant_s", 5.0)?;
    let wheel_max_age = f("wheel_max_age_s", 0.5)?;
    let noise_file = s("imu_noise_file", "~/.config/roomba/imu_noise.yaml")?;
    let raw_glob = s("imu_raw_glob", "~/maps/imu_raw_*.npz")?;
    let quiet_param = f("quiet_gyro_rate", 0.0)?;
    let wheel_topic = s("wheel_topic", "/wheel/odom")?;
    let imu_topic = s("imu_topic", "/camera/imu/data_raw")?;

    // Calibration (same rules as the Python node).
    let mut model = GyroModel { noise_n: None, noise_k: 0.0, imu_rate: 200.0, average, motion_ratio, turn_on_sd, bias_tau, fallback_sd };
    let mut bias0: V3 = [0.0; 3];
    let mut bias_from_file = false;
    match std::fs::read_to_string(expand(&noise_file)) {
        Ok(text) => {
            let (values, inflation) = parse_imu_noise(&text);
            if let Some(CalValue::Num(n)) = values.get("gyro_noise_density") {
                let infl = inflation.unwrap_or_else(|| {
                    log!(node.warn(), "{noise_file}: no \"raw values xN\" in the header; assuming raw values");
                    1.0
                });
                let num = |k: &str, d: f64| if let Some(CalValue::Num(v)) = values.get(k) { *v } else { d };
                model.noise_n = Some(n / infl);
                model.noise_k = num("gyro_random_walk", 0.0) / infl;
                model.imu_rate = num("calibration_frequency", 200.0);
                let f_out = model.imu_rate / average as f64;
                log!(node.info(), "Gyro noise from {noise_file} (stored x{infl}, divided back): N = {:.4e} rad/s/sqrt(Hz), \
                     K = {:.4e} rad/s^2/sqrt(Hz), {} Hz -> white sd {:.2e} rad/s at {f_out:.1} Hz; motion allowance {:.0}% of rate, \
                     turn-on bias sd {turn_on_sd} rad/s", n / infl, model.noise_k, model.imu_rate, n / infl * f_out.sqrt(), motion_ratio * 100.0);
            }
            if let Some(CalValue::List(b)) = values.get("gyro_static_bias") {
                if b.len() >= 3 {
                    bias0 = [b[0], b[1], b[2]];
                    bias_from_file = true;
                    log!(node.info(), "Initial gyro bias from {noise_file}: {:.4?} deg/s", bias0.map(f64::to_degrees));
                }
            }
        }
        Err(_) => log!(node.warn(), "{noise_file} not found: fixed gyro sd {fallback_sd} rad/s (uncalibrated default)"),
    }
    if model.noise_n.is_none() {
        log!(node.warn(), "{noise_file}: no gyro_noise_density: fixed gyro sd {fallback_sd} rad/s (uncalibrated default)");
    }
    if !bias_from_file {
        let pattern = expand(&raw_glob);
        let (dir, name) = (pattern.parent().map(|p| p.to_path_buf()).unwrap_or_default(), pattern.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default());
        let (pre, post) = name.split_once('*').unwrap_or((name.as_str(), ""));
        let mut raws: Vec<_> = std::fs::read_dir(&dir).into_iter().flatten().flatten()
            .map(|e| e.path())
            .filter(|p| p.file_name().is_some_and(|n| { let n = n.to_string_lossy(); n.starts_with(pre) && n.ends_with(post) }))
            .collect();
        raws.sort();
        match raws.last() {
            Some(raw) => match npz_gyro_mean(raw) {
                Ok((mean, hours)) => {
                    bias0 = mean;
                    log!(node.info(), "Initial gyro bias = mean of the {hours:.1} h static calibration recording {}: {:.4?} deg/s \
                         (relearned at standstill)", raw.display(), mean.map(f64::to_degrees));
                }
                Err(e) => log!(node.warn(), "Could not read {}: {e}", raw.display()),
            },
            None => log!(node.info(), "No static gyro bias available: starting from zero (learned at the first standstill)"),
        }
    }
    let quiet_rate = if quiet_param > 0.0 {
        quiet_param
    } else {
        model.noise_n.map_or(0.02, |n| (6.0 * n * model.imu_rate.sqrt()).max(0.005))
    };

    let bias = BiasEstimator::new(bias0, still_s, quiet_rate, model.imu_rate, bias_tau);
    let state = State {
        core: EkfCore::new(wheel, model, bias, wheel_max_age),
        start: Instant::now(),
        pub_wheel: node.create_publisher("/ekf/wheel_odom".keep_last(20))?,
        pub_imu: node.create_publisher("/ekf/imu".keep_last(50))?,
        pub_status: node.create_publisher("/ekf/inputs_status".keep_last(2))?,
    };
    let worker = node.create_worker::<State>(state);
    let _wheel = worker.create_subscription::<nav_msgs::msg::Odometry, _>(
        wheel_topic.as_str().keep_last(20),
        |s: &mut State, m: nav_msgs::msg::Odometry| s.on_wheel(m),
    )?;
    let _imu = worker.create_subscription::<sensor_msgs::msg::Imu, _>(
        imu_topic.as_str().sensor_data_qos(),
        |s: &mut State, m: sensor_msgs::msg::Imu| s.on_imu(m),
    )?;
    let _timer = worker.create_timer_repeating(Duration::from_secs(1), |s: &mut State| s.status())?;

    log!(node.info(), "ekf_inputs (Rust) running");
    roomba_rs::node_params::spin(&mut executor)?;
    Ok(())
}