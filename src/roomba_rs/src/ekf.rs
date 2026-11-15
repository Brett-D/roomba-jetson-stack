//! Wheel-odometry and gyro conditioning for robot_localization, ported from
//! roomba_perception/ekf_inputs.py (see that file for the noise model). Pure logic only.

use std::collections::{HashMap, VecDeque};
use std::io::Read;

pub type V3 = [f64; 3];

#[derive(Clone, Debug, PartialEq)]
pub enum CalValue {
    Num(f64),
    List(Vec<f64>),
}

/// (values, inflation) from imu_calibrate's imu_noise.yaml text; inflation is the N in the
/// header comment "raw values xN".
pub fn parse_imu_noise(text: &str) -> (HashMap<String, CalValue>, Option<f64>) {
    let mut values = HashMap::new();
    let mut inflation = None;
    for line in text.lines() {
        if line.trim_start().starts_with('#') {
            if let Some(i) = line.find("raw values x") {
                let num: String = line[i + 12..].chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
                inflation = num.parse().ok();
            }
            continue;
        }
        let Some((key, value)) = line.split_once(':') else { continue };
        let value = value.trim();
        let parsed = if value.starts_with('[') {
            value.trim_matches(|c| c == '[' || c == ']').split(',').map(|v| v.trim().parse::<f64>()).collect::<Result<Vec<_>, _>>().ok().map(CalValue::List)
        } else {
            value.parse::<f64>().ok().map(CalValue::Num)
        };
        if let Some(v) = parsed {
            values.insert(key.trim().to_string(), v);
        }
    }
    (values, inflation)
}

/// Reads a little-endian float64, C-order array (shape, data) from .npy bytes.
pub fn parse_npy_f64(bytes: &[u8]) -> Result<(Vec<usize>, Vec<f64>), String> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("not a .npy file".into());
    }
    let (hlen, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        _ => (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12),
    };
    let header = std::str::from_utf8(&bytes[start..start + hlen]).map_err(|e| e.to_string())?;
    if !header.contains("'<f8'") || header.contains("'fortran_order': True") {
        return Err(format!("unsupported array: {header}"));
    }
    let shape_txt = header.split("'shape':").nth(1).and_then(|s| s.split(')').next()).ok_or("no shape")?;
    let shape: Vec<usize> = shape_txt.trim().trim_start_matches('(').split(',').filter_map(|s| s.trim().parse().ok()).collect();
    let data: Vec<f64> = bytes[start + hlen..].chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    if data.len() != shape.iter().product::<usize>() {
        return Err("size mismatch".into());
    }
    Ok((shape, data))
}

/// (mean gyro, hours) of imu_calibrate's static recording (npz with 't' and 'gyro' arrays).
pub fn npz_gyro_mean(path: &std::path::Path) -> Result<(V3, f64), String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut zip = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;
    let mut read = |name: &str| -> Result<(Vec<usize>, Vec<f64>), String> {
        let mut buf = Vec::new();
        zip.by_name(name).map_err(|e| e.to_string())?.read_to_end(&mut buf).map_err(|e| e.to_string())?;
        parse_npy_f64(&buf)
    };
    let (shape, gyro) = read("gyro.npy")?;
    if shape.len() != 2 || shape[1] != 3 || shape[0] == 0 {
        return Err(format!("gyro shape {shape:?}"));
    }
    let mut sum = [0.0f64; 3];
    for row in gyro.chunks_exact(3) {
        for i in 0..3 {
            sum[i] += row[i];
        }
    }
    let n = shape[0] as f64;
    let (_, t) = read("t.npy")?;
    let hours = (t.last().copied().unwrap_or(0.0) - t.first().copied().unwrap_or(0.0)) / 3600.0;
    Ok(([sum[0] / n, sum[1] / n, sum[2] / n], hours))
}

#[derive(Clone, Debug)]
pub struct WheelParams {
    pub lin_scale: f64,
    pub ang_scale: f64,
    pub vx_sd0: f64,
    pub vx_sd_rel: f64,
    pub vx_sd_rel_rev: f64,
    pub vy_sd: f64,
    pub wz_sd0: f64,
    pub wz_sd_rel: f64,
}

/// Calibrated (vx, wz) and the 6x6 twist covariance for one wheel odometry message.
pub fn wheel_twist(p: &WheelParams, vx_in: f64, wz_in: f64) -> (f64, f64, [f64; 36]) {
    let vx = vx_in * p.lin_scale;
    let wz = wz_in * p.ang_scale;
    let rel = if vx < 0.0 { p.vx_sd_rel_rev } else { p.vx_sd_rel };
    let mut cov = [0.0; 36];
    cov[0] = p.vx_sd0.powi(2) + (rel * vx).powi(2);
    cov[7] = p.vy_sd.powi(2);
    cov[14] = 1e3;
    cov[21] = 1e3;
    cov[28] = 1e3; // z, roll, pitch rates: unused (two_d_mode)
    cov[35] = p.wz_sd0.powi(2) + (p.wz_sd_rel * wz).powi(2);
    (vx, wz, cov)
}

/// Pose covariance marking the (unfused) wheel pose as unknown.
pub fn unknown_pose_cov() -> [f64; 36] {
    let mut c = [0.0; 36];
    for i in (0..36).step_by(7) {
        c[i] = 1e3;
    }
    c
}

#[derive(Clone, Debug)]
pub struct GyroModel {
    pub noise_n: Option<f64>,
    pub noise_k: f64,
    pub imu_rate: f64,
    pub average: usize,
    pub motion_ratio: f64,
    pub turn_on_sd: f64,
    pub bias_tau: f64,
    pub fallback_sd: f64,
}

impl GyroModel {
    /// sd (rad/s) of one averaged gyro sample with magnitude `rate` at stamp t.
    pub fn sd(&self, bias_ready: bool, bias_stamp: Option<f64>, t: f64, rate: f64) -> f64 {
        let Some(n) = self.noise_n else {
            return self.fallback_sd * if bias_ready { 1.0 } else { 10.0 };
        };
        let white = n * (self.imu_rate / self.average as f64).sqrt();
        let (sb0, dt) = if bias_ready {
            (n / (2.0 * self.bias_tau).sqrt(), (t - bias_stamp.unwrap_or(t)).max(0.0))
        } else {
            (self.turn_on_sd, 0.0)
        };
        let bias = (sb0 * sb0 + self.noise_k * self.noise_k * dt).sqrt();
        let motion = self.motion_ratio * rate;
        (white * white + bias * bias + motion * motion).sqrt()
    }
}

/// Gyro bias learned at standstill (wheels report no motion, or without wheel data a quiet gyro).
pub struct BiasEstimator {
    pub bias: V3,
    pub ready: bool,
    pub samples: u64,
    pub stamp: Option<f64>,
    window: VecDeque<(f64, V3)>,
    pub still_s: f64,
    pub quiet_rate: f64,
    pub imu_rate: f64,
    pub bias_tau: f64,
}

impl BiasEstimator {
    pub fn new(bias: V3, still_s: f64, quiet_rate: f64, imu_rate: f64, bias_tau: f64) -> Self {
        Self { bias, ready: false, samples: 0, stamp: None, window: VecDeque::new(), still_s, quiet_rate, imu_rate, bias_tau }
    }

    fn quiet(&self, t: f64) -> bool {
        if self.window.len() < 20 || t - self.window[0].0 < 0.9 * self.still_s {
            return false;
        }
        let n = self.window.len() as f64;
        let mut mean = [0.0; 3];
        for (_, g) in &self.window {
            for i in 0..3 {
                mean[i] += g[i] / n;
            }
        }
        let mut var = [0.0; 3];
        for (_, g) in &self.window {
            for i in 0..3 {
                var[i] += (g[i] - mean[i]).powi(2) / n;
            }
        }
        // Before the first bias estimate allow up to ~3 deg/s offset (a MEMS bias, not a turn).
        let mean_tol = if self.ready { self.quiet_rate } else { 0.05 };
        (0..3).all(|i| (mean[i] - self.bias[i]).abs() < mean_tol && var[i].sqrt() < self.quiet_rate)
    }

    /// Adds a raw gyro sample. `wheels_still`: Some(still?) when fresh wheel data decides, None to
    /// use the quiet-gyro test. Returns true if the bias was updated.
    pub fn add(&mut self, t: f64, g: V3, wheels_still: Option<bool>) -> bool {
        self.window.push_back((t, g));
        while self.window.front().is_some_and(|w| t - w.0 > self.still_s) {
            self.window.pop_front();
        }
        let still = wheels_still.unwrap_or_else(|| self.quiet(t));
        if !still {
            return false;
        }
        if !self.ready {
            let n = self.window.len() as f64;
            let mut m = [0.0; 3];
            for (_, s) in &self.window {
                for i in 0..3 {
                    m[i] += s[i];
                }
            }
            self.bias = [m[0] / n, m[1] / n, m[2] / n];
            self.ready = true;
        } else {
            let a = (1.0 / (self.imu_rate * self.bias_tau)).min(1.0);
            for i in 0..3 {
                self.bias[i] = (1.0 - a) * self.bias[i] + a * g[i];
            }
        }
        self.stamp = Some(t);
        self.samples += 1;
        true
    }
}

/// The whole ekf_inputs pipeline without ROS: callers pass message stamps and a monotonic clock
/// (seconds) so it can be driven by a live node or replayed deterministically.
pub struct EkfCore {
    pub wheel: WheelParams,
    pub model: GyroModel,
    pub bias: BiasEstimator,
    acc: Vec<V3>,
    wheel_still_since: Option<f64>,
    last_wheel_mono: Option<f64>,
    pub wheel_max_age: f64,
    pub last_sd: Option<f64>,
}

impl EkfCore {
    pub fn new(wheel: WheelParams, model: GyroModel, bias: BiasEstimator, wheel_max_age: f64) -> Self {
        Self { wheel, model, bias, acc: Vec::new(), wheel_still_since: None, last_wheel_mono: None, wheel_max_age, last_sd: None }
    }

    /// Wheel odometry at stamp t: returns calibrated (vx, wz) and the twist covariance.
    pub fn on_wheel(&mut self, t: f64, vx_in: f64, wz_in: f64, mono: f64) -> (f64, f64, [f64; 36]) {
        let out = wheel_twist(&self.wheel, vx_in, wz_in);
        if vx_in.abs() > 1e-3 || wz_in.abs() > 1e-3 {
            self.wheel_still_since = None;
        } else if self.wheel_still_since.is_none() {
            self.wheel_still_since = Some(t);
        }
        self.last_wheel_mono = Some(mono);
        out
    }

    /// Raw gyro sample at stamp t. Every `average` samples returns the bias-free mean and its sd.
    pub fn on_imu(&mut self, t: f64, g: V3, mono: f64) -> Option<(V3, f64)> {
        let fresh = self.last_wheel_mono.is_some_and(|m| mono - m < self.wheel_max_age);
        let wheels_still = fresh.then(|| self.wheel_still_since.is_some_and(|s| t - s >= self.bias.still_s));
        self.bias.add(t, g, wheels_still);
        self.acc.push(g);
        if self.acc.len() < self.model.average {
            return None;
        }
        let n = self.acc.len() as f64;
        let mut mean = [0.0; 3];
        for s in &self.acc {
            for i in 0..3 {
                mean[i] += s[i];
            }
        }
        for i in 0..3 {
            mean[i] = mean[i] / n - self.bias.bias[i];
        }
        self.acc.clear();
        let rate = (mean[0] * mean[0] + mean[1] * mean[1] + mean[2] * mean[2]).sqrt();
        let sd = self.model.sd(self.bias.ready, self.bias.stamp, t, rate);
        self.last_sd = Some(sd);
        Some((mean, sd))
    }

    pub fn wheel_age(&self, mono: f64) -> Option<f64> {
        self.last_wheel_mono.map(|m| mono - m)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_calibration_file() {
        let (v, infl) = parse_imu_noise(
            "# cuVSLAM IMU noise\n# 3.0 h static recording at 199.69 Hz; raw values x10.0\n\
             gyro_noise_density: 0.0013194513\ncalibration_frequency: 199.7\ngyro_static_bias: [0.1, -0.2, 0.3]\n",
        );
        assert_eq!(infl, Some(10.0));
        assert_eq!(v["gyro_noise_density"], CalValue::Num(0.0013194513));
        assert_eq!(v["gyro_static_bias"], CalValue::List(vec![0.1, -0.2, 0.3]));
    }

    #[test]
    fn npy_round_trip() {
        let header = "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }";
        let mut b = b"\x93NUMPY\x01\x00".to_vec();
        b.extend_from_slice(&(header.len() as u16).to_le_bytes());
        b.extend_from_slice(header.as_bytes());
        for x in [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0] {
            b.extend_from_slice(&x.to_le_bytes());
        }
        let (shape, data) = parse_npy_f64(&b).unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(data[5], 6.0);
    }

    #[test]
    fn wheel_covariance_grows_with_speed_and_reverse() {
        let p = WheelParams { lin_scale: 0.952, ang_scale: 0.71, vx_sd0: 0.01, vx_sd_rel: 0.1, vx_sd_rel_rev: 0.5, vy_sd: 0.01, wz_sd0: 0.01, wz_sd_rel: 0.4 };
        let (vx, wz, c) = wheel_twist(&p, 0.2, 1.0);
        assert!((vx - 0.1904).abs() < 1e-12 && (wz - 0.71).abs() < 1e-12);
        assert!((c[0] - (1e-4 + (0.1 * 0.1904f64).powi(2))).abs() < 1e-15);
        assert!((c[35] - (1e-4 + (0.4 * 0.71f64).powi(2))).abs() < 1e-15);
        let (_, _, r) = wheel_twist(&p, -0.2, 0.0);
        assert!(r[0] > c[0] * 10.0);
    }

    #[test]
    fn bias_learned_only_when_still() {
        let mut b = BiasEstimator::new([0.0; 3], 1.0, 0.01, 200.0, 5.0);
        // turning: wheels say moving -> nothing learned
        for i in 0..400 {
            assert!(!b.add(i as f64 * 0.005, [0.0, 0.0, 0.5], Some(false)));
        }
        // no wheel data, quiet gyro with a 0.02 rad/s offset on z: learned after ~1 s
        let mut learned_at = None;
        for i in 0..400 {
            let t = 2.0 + i as f64 * 0.005;
            if b.add(t, [0.0, 0.0, 0.02 + if i % 2 == 0 { 1e-4 } else { -1e-4 }], None) && learned_at.is_none() {
                learned_at = Some(t);
            }
        }
        assert!(b.ready && (b.bias[2] - 0.02).abs() < 1e-3, "{:?}", b.bias);
        assert!(learned_at.unwrap() - 2.0 >= 0.85);
    }

    #[test]
    fn gyro_sd_model() {
        let m = GyroModel { noise_n: Some(1e-4), noise_k: 1e-6, imu_rate: 200.0, average: 4, motion_ratio: 0.02, turn_on_sd: 0.001, bias_tau: 5.0, fallback_sd: 0.03 };
        let white = 1e-4 * 50f64.sqrt();
        let before = m.sd(false, None, 0.0, 0.0);
        assert!((before - (white * white + 1e-6).sqrt()).abs() < 1e-12);
        let after = m.sd(true, Some(10.0), 110.0, 1.0);
        let sb0 = 1e-4 / 10f64.sqrt();
        let want = (white * white + sb0 * sb0 + 1e-12 * 100.0 + 0.02f64.powi(2)).sqrt();
        assert!((after - want).abs() < 1e-12);
        let unc = GyroModel { noise_n: None, ..m };
        assert_eq!(unc.sd(false, None, 0.0, 0.0), 0.3);
    }
}