//! Deterministic replay of ekf_inputs' core for tools/ab_ekf_offline.py: reads events from stdin
//!   C,noise_n,noise_k,imu_rate,bias_x,bias_y,bias_z,quiet_rate   (first line; noise_n 0 = uncalibrated)
//!   W,mono,stamp,vx,wz            wheel odometry
//!   I,mono,stamp,gx,gy,gz         raw gyro
//! and prints W,stamp,vx,wz,cov0,cov35 / I,stamp,mx,my,mz,sd for each output (node defaults otherwise).
use std::io::{self, BufRead, BufWriter, Write};

use roomba_rs::ekf::{BiasEstimator, EkfCore, GyroModel, WheelParams};

fn main() {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines().map(|l| l.expect("stdin"));
    let c: Vec<f64> = lines.next().expect("C line").split(',').skip(1).map(|x| x.parse().unwrap()).collect();
    let model = GyroModel {
        noise_n: (c[0] > 0.0).then_some(c[0]),
        noise_k: c[1],
        imu_rate: c[2],
        average: 4,
        motion_ratio: 0.02,
        turn_on_sd: 0.001,
        bias_tau: 5.0,
        fallback_sd: 0.03,
    };
    let wheel = WheelParams { lin_scale: 0.952, ang_scale: 0.71, vx_sd0: 0.01, vx_sd_rel: 0.1, vx_sd_rel_rev: 0.5, vy_sd: 0.01, wz_sd0: 0.01, wz_sd_rel: 0.4 };
    let bias = BiasEstimator::new([c[3], c[4], c[5]], 1.0, c[6], c[2], 5.0);
    let mut core = EkfCore::new(wheel, model, bias, 0.5);
    let mut out = BufWriter::new(io::stdout().lock());
    for line in lines {
        let p: Vec<&str> = line.split(',').collect();
        let num = |i: usize| p[i].parse::<f64>().unwrap();
        let (mono, t) = (num(1), num(2));
        match p[0] {
            "W" => {
                let (vx, wz, cov) = core.on_wheel(t, num(3), num(4), mono);
                writeln!(out, "W,{t},{vx:e},{wz:e},{:e},{:e}", cov[0], cov[35]).unwrap();
            }
            "I" => {
                if let Some((m, sd)) = core.on_imu(t, [num(3), num(4), num(5)], mono) {
                    writeln!(out, "I,{t},{:e},{:e},{:e},{sd:e}", m[0], m[1], m[2]).unwrap();
                }
            }
            _ => {}
        }
    }
}