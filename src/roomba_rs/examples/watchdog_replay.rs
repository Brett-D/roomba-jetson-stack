//! Deterministic replay of vslam_watchdog's core for tools/ab_watchdog_offline.py. Reads events
//! from stdin, one per line (mono = monotonic seconds, stamps as sec,nanosec):
//!   P,name,value                                  parameter (before the first event)
//!   R,pause,reset,new_map,resume                  service availability, 0/1 each
//!   F,parent,child,qx,qy,qz,qw                    /tf_static transform (rotation)
//!   I,mono,sec,nsec,frame,wx,wy,wz,ax,ay,az       raw IMU
//!   W,mono,sec,nsec,x,y,qx,qy,qz,qw,vx,wz         wheel odometry
//!   S,mono,vo_state                               VIO status
//!   O,mono,sec,nsec,px,py,pz,qx,qy,qz,qw,pose_cov[36],lin[3],ang[3],twist_cov[36]   VIO odometry
//!   T,mono / U,mono                               10 Hz tick / 1 Hz status timer
//!   D,mono,service[,error]                        service response
//! and prints one line per output: TF,sec,nsec,base[7],anchor[7] / OD,sec,nsec,pos[3],quat[4],
//! pose_cov[36],lin[3],ang[3],twist_cov[36] / ST,json / CA,service / LG,level,message.
use std::collections::HashMap;
use std::io::{self, BufRead, BufWriter, Write};

use roomba_rs::static_tf::StaticTree;
use roomba_rs::watchdog::{Io, Level, Odom, ParamSource, Params, Pose, Stamp, Svc, Twist, Watchdog};

struct Overrides(HashMap<String, String>);

impl ParamSource for Overrides {
    fn float(&mut self, name: &str, d: f64) -> anyhow::Result<f64> {
        Ok(self.0.get(name).map_or(Ok(d), |v| v.parse())?)
    }
    fn int(&mut self, name: &str, d: i64) -> anyhow::Result<i64> {
        Ok(self.0.get(name).map_or(Ok(d), |v| v.parse())?)
    }
    fn boolean(&mut self, name: &str, d: bool) -> anyhow::Result<bool> {
        Ok(self.0.get(name).map_or(d, |v| v.eq_ignore_ascii_case("true")))
    }
    fn string(&mut self, name: &str, d: &str) -> anyhow::Result<String> {
        Ok(self.0.get(name).cloned().unwrap_or_else(|| d.to_string()))
    }
}

struct Printer<W: Write> {
    ready: [bool; 4],
    w: W,
}

fn idx(svc: Svc) -> usize {
    Svc::ALL.iter().position(|&s| s == svc).unwrap()
}

fn join(v: &[f64]) -> String {
    v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
}

impl<W: Write> Io for Printer<W> {
    fn service_ready(&self, svc: Svc) -> bool {
        self.ready[idx(svc)]
    }
    fn tf(&mut self, s: Stamp, (bp, bq): Pose, (ap, aq): Pose) {
        writeln!(self.w, "TF,{},{},{},{},{},{}", s.sec, s.nanosec, join(&bp), join(&bq), join(&ap), join(&aq)).unwrap();
    }
    fn odom(&mut self, o: Odom) {
        let t = &o.twist;
        writeln!(
            self.w,
            "OD,{},{},{},{},{},{},{},{}",
            o.stamp.sec,
            o.stamp.nanosec,
            join(&o.pos),
            join(&o.quat),
            join(&o.pose_covariance),
            join(&t.linear),
            join(&t.angular),
            join(&t.covariance)
        )
        .unwrap();
    }
    fn call(&mut self, svc: Svc) {
        writeln!(self.w, "CA,{}", svc.key()).unwrap();
    }
    fn log(&mut self, level: Level, msg: String) {
        let l = match level {
            Level::Info => "info",
            Level::Warn => "warn",
            Level::Error => "error",
        };
        writeln!(self.w, "LG,{l},{msg}").unwrap();
    }
}

fn main() -> anyhow::Result<()> {
    let mut overrides = HashMap::new();
    let mut core: Option<Watchdog> = None;
    let mut tree = StaticTree::default();
    let mut out = Printer { ready: [true; 4], w: BufWriter::new(io::stdout().lock()) };
    for line in io::stdin().lock().lines() {
        let line = line?;
        let f: Vec<&str> = line.split(',').collect();
        if f[0] == "P" {
            overrides.insert(f[1].to_string(), f[2..].join(","));
            continue;
        }
        if core.is_none() {
            core = Some(Watchdog::new(Params::load(&mut Overrides(std::mem::take(&mut overrides)))?, 0.0));
        }
        let w = core.as_mut().unwrap();
        let num = |i: usize| f[i].parse::<f64>().unwrap();
        let v3 = |i: usize| [num(i), num(i + 1), num(i + 2)];
        let v4 = |i: usize| [num(i), num(i + 1), num(i + 2), num(i + 3)];
        let c36 = |i: usize| -> [f64; 36] { std::array::from_fn(|k| num(i + k)) };
        let stamp = || Stamp { sec: f[2].parse().unwrap(), nanosec: f[3].parse().unwrap() };
        match f[0] {
            "R" => out.ready = std::array::from_fn(|k| f[1 + k] == "1"),
            "F" => tree.insert(f[1], f[2], v4(3)),
            "I" => {
                let (base, frame) = (w.p.base_frame.clone(), f[4]);
                w.on_imu(stamp().secs(), v3(5), v3(8), num(1), || tree.rotation(&base, frame), &mut out);
            }
            "W" => w.on_wheel(stamp().secs(), num(4), num(5), v4(6), num(10), num(11), num(1)),
            "S" => w.on_status(f[2].parse()?, num(1), &mut out),
            "O" => {
                let odom = Odom {
                    stamp: stamp(),
                    pos: v3(4),
                    quat: v4(7),
                    pose_covariance: c36(11),
                    twist: Twist { linear: v3(47), angular: v3(50), covariance: c36(53) },
                };
                w.on_odom(&odom, num(1), &mut out);
            }
            "T" => w.tick(num(1), &mut out),
            "U" => {
                let status = w.status(num(1));
                writeln!(out.w, "ST,{status}")?;
            }
            "D" => {
                let svc = *Svc::ALL.iter().find(|s| s.key() == f[2]).expect("service");
                w.service_done(svc, f.get(3).map(|e| e.to_string()), num(1), &mut out);
            }
            other => anyhow::bail!("unknown event {other}"),
        }
    }
    Ok(())
}
