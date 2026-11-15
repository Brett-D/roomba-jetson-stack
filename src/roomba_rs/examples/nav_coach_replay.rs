//! Deterministic replay of nav_coach's core for tools/ab_nav_coach_offline.py. Reads events from
//! stdin, one JSON array per line [kind, mono_s, ...]:
//!   ["P", 0, {params}, cloud_file]                    parameters (first line)
//!   ["TF", t, is_static, [[parent, child, stamp_ns, tx, ty, tz, qx, qy, qz, qw], ...]]
//!   ["V", t, vx, wz] /cmd_vel_nav   ["O", t, vx, wz] /cmd_vel   ["CM", t, action_type]
//!   ["OB", t, frame, width, offset, len]               /perception/obstacles (data in cloud_file)
//!   ["ST", t, [[sec, nanosec, status], ...]]           goal status array
//!   ["FB", t, distance_remaining]                      feedback
//!   ["K", t]                                           0.2 s tick
//!   ["MM", t, offset, len, [tx, ty, tz, qx, qy, qz, qw], xy_offset]   diagnostic: base-frame x, y of
//!       every point against numpy's (float64 pairs at xy_offset in cloud_file + ".xy")
//! and prints ["IN", json] per tick, ["AH", points_in_stop_zone|null, blocked] per cloud and
//! ["MM", points, mismatches_fma, mismatches_plain] per diagnostic.
use std::fs::File;
use std::io::{self, BufRead, BufWriter, Read, Seek, SeekFrom, Write};

use serde_json::{json, Map, Value};

use roomba_rs::nav_coach::{points_ahead, quat_to_matrix, NavCoach, Params};
use roomba_rs::tf_buffer::{TfBuffer, Transform};
use roomba_rs::watchdog::ParamSource;

struct Overrides(Map<String, Value>);

impl ParamSource for Overrides {
    fn float(&mut self, name: &str, d: f64) -> anyhow::Result<f64> {
        Ok(self.0.get(name).and_then(Value::as_f64).unwrap_or(d))
    }
    fn int(&mut self, name: &str, d: i64) -> anyhow::Result<i64> {
        Ok(self.0.get(name).and_then(Value::as_i64).unwrap_or(d))
    }
    fn boolean(&mut self, name: &str, d: bool) -> anyhow::Result<bool> {
        Ok(self.0.get(name).and_then(Value::as_bool).unwrap_or(d))
    }
    fn string(&mut self, name: &str, d: &str) -> anyhow::Result<String> {
        Ok(self.0.get(name).and_then(Value::as_str).unwrap_or(d).to_string())
    }
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn read(file: &mut File, offset: u64, len: usize) -> io::Result<Vec<u8>> {
    file.seek(SeekFrom::Start(offset))?;
    let mut buf = vec![0u8; len];
    file.read_exact(&mut buf)?;
    Ok(buf)
}

/// Base-frame x, y of each point: fused multiply-adds (as the node) and plain products and sums.
fn mismatches(data: &[u8], t: [f64; 3], q: [f64; 4], expected: &[u8]) -> (usize, usize, usize) {
    let r = quat_to_matrix(q[0], q[1], q[2], q[3]);
    let (mut n, mut bad_fma, mut bad_plain) = (0, 0, 0);
    for (p, e) in data.chunks_exact(12).zip(expected.chunks_exact(16)) {
        let g = |b: &[u8], i: usize| f32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as f64;
        let (px, py, pz) = (g(p, 0), g(p, 4), g(p, 8));
        let want = |i: usize| f64::from_le_bytes(e[i..i + 8].try_into().unwrap());
        let fma = |row: usize, tt: f64| pz.mul_add(r[row][2], py.mul_add(r[row][1], px * r[row][0])) + tt;
        let plain = |row: usize, tt: f64| px * r[row][0] + py * r[row][1] + pz * r[row][2] + tt;
        n += 1;
        if fma(0, t[0]).to_bits() != want(0).to_bits() || fma(1, t[1]).to_bits() != want(8).to_bits() {
            bad_fma += 1;
        }
        if plain(0, t[0]).to_bits() != want(0).to_bits() || plain(1, t[1]).to_bits() != want(8).to_bits() {
            bad_plain += 1;
        }
    }
    (n, bad_fma, bad_plain)
}

fn main() -> anyhow::Result<()> {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let first: Value = serde_json::from_str(&lines.next().expect("no parameter line")?)?;
    let params = Params::load(&mut Overrides(first[2].as_object().cloned().unwrap_or_default()))?;
    let cloud_path = first[3].as_str().unwrap_or("").to_string();
    let mut clouds = File::open(&cloud_path).ok();
    let mut xy = File::open(format!("{cloud_path}.xy")).ok();
    let mut out = BufWriter::new(io::stdout().lock());
    let mut tf = TfBuffer::new();
    let base = params.base_frame.clone();
    let (stop_dist, half_width) = (params.stop_dist, params.stop_half_width);
    let mut core = NavCoach::new(params);
    for line in lines {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let e: Value = serde_json::from_str(&line)?;
        let t = f(&e[1]);
        match e[0].as_str().unwrap_or("") {
            "TF" => {
                let is_static = e[2].as_bool().unwrap_or(false);
                for x in e[3].as_array().unwrap() {
                    let tr = Transform { translation: [f(&x[3]), f(&x[4]), f(&x[5])], rotation: [f(&x[6]), f(&x[7]), f(&x[8]), f(&x[9])] };
                    let _ = tf.set_transform(x[0].as_str().unwrap(), x[1].as_str().unwrap(), x[2].as_i64().unwrap(), &tr, "default_authority", is_static);
                }
            }
            "V" => core.on_cmd_nav(f(&e[2]), f(&e[3]), t),
            "O" => core.on_cmd_out(f(&e[2]), f(&e[3]), t),
            "CM" => core.on_collision_monitor(e[2].as_u64().unwrap() as u8, t),
            "OB" => {
                let frame = e[2].as_str().unwrap();
                let width = e[3].as_u64().unwrap() as u32;
                let data = read(clouds.as_mut().expect("no cloud file"), e[4].as_u64().unwrap(), e[5].as_u64().unwrap() as usize)?;
                let lookup = tf.lookup_transform(&base, frame, 0).ok().map(|s| (s.transform.translation, s.transform.rotation));
                let n = lookup.map(|(tt, q)| points_ahead(if width > 0 { &data } else { &[] }, tt, q, stop_dist, half_width));
                core.on_obstacles(&data, width, lookup, t);
                writeln!(out, "{}", json!(["AH", n, core.blocked()]))?;
            }
            "ST" => {
                let list: Vec<(i32, u32, i8)> = e[2]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|s| (s[0].as_i64().unwrap() as i32, s[1].as_u64().unwrap() as u32, s[2].as_i64().unwrap() as i8))
                    .collect();
                core.on_status(&list, t);
            }
            "FB" => core.on_feedback(f(&e[2]) as f32),
            "K" => writeln!(out, "{}", json!(["IN", core.tick(t)]))?,
            "MM" => {
                let data = read(clouds.as_mut().expect("no cloud file"), e[2].as_u64().unwrap(), e[3].as_u64().unwrap() as usize)?;
                let p = e[4].as_array().unwrap();
                let expected = read(xy.as_mut().expect("no .xy file"), e[5].as_u64().unwrap(), data.len() / 12 * 16)?;
                let (n, a, b) = mismatches(&data, [f(&p[0]), f(&p[1]), f(&p[2])], [f(&p[3]), f(&p[4]), f(&p[5]), f(&p[6])], &expected);
                writeln!(out, "{}", json!(["MM", n, a, b]))?;
            }
            other => anyhow::bail!("unknown event {other}"),
        }
    }
    out.flush()?;
    Ok(())
}
