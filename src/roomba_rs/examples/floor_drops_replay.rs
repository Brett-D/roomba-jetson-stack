//! Replay of roomba_rs::floor::drop_points for tools/ab_floor_drops_offline.py: one JSON object per
//! input line (every float as its IEEE-754 bit pattern, u64), prints the drop points of each as a
//! JSON array of [x, y, z] bit patterns.
use std::io::{self, BufRead, BufWriter, Write};

use serde_json::{json, Value};

use roomba_rs::floor::drop_points;

fn f(v: &Value) -> f64 {
    f64::from_bits(v.as_u64().expect("float bits"))
}

fn v3(v: &Value) -> [f64; 3] {
    let a = v.as_array().expect("3-vector");
    [f(&a[0]), f(&a[1]), f(&a[2])]
}

fn main() -> anyhow::Result<()> {
    let mut out = BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let c: Value = serde_json::from_str(&line)?;
        let pts: Vec<[f64; 3]> = c["pts"].as_array().unwrap().iter().map(v3).collect();
        let h: Vec<f64> = c["h"].as_array().unwrap().iter().map(f).collect();
        let d = drop_points(
            &pts,
            &h,
            f(&c["floor_h"]),
            v3(&c["up"]),
            f(&c["below"]),
            f(&c["max_range"]),
            f(&c["lift"]),
            v3(&c["fwd"]),
            v3(&c["side"]),
            f(&c["cell"]),
            c["min_points"].as_u64().unwrap() as usize,
        );
        let q: Vec<Value> = d.iter().map(|p| json!([p[0].to_bits(), p[1].to_bits(), p[2].to_bits()])).collect();
        writeln!(out, "{}", Value::from(q))?;
    }
    out.flush()?;
    Ok(())
}
