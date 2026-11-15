//! Evaluates roomba_rs::scene_filter on cases from tools/ab_scene_filter.py (JSON lines in, JSON lines out).
//!   cargo run --release --example scene_filter_replay -- CASES > OUT

use std::io::{BufRead, BufReader, BufWriter, Write};

use serde_json::{json, Value};

use roomba_rs::scene_filter::{point_in_polygon, GridLineOfSight, ZoneSet};

fn xy(v: &Value) -> [f64; 2] {
    [v[0].as_f64().unwrap_or(f64::NAN), v[1].as_f64().unwrap_or(f64::NAN)]
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).ok_or_else(|| anyhow::anyhow!("usage: scene_filter_replay CASES"))?;
    let mut out = BufWriter::new(std::io::stdout().lock());
    for line in BufReader::new(std::fs::File::open(path)?).lines() {
        let c: Value = serde_json::from_str(&line?)?;
        let queries = c["q"].as_array().cloned().unwrap_or_default();
        let result = if let Some(poly) = c["pip"].as_array() {
            let poly: Vec<[f64; 2]> = poly.iter().map(xy).collect();
            json!(queries.iter().map(|q| { let [x, y] = xy(q); point_in_polygon(x, y, &poly) }).collect::<Vec<_>>())
        } else if let Some(text) = c["zones"].as_str() {
            let mut z = ZoneSet::default();
            match z.load_json(text) {
                Ok(()) => json!({"frame": z.frame, "names": queries.iter().map(|q| { let [x, y] = xy(q); z.containing(x, y) }).collect::<Vec<_>>()}),
                Err(_) => json!("error"),
            }
        } else {
            let g = &c["grid"];
            let data: Vec<i8> = g["data"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i8).collect();
            let mut los = GridLineOfSight::new(c["threshold"].as_i64().unwrap_or(65), 2);
            let q = |i: usize| g["q"][i].as_f64().unwrap();
            los.update(
                g["w"].as_u64().unwrap() as usize,
                g["h"].as_u64().unwrap() as usize,
                g["res"].as_f64().unwrap() as f32,
                [g["ox"].as_f64().unwrap(), g["oy"].as_f64().unwrap()],
                [q(0), q(1), q(2), q(3)],
                "map",
                &data,
            );
            json!(c["rays"].as_array().unwrap().iter().map(|r| {
                let v = |i: usize| r[i].as_f64().unwrap();
                los.blocked([v(0), v(1)], [v(2), v(3)], v(4), v(5))
            }).collect::<Vec<_>>())
        };
        writeln!(out, "{result}")?;
    }
    Ok(())
}
