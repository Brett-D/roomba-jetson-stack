//! Replays ignore_zones events through roomba_rs::zones for tools/ab_ignore_zones_offline.py.
//!
//! Input: JSON lines {"ev": "start", "params": {...}} | {"ev": "tf", "static", "parent", "child",
//! "stamp_ns", "t", "q"} | {"ev": "click", "frame", "p"} | {"ev": "cmd", "text", "now"} |
//! {"ev": "markers"} | {"ev": "expire", "now"}. Output: one JSON line per event with what was
//! published per topic and logged, and the zones file (and its .bak) afterwards.
//!   cargo run --release --example zones_replay -- EVENTS > OUT

use std::io::{BufRead, BufReader, BufWriter, Write};

use serde_json::{json, Value};

use roomba_rs::tf_buffer::{TfBuffer, Transform};
use roomba_rs::zones::{expand_user, IgnoreZones, Level, MarkerSpec, Output, Params, DELETEALL};

fn marker(m: &MarkerSpec, frame: &str) -> Value {
    json!({
        "ns": m.ns, "id": m.id, "type": m.kind, "action": m.action, "frame": frame,
        "position": m.position, "orientation": [0.0, 0.0, 0.0, m.orientation_w], "scale": m.scale,
        "color": m.color.iter().map(|&c| c as f64).collect::<Vec<_>>(), "points": m.points, "text": m.text,
    })
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).ok_or_else(|| anyhow::anyhow!("usage: zones_replay EVENTS"))?;
    let mut out = BufWriter::new(std::io::stdout().lock());
    let mut tf = TfBuffer::new();
    let mut zones: Option<IgnoreZones> = None;
    for line in BufReader::new(std::fs::File::open(path)?).lines() {
        let ev: Value = serde_json::from_str(&line?)?;
        let f = |k: &str| ev[k].as_f64().unwrap_or(0.0);
        let mut outputs = Vec::new();
        let mut filter_info = Vec::new();
        match ev["ev"].as_str().unwrap_or("") {
            "start" => {
                let p = &ev["params"];
                let params = Params {
                    zones_file: expand_user(p["zones_file"].as_str().unwrap_or("")),
                    map_frame: p["map_frame"].as_str().unwrap_or("map").into(),
                    close_distance_m: p["close_distance_m"].as_f64().unwrap_or(0.25),
                    keepout_margin_m: p["keepout_margin_m"].as_f64().unwrap_or(0.2),
                    keepout_resolution_m: p["keepout_resolution_m"].as_f64().unwrap_or(0.05),
                    clear_on_start: p["clear_on_start"].as_bool().unwrap_or(false),
                };
                tf = TfBuffer::new();
                filter_info.push(json!({"type": 0, "topic": "/keepout_filter_mask", "base": 0.0, "multiplier": 1.0, "frame": params.map_frame}));
                let (z, o) = IgnoreZones::new(params);
                zones = Some(z);
                outputs = o;
            }
            "tf" => {
                let v = |k: &str, i: usize| ev[k][i].as_f64().unwrap_or(f64::NAN);
                let t = Transform { translation: [v("t", 0), v("t", 1), v("t", 2)], rotation: [v("q", 0), v("q", 1), v("q", 2), v("q", 3)] };
                let _ = tf.set_transform(
                    ev["parent"].as_str().unwrap_or(""),
                    ev["child"].as_str().unwrap_or(""),
                    ev["stamp_ns"].as_i64().unwrap_or(0),
                    &t,
                    "default_authority",
                    ev["static"].as_bool().unwrap_or(false),
                );
            }
            "click" => {
                let p = [ev["p"][0].as_f64().unwrap_or(0.0), ev["p"][1].as_f64().unwrap_or(0.0), ev["p"][2].as_f64().unwrap_or(0.0)];
                outputs = zones.as_mut().unwrap().on_click(ev["frame"].as_str().unwrap_or(""), p, &tf);
            }
            "cmd" => outputs = zones.as_mut().unwrap().on_command(ev["text"].as_str().unwrap_or(""), f("now")),
            "markers" => outputs = vec![zones.as_ref().unwrap().markers()],
            "expire" => outputs = zones.as_mut().unwrap().expire(f("now")),
            other => anyhow::bail!("unknown event {other}"),
        }
        let z = zones.as_ref().unwrap();
        let frame = z.params.map_frame.clone();
        let (mut topics_zones, mut masks, mut markers, mut logs) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for o in outputs {
            match o {
                Output::Zones(s) => topics_zones.push(json!(s)),
                Output::Mask(m) => masks.push(json!({
                    "frame": frame, "resolution": z.params.keepout_resolution_m as f32 as f64, "width": m.width,
                    "height": m.height, "origin": [m.origin_x, m.origin_y, 0.0, 1.0], "data": m.data,
                })),
                Output::Markers(ms) => {
                    let clear = MarkerSpec {
                        ns: String::new(),
                        id: 0,
                        kind: 0,
                        action: DELETEALL,
                        position: [0.0; 3],
                        orientation_w: 1.0,
                        scale: [0.0; 3],
                        color: [0.0; 4],
                        points: Vec::new(),
                        text: String::new(),
                    };
                    markers.push(json!(std::iter::once(&clear).chain(ms.iter()).map(|m| marker(m, &frame)).collect::<Vec<_>>()));
                }
                Output::Log(level, text) => {
                    let l = match level {
                        Level::Info => "info",
                        Level::Warn => "warn",
                        Level::Error => "error",
                    };
                    logs.push(json!([l, text]));
                }
            }
        }
        let read = |p: std::path::PathBuf| std::fs::read(p).ok().map(|b| String::from_utf8_lossy(&b).into_owned());
        let file = z.zones_file().to_path_buf();
        let bak = std::path::PathBuf::from(format!("{}.bak", file.display()));
        let rec = json!({"zones": topics_zones, "mask": masks, "markers": markers, "filter_info": filter_info, "logs": logs,
                         "file": [read(file), read(bak)]});
        writeln!(out, "{rec}")?;
    }
    Ok(())
}
