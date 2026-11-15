//! Drives roomba_rs::person with a recorded message sequence for tools/ab_person_detector_offline.py:
//! binary events on stdin (camera infos, images, TF, map, zones, localization, grid toggles, clock),
//! one JSON line per processed frame and per log line on stdout.
//!   person_replay PARAMS_JSON < events > results
//! PARAMS_JSON holds the node parameters ({"classes": [0, 15, 16], ...}; missing = default).

use std::io::{BufReader, BufWriter, Read, Write};

use serde_json::{json, Value};

use roomba_rs::person::{Detector, Frame, ImageView, Log, Params, TimeSync3};
use roomba_rs::tf_buffer::{TfBuffer, Transform};
use roomba_rs::zones::Level;

struct Reader<R: Read>(R);

impl<R: Read> Reader<R> {
    fn bytes<const N: usize>(&mut self) -> std::io::Result<[u8; N]> {
        let mut b = [0u8; N];
        self.0.read_exact(&mut b)?;
        Ok(b)
    }
    fn u8(&mut self) -> std::io::Result<u8> {
        Ok(self.bytes::<1>()?[0])
    }
    fn u32(&mut self) -> std::io::Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn i64(&mut self) -> std::io::Result<i64> {
        Ok(i64::from_le_bytes(self.bytes()?))
    }
    fn f32(&mut self) -> std::io::Result<f32> {
        Ok(f32::from_le_bytes(self.bytes()?))
    }
    fn f64(&mut self) -> std::io::Result<f64> {
        Ok(f64::from_le_bytes(self.bytes()?))
    }
    fn f64s<const N: usize>(&mut self) -> std::io::Result<[f64; N]> {
        let mut v = [0.0; N];
        for x in &mut v {
            *x = self.f64()?;
        }
        Ok(v)
    }
    fn blob(&mut self) -> std::io::Result<Vec<u8>> {
        let n = self.u32()? as usize;
        let mut v = vec![0u8; n];
        self.0.read_exact(&mut v)?;
        Ok(v)
    }
    fn string(&mut self) -> std::io::Result<String> {
        Ok(String::from_utf8_lossy(&self.blob()?).into_owned())
    }
}

struct Img {
    stamp: i64,
    frame_id: String,
    width: usize,
    height: usize,
    step: usize,
    data: Vec<u8>,
}

fn view(m: &Img) -> ImageView<'_> {
    ImageView { data: &m.data, width: m.width, height: m.height, step: m.step }
}

/// zlib.crc32
fn crc32(data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let t = TABLE.get_or_init(|| {
        std::array::from_fn(|i| (0..8).fold(i as u32, |c, _| if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 }))
    });
    !data.iter().fold(!0u32, |c, &b| t[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8))
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn base64(data: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for k in 0..4 {
            s.push(if k <= c.len() { A[(n >> (18 - 6 * k) & 63) as usize] as char } else { '=' });
        }
    }
    s
}

fn params(v: &Value) -> Params {
    let mut p = Params::default();
    let s = |k: &str, d: &mut String| {
        if let Some(x) = v[k].as_str() {
            *d = x.into();
        }
    };
    let f = |k: &str, d: &mut f64| {
        if let Some(x) = v[k].as_f64() {
            *d = x;
        }
    };
    let i = |k: &str, d: &mut i64| {
        if let Some(x) = v[k].as_i64() {
            *d = x;
        }
    };
    let b = |k: &str, d: &mut bool| {
        if let Some(x) = v[k].as_bool() {
            *d = x;
        }
    };
    s("engine_path", &mut p.engine_path);
    f("conf_threshold", &mut p.conf_threshold);
    f("nms_threshold", &mut p.nms_threshold);
    if let Some(c) = v["classes"].as_array() {
        p.classes = c.iter().filter_map(Value::as_i64).collect();
    }
    i("jpeg_quality", &mut p.jpeg_quality);
    i("min_depth_pixels", &mut p.min_depth_pixels);
    s("map_frame", &mut p.map_frame);
    b("ignore_zones", &mut p.ignore_zones);
    b("line_of_sight", &mut p.line_of_sight);
    i("occupied_threshold", &mut p.occupied_threshold);
    i("min_blocking_cells", &mut p.min_blocking_cells);
    f("target_margin_m", &mut p.target_margin_m);
    b("require_localization", &mut p.require_localization);
    b("floor_grid", &mut p.floor_grid);
    s("floor_grid_fallback_frame", &mut p.floor_grid_fallback_frame);
    p
}

fn level(l: Level) -> &'static str {
    match l {
        Level::Info => "info",
        Level::Warn => "warn",
        Level::Error => "error",
    }
}

fn emit(out: &mut impl Write, logs: Vec<Log>) -> std::io::Result<()> {
    for (l, text) in logs {
        writeln!(out, "{}", json!({"log": [level(l), text]}))?;
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).ok_or_else(|| anyhow::anyhow!("usage: person_replay PARAMS_JSON < events"))?;
    let p = params(&serde_json::from_str(&std::fs::read_to_string(path)?)?);
    let mut out = BufWriter::new(std::io::stdout().lock());
    let (mut det, loaded) = Detector::new(p).map_err(|e| anyhow::anyhow!(e))?;
    emit(&mut out, vec![loaded])?;
    let mut tf = TfBuffer::new();
    let mut sync = TimeSync3::<Img>::new(10);
    let (mut now, mut annotate) = (0.0f64, true);
    let mut r = Reader(BufReader::with_capacity(1 << 22, std::io::stdin().lock()));
    let mut frames = 0u64;
    loop {
        let kind = match r.u8() {
            Ok(k) => k,
            Err(_) => break,
        };
        match kind {
            0 => break,
            1 => det.on_left_info(r.f64s::<12>()?),
            2 => det.on_right_info(r.f64s::<12>()?),
            3 => {
                let input = r.u8()? as usize;
                let stamp = r.i64()?;
                let frame_id = r.string()?;
                let (width, height, step) = (r.u32()? as usize, r.u32()? as usize, r.u32()? as usize);
                let data = r.blob()?;
                let Some([c, d, rt]) = sync.add(input, stamp, Img { stamp, frame_id, width, height, step, data }) else { continue };
                let frame = Frame { stamp_ns: c.stamp, frame_id: &c.frame_id, color: view(&c), depth: view(&d), right: view(&rt) };
                let mut logs = Vec::new();
                let res = det.process(&frame, annotate, &tf, &|| now, &mut logs).map_err(|e| anyhow::anyhow!(e))?;
                emit(&mut out, logs)?;
                let Some(o) = res else { continue };
                frames += 1;
                let d3 = |v: &[roomba_rs::person::Det3d]| -> Vec<Value> {
                    v.iter().map(|d| json!([d.position, d.size, d.class_id, d.score, d.id])).collect()
                };
                let line = json!({
                    "frame": c.stamp,
                    "in_crc": crc32(&f32_bytes(det.engine().input())),
                    "out_crc": crc32(&f32_bytes(det.engine().output())),
                    "d2": o.det2d.iter().map(|d| json!([d.center, d.size, d.class_id, d.score])).collect::<Vec<_>>(),
                    "d3": d3(&o.det3d),
                    "ig": d3(&o.ignored),
                    "mk": o.markers.iter().map(|m| json!([m.ns, m.id, m.kind, m.position, m.scale, m.color.map(f64::from), m.text])).collect::<Vec<_>>(),
                    "img_crc": o.annotated.as_ref().map(|a| crc32(&a.data)),
                    "jpeg": o.jpeg.as_ref().map(|j| base64(j)),
                    "src": o.sources.iter().map(|s| format!("{s:?}").to_lowercase()).collect::<Vec<_>>(),
                });
                writeln!(out, "{line}")?;
                out.flush()?;
            }
            4 => {
                let is_static = r.u8()? != 0;
                for _ in 0..r.u32()? {
                    let (parent, child, stamp) = (r.string()?, r.string()?, r.i64()?);
                    let v = r.f64s::<7>()?;
                    let t = Transform { translation: [v[0], v[1], v[2]], rotation: [v[3], v[4], v[5], v[6]] };
                    tf.set_transform(&parent, &child, stamp, &t, "default_authority", is_static).ok();
                }
            }
            5 => {
                let frame = r.string()?;
                let (w, h, res) = (r.u32()? as usize, r.u32()? as usize, r.f32()?);
                let origin = r.f64s::<2>()?;
                let q = r.f64s::<4>()?;
                let data: Vec<i8> = r.blob()?.into_iter().map(|b| b as i8).collect();
                det.on_map(w, h, res, origin, q, &frame, &data);
            }
            6 => {
                let text = r.string()?;
                emit(&mut out, vec![det.on_zones(&text)])?;
            }
            7 => emit(&mut out, det.on_localized().into_iter().collect())?,
            8 => {
                let on = r.u8()? != 0;
                emit(&mut out, vec![det.on_grid_toggle(on)])?;
            }
            9 => {
                now = r.f64()?;
                annotate = r.u8()? != 0;
            }
            10 => {
                // timing differs run to run: only report whether a line was logged
                let had = det.stats().is_some();
                writeln!(out, "{}", json!({"stats": had}))?;
                out.flush()?;
            }
            k => anyhow::bail!("unknown event {k}"),
        }
    }
    writeln!(out, "{}", json!({"end": frames}))?;
    out.flush()?;
    Ok(())
}
