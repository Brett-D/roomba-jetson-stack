//! ignore_zones' logic: exclusion zones (people inside are ignored) and navigation keep-out zones,
//! drawn in the 3D panel (clicked points) or sent as commands by the web app and the explorer (its
//! bump and cliff marks), saved next to the map database and published as JSON, markers and a
//! Nav2 keep-out filter mask. Same behaviour as roomba_perception/ignore_zones.py, including the
//! zones file format (Python's json.dump with indent=1), the JSON text published, float rounding
//! and the mask computation (numpy's float operations, in order), checked by
//! tools/ab_ignore_zones_offline.py.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde_json::{Map, Number, Value};

use crate::tf_buffer::TfBuffer;

pub const MAX_ZONES: usize = 100;
pub const MAX_ZONE_POINTS: usize = 64;
pub const KINDS: [&str; 2] = ["exclusion", "keepout"];

// ---- Python compatibility: float repr/round/float(), str(), json.dumps, sum() ----

/// Python repr(float) (shortest round-trip digits; exponent below 1e-4 and from 1e16).
pub fn py_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    let s = format!("{x:e}");
    let (mant, exp) = s.split_once('e').unwrap_or((&s, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let digits: String = mant.chars().filter(char::is_ascii_digit).collect();
    let (n, decpt) = (digits.len() as i32, exp + 1);
    let body = if -4 < decpt && decpt <= 16 {
        if decpt <= 0 {
            format!("0.{}{digits}", "0".repeat(-decpt as usize))
        } else if decpt >= n {
            format!("{digits}{}.0", "0".repeat((decpt - n) as usize))
        } else {
            format!("{}.{}", &digits[..decpt as usize], &digits[decpt as usize..])
        }
    } else {
        let m = if n > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits.clone() };
        format!("{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    };
    if mant.starts_with('-') {
        format!("-{body}")
    } else {
        body
    }
}

/// Python round(x, ndigits): correctly rounded, halves to even.
pub fn py_round(x: f64, ndigits: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.ndigits$}").parse().unwrap_or(x)
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) if n.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

fn num(n: &Number) -> f64 {
    n.as_f64().unwrap_or(f64::NAN)
}

/// Python float(v) for a JSON value; Err is the TypeError/ValueError text.
pub fn py_float(v: &Value) -> Result<f64, String> {
    match v {
        Value::Number(n) => Ok(num(n)),
        Value::Bool(b) => Ok(*b as u8 as f64),
        Value::String(s) => parse_float(s).ok_or_else(|| format!("could not convert string to float: {}", repr_str(s))),
        _ => Err(format!("float() argument must be a string or a real number, not '{}'", type_name(v))),
    }
}

fn parse_float(s: &str) -> Option<f64> {
    let t = s.trim();
    let b = t.as_bytes();
    // PEP 515: single underscores between digits.
    for (i, &c) in b.iter().enumerate() {
        if c == b'_' && !(i > 0 && i + 1 < b.len() && b[i - 1].is_ascii_digit() && b[i + 1].is_ascii_digit()) {
            return None;
        }
    }
    let t = t.replace('_', "");
    let body = t.strip_prefix(['+', '-']).unwrap_or(&t).to_ascii_lowercase();
    let named = matches!(body.as_str(), "inf" | "infinity" | "nan");
    if !named && !body.bytes().all(|c| c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'+' | b'-')) {
        return None;
    }
    t.parse().ok()
}

/// Python repr(str), close enough for the characters _clean_name keeps.
fn repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::from(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

fn py_repr_value(v: &Value) -> String {
    match v {
        Value::String(s) => repr_str(s),
        _ => py_str(v),
    }
}

/// Python str(v) of a JSON value (as json.loads returns it).
pub fn py_str(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Number(n) if n.is_f64() => py_repr(num(n)),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        Value::Array(a) => format!("[{}]", a.iter().map(py_repr_value).collect::<Vec<_>>().join(", ")),
        Value::Object(m) => format!(
            "{{{}}}",
            m.iter().map(|(k, v)| format!("{}: {}", repr_str(k), py_repr_value(v))).collect::<Vec<_>>().join(", ")
        ),
    }
}

/// Python truthiness.
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => num(n) != 0.0,
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(m) => !m.is_empty(),
    }
}

/// Python == between JSON values (1 == 1.0 == True).
fn py_eq(a: &Value, b: &Value) -> bool {
    let n = |v: &Value| match v {
        Value::Number(n) => Some(num(n)),
        Value::Bool(b) => Some(*b as u8 as f64),
        _ => None,
    };
    match (a, b) {
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| py_eq(p, q)),
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| py_eq(v, w)))
        }
        _ => match (n(a), n(b)) {
            (Some(p), Some(q)) => p == q,
            (None, None) => a == b,
            _ => false,
        },
    }
}

fn float_json(x: f64) -> String {
    if x.is_nan() {
        "NaN".into()
    } else if x.is_infinite() {
        if x > 0.0 { "Infinity" } else { "-Infinity" }.into()
    } else {
        py_repr(x)
    }
}

fn json_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            c => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    let _ = write!(out, "\\u{u:04x}");
                }
            }
        }
    }
    out.push('"');
}

fn json_value(out: &mut String, v: &Value, indent: Option<usize>, level: usize) {
    let newline = |out: &mut String, level: usize| {
        if let Some(k) = indent {
            out.push('\n');
            out.push_str(&" ".repeat(k * level));
        }
    };
    let sep = if indent.is_some() { "," } else { ", " };
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) if n.is_f64() => out.push_str(&float_json(num(n))),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => json_str(out, s),
        Value::Array(a) if a.is_empty() => out.push_str("[]"),
        Value::Object(m) if m.is_empty() => out.push_str("{}"),
        Value::Array(a) => {
            out.push('[');
            for (i, item) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(sep);
                }
                newline(out, level + 1);
                json_value(out, item, indent, level + 1);
            }
            newline(out, level);
            out.push(']');
        }
        Value::Object(m) => {
            out.push('{');
            for (i, (k, item)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(sep);
                }
                newline(out, level + 1);
                json_str(out, k);
                out.push_str(": ");
                json_value(out, item, indent, level + 1);
            }
            newline(out, level);
            out.push('}');
        }
    }
}

/// Python json.dumps(v) (indent None) or json.dump(v, f, indent=k).
pub fn py_json(v: &Value, indent: Option<usize>) -> String {
    let mut out = String::new();
    json_value(&mut out, v, indent, 0);
    out
}

/// Python sum() of JSON numbers (int arithmetic until the first float, then Neumaier-compensated
/// float summation as in CPython 3.12).
fn py_sum(values: impl Iterator<Item = Value>) -> f64 {
    let mut values = values.peekable();
    let mut ints: i64 = 0;
    let mut f = loop {
        match values.next() {
            None => return ints as f64,
            Some(Value::Number(n)) if !n.is_f64() => ints += n.as_i64().unwrap_or(0),
            Some(Value::Bool(b)) => ints += b as i64,
            Some(v) => break ints as f64 + py_float(&v).unwrap_or(f64::NAN),
        }
    };
    let mut c = 0.0f64;
    for v in values {
        match v {
            Value::Number(n) if n.is_f64() => {
                let x = num(&n);
                let t = f + x;
                if f.abs() >= x.abs() {
                    c += (f - t) + x;
                } else {
                    c += (x - t) + f;
                }
                f = t;
            }
            other => f += py_float(&other).unwrap_or(f64::NAN),
        }
    }
    if c != 0.0 && c.is_finite() {
        f += c;
    }
    f
}

// ---- keep-out mask ----

/// A keep-out filter mask (nav_msgs/OccupancyGrid data, row-major, 100 = keep out).
#[derive(Clone, Debug, PartialEq)]
pub struct Mask {
    pub origin_x: f64,
    pub origin_y: f64,
    pub width: usize,
    pub height: usize,
    pub data: Vec<i8>,
}

/// keepout_mask(): cells covering the polygons plus their margins (`margins[i]` None = `margin`),
/// with numpy's operations in numpy's order. None without polygons.
pub fn keepout_mask(polygons: &[Vec<[f64; 2]>], resolution: f64, margin: f64, margins: &[Option<f64>]) -> Option<Mask> {
    if polygons.is_empty() {
        return None;
    }
    let margins: Vec<f64> = margins.iter().map(|m| m.unwrap_or(margin)).collect();
    let pts: Vec<[f64; 2]> = polygons.iter().flatten().copied().collect();
    let max_margin = margins.iter().copied().reduce(|a, b| if b > a { b } else { a }).unwrap_or(margin);
    let pad = max_margin + 2.0 * resolution;
    let min = |k: usize| pts.iter().map(|p| p[k]).fold(f64::INFINITY, f64::min);
    let max = |k: usize| pts.iter().map(|p| p[k]).fold(f64::NEG_INFINITY, f64::max);
    let (ox, oy) = (min(0) - pad, min(1) - pad);
    let w = ((max(0) + pad - ox) / resolution).ceil() as usize;
    let h = ((max(1) + pad - oy) / resolution).ceil() as usize;
    let xs: Vec<f64> = (0..w).map(|i| ox + (i as f64 + 0.5) * resolution).collect();
    let ys: Vec<f64> = (0..h).map(|i| oy + (i as f64 + 0.5) * resolution).collect();
    let mut blocked = vec![false; w * h];
    let mut inside = vec![false; w * h];
    let mut near = vec![false; w * h];
    for (poly, &poly_margin) in polygons.iter().zip(&margins) {
        inside.fill(false);
        near.fill(false);
        let n = poly.len();
        for k in 0..n {
            let ([x1, y1], [x2, y2]) = (poly[k], poly[(k + 1) % n]);
            let dy_or = if y2 - y1 != 0.0 { y2 - y1 } else { 1e-12 };
            let (dx, dy) = (x2 - x1, y2 - y1);
            let len2 = dx * dx + dy * dy;
            let len2 = if len2 != 0.0 { len2 } else { 1e-12 };
            for (r, &cy) in ys.iter().enumerate() {
                for (c, &cx) in xs.iter().enumerate() {
                    let i = r * w + c;
                    // Ray casting for the inside test.
                    if ((y1 > cy) != (y2 > cy)) && cx < (x2 - x1) * (cy - y1) / dy_or + x1 {
                        inside[i] = !inside[i];
                    }
                    // Distance to this edge for the margin.
                    let t = (((cx - x1) * dx + (cy - y1) * dy) / len2).clamp(0.0, 1.0);
                    if (cx - (x1 + t * dx)).hypot(cy - (y1 + t * dy)) <= poly_margin {
                        near[i] = true;
                    }
                }
            }
        }
        for i in 0..w * h {
            blocked[i] |= inside[i] | near[i];
        }
    }
    Some(Mask { origin_x: ox, origin_y: oy, width: w, height: h, data: blocked.iter().map(|&b| if b { 100 } else { 0 }).collect() })
}

// ---- the node's state ----

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Info,
    Warn,
    Error,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MarkerSpec {
    pub ns: String,
    pub id: i32,
    pub kind: i32,
    pub action: i32,
    pub position: [f64; 3],
    pub orientation_w: f64,
    pub scale: [f64; 3],
    pub color: [f32; 4],
    pub points: Vec<[f64; 3]>,
    pub text: String,
}

pub const LINE_STRIP: i32 = 4;
pub const SPHERE_LIST: i32 = 7;
pub const TEXT_VIEW_FACING: i32 = 9;
pub const ADD: i32 = 0;
pub const DELETEALL: i32 = 3;

/// What the node publishes or logs, in order.
#[derive(Clone, Debug, PartialEq)]
pub enum Output {
    /// /perception/ignore_zones
    Zones(String),
    /// /keepout_filter_mask
    Mask(Mask),
    /// /perception/ignore_zones/markers (a DELETEALL first, then these)
    Markers(Vec<MarkerSpec>),
    Log(Level, String),
}

#[derive(Clone, Debug)]
pub struct Params {
    pub zones_file: PathBuf,
    pub map_frame: String,
    pub close_distance_m: f64,
    pub keepout_margin_m: f64,
    pub keepout_resolution_m: f64,
    pub clear_on_start: bool,
}

/// os.path.expanduser for "~" and "~/...".
pub fn expand_user(path: &str) -> PathBuf {
    match (path.strip_prefix('~'), std::env::var_os("HOME")) {
        (Some(rest), Some(home)) if rest.is_empty() || rest.starts_with('/') => {
            PathBuf::from(format!("{}{rest}", home.to_string_lossy()))
        }
        _ => PathBuf::from(path),
    }
}

pub struct IgnoreZones {
    pub params: Params,
    pub zones: Vec<Map<String, Value>>,
    pub draft: Vec<[f64; 2]>,
}

fn name_of(z: &Map<String, Value>) -> Value {
    z.get("name").cloned().unwrap_or(Value::Null)
}

fn points_of(v: Option<&Value>) -> Vec<[f64; 2]> {
    let Some(Value::Array(pts)) = v else { return Vec::new() };
    pts.iter()
        .filter_map(|p| match p {
            Value::Array(xy) if xy.len() >= 2 => Some([py_float(&xy[0]).ok()?, py_float(&xy[1]).ok()?]),
            _ => None,
        })
        .collect()
}

fn fval(x: f64) -> Value {
    Number::from_f64(x).map_or(Value::Null, Value::Number)
}

/// Python len() of a JSON value and its items as `for p in pts` sees them.
fn py_items(v: &Value) -> Result<Vec<Value>, String> {
    match v {
        Value::Array(a) => Ok(a.clone()),
        Value::String(s) => Ok(s.chars().map(|c| Value::String(c.to_string())).collect()),
        Value::Object(m) => Ok(m.keys().map(|k| Value::String(k.clone())).collect()),
        _ => Err(format!("object of type '{}' has no len()", type_name(v))),
    }
}

/// p[i] as Python evaluates it for a JSON value.
fn py_index(p: &Value, i: usize) -> Result<Value, String> {
    match p {
        Value::Array(a) => a.get(i).cloned().ok_or_else(|| "list index out of range".into()),
        Value::String(s) => s.chars().nth(i).map(|c| Value::String(c.to_string())).ok_or_else(|| "string index out of range".into()),
        Value::Object(_) => Err(i.to_string()), // KeyError(0): JSON keys are strings
        _ => Err(format!("'{}' object is not subscriptable", type_name(p))),
    }
}

impl IgnoreZones {
    /// Startup: moves old zones aside with clear_on_start, loads the file, publishes the zones and
    /// mask. (The node publishes the CostmapFilterInfo itself.)
    pub fn new(params: Params) -> (Self, Vec<Output>) {
        let mut out = Vec::new();
        let mut s = IgnoreZones { params, zones: Vec::new(), draft: Vec::new() };
        let file = s.params.zones_file.clone();
        if s.params.clear_on_start && file.exists() {
            // A new mapping session starts a new map frame; old zones would land in wrong places.
            let bak = PathBuf::from(format!("{}.bak", file.display()));
            match std::fs::rename(&file, &bak) {
                Ok(()) => out.push(Output::Log(Level::Info, format!("New map session: previous zones moved to {}", bak.display()))),
                Err(e) => out.push(Output::Log(Level::Error, format!("Could not move {} aside: {e}", file.display()))),
            }
        }
        s.load(&mut out);
        s.publish_zones(&mut out);
        out.push(Output::Log(Level::Info, format!("{} ignore zone(s) loaded from {}", s.zones.len(), file.display())));
        (s, out)
    }

    fn load(&mut self, out: &mut Vec<Output>) {
        let file = &self.params.zones_file;
        self.zones = match std::fs::read_to_string(file) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => {
                out.push(Output::Log(Level::Error, format!("Could not read {}: {e}", file.display())));
                Vec::new()
            }
            Ok(text) => match serde_json::from_str::<Value>(&text) {
                Ok(Value::Object(data)) => match data.get("zones") {
                    None => Vec::new(),
                    Some(Value::Array(zones)) => zones.iter().filter_map(|z| z.as_object().cloned()).collect(),
                    Some(_) => {
                        out.push(Output::Log(Level::Error, format!("Could not read {}: zones is not a list", file.display())));
                        Vec::new()
                    }
                },
                Ok(_) => {
                    out.push(Output::Log(Level::Error, format!("Could not read {}: not a JSON object", file.display())));
                    Vec::new()
                }
                Err(e) => {
                    out.push(Output::Log(Level::Error, format!("Could not read {}: {e}", file.display())));
                    Vec::new()
                }
            },
        };
    }

    fn save(&self, out: &mut Vec<Output>) {
        let file = &self.params.zones_file;
        let saved: Vec<Value> = self.zones.iter().filter(|z| !z.contains_key("expires")).map(|z| Value::Object(z.clone())).collect();
        let mut doc = Map::new();
        doc.insert("frame".into(), Value::String(self.params.map_frame.clone()));
        doc.insert("zones".into(), Value::Array(saved));
        let tmp = PathBuf::from(format!("{}.tmp", file.display()));
        let result = file
            .parent()
            .map_or(Ok(()), |d| if d.as_os_str().is_empty() { Ok(()) } else { std::fs::create_dir_all(d) })
            .and_then(|_| std::fs::write(&tmp, py_json(&Value::Object(doc), Some(1))))
            .and_then(|_| std::fs::rename(&tmp, file));
        if let Err(e) = result {
            out.push(Output::Log(Level::Error, format!("Could not save {}: {e}", file.display())));
        }
    }

    fn changed(&self, out: &mut Vec<Output>) {
        self.save(out);
        self.publish_zones(out);
        out.push(self.markers());
    }

    fn publish_zones(&self, out: &mut Vec<Output>) {
        let mut doc = Map::new();
        doc.insert("frame".into(), Value::String(self.params.map_frame.clone()));
        doc.insert("zones".into(), Value::Array(self.zones.iter().cloned().map(Value::Object).collect()));
        out.push(Output::Zones(py_json(&Value::Object(doc), None)));
        out.push(Output::Mask(self.mask()));
    }

    /// The keep-out mask; a single free cell when there are no keep-out zones.
    pub fn mask(&self) -> Mask {
        let keep: Vec<&Map<String, Value>> =
            self.zones.iter().filter(|z| z.get("kind").is_some_and(|k| k == "keepout")).collect();
        let polygons: Vec<Vec<[f64; 2]>> = keep.iter().map(|z| points_of(z.get("points"))).collect();
        let margins: Vec<Option<f64>> =
            keep.iter().map(|z| z.get("margin").filter(|m| !m.is_null()).and_then(|m| py_float(m).ok())).collect();
        keepout_mask(&polygons, self.params.keepout_resolution_m, self.params.keepout_margin_m, &margins)
            .unwrap_or(Mask { origin_x: 0.0, origin_y: 0.0, width: 1, height: 1, data: vec![0] })
    }

    /// The 2 s timer: drops temporary zones whose time is up (`now` = wall clock, s).
    pub fn expire(&mut self, now: f64) -> Vec<Output> {
        let mut out = Vec::new();
        let expires = |z: &Map<String, Value>| z.get("expires").map_or(f64::INFINITY, |e| py_float(e).unwrap_or(f64::INFINITY));
        let expired: Vec<String> = self.zones.iter().filter(|z| expires(z) <= now).map(|z| py_str(&name_of(z))).collect();
        if !expired.is_empty() {
            self.zones.retain(|z| expires(z) > now);
            out.push(Output::Log(Level::Info, format!("Expired temporary zone(s): {}", expired.join(", "))));
            self.changed(&mut out);
        }
        out
    }

    fn next_name(&self, kind: &str) -> String {
        let prefix = if kind == "keepout" { "keepout" } else { "zone" };
        (1..).map(|i| format!("{prefix}_{i}")).find(|n| !self.zones.iter().any(|z| z.get("name").is_some_and(|v| v == n))).unwrap_or_default()
    }

    fn close(&mut self, out: &mut Vec<Output>) {
        if self.draft.len() < 3 {
            out.push(Output::Log(Level::Warn, "A zone needs at least 3 corners".into()));
            return;
        }
        let name = self.next_name("exclusion");
        let mut zone = Map::new();
        zone.insert("name".into(), Value::String(name.clone()));
        zone.insert("points".into(), points_value(&self.draft));
        self.zones.push(zone);
        self.draft.clear();
        out.push(Output::Log(Level::Info, format!("Added {name}")));
        self.changed(out);
    }

    /// /clicked_point: adds a corner (map frame) or closes the polygon near the first corner.
    pub fn on_click(&mut self, frame_id: &str, point: [f64; 3], tf: &TfBuffer) -> Vec<Output> {
        let mut out = Vec::new();
        let t = match tf.lookup_transform(&self.params.map_frame, frame_id, 0) {
            Ok(t) => t,
            Err(e) => {
                out.push(Output::Log(Level::Warn, format!("Cannot place point from {frame_id}: {e}")));
                return out;
            }
        };
        let [x, y, _] = t.transform.apply(point);
        if self.draft.len() >= 3 && (x - self.draft[0][0]).hypot(y - self.draft[0][1]) < self.params.close_distance_m {
            self.close(&mut out);
            return out;
        }
        self.draft.push([py_round(x, 3), py_round(y, 3)]);
        out.push(Output::Log(Level::Info, format!("Corner {} at ({x:.2}, {y:.2})", self.draft.len())));
        out.push(self.markers());
        out
    }

    /// ~/command: "close", "undo", "cancel", "clear" or a JSON op from the web app / explorer.
    pub fn on_command(&mut self, data: &str, now: f64) -> Vec<Output> {
        let mut out = Vec::new();
        let text = data.trim();
        if text.starts_with('{') {
            match self.json_command(text, now, &mut out) {
                Ok(()) => self.changed(&mut out),
                Err(e) => out.push(Output::Log(Level::Warn, format!("Rejected zone command: {e}"))),
            }
            return out;
        }
        match text.to_lowercase().as_str() {
            "close" => self.close(&mut out),
            "cancel" => {
                self.draft.clear();
                out.push(self.markers());
            }
            "undo" => {
                if self.draft.pop().is_some() {
                    out.push(self.markers());
                } else if let Some(removed) = self.zones.pop() {
                    out.push(Output::Log(Level::Info, format!("Removed {}", py_str(&name_of(&removed)))));
                    self.changed(&mut out);
                }
            }
            "clear" => {
                self.zones.clear();
                self.draft.clear();
                out.push(Output::Log(Level::Info, "Cleared all zones".into()));
                self.changed(&mut out);
            }
            _ => out.push(Output::Log(Level::Warn, format!("Unknown command \"{data}\" (use close/undo/cancel/clear)"))),
        }
        out
    }

    fn clean_name(name: Option<&Value>) -> Option<String> {
        let name = name.filter(|v| truthy(v))?;
        let kept: String = py_str(name).chars().filter(|c| c.is_alphanumeric() || " _-".contains(*c)).collect();
        let cleaned: String = kept.trim_matches(' ').chars().take(40).collect();
        (!cleaned.is_empty()).then_some(cleaned)
    }

    /// JSON ops: {"op": "add", "points": [[x, y], ...], "name"?, "kind"?, "margin"?, "ttl_s"?},
    /// {"op": "delete", "name"}, {"op": "rename", "name", "new_name"}, {"op": "clear"}.
    fn json_command(&mut self, text: &str, now: f64, out: &mut Vec<Output>) -> Result<(), String> {
        let cmd: Map<String, Value> = match serde_json::from_str::<Value>(text).map_err(|e| e.to_string())? {
            Value::Object(m) => m,
            other => return Err(format!("'{}' object has no attribute 'get'", type_name(&other))),
        };
        let get = |k: &str| cmd.get(k).cloned().unwrap_or(Value::Null);
        match get("op") {
            Value::String(op) if op == "add" => {
                let kind = cmd.get("kind").filter(|v| truthy(v)).cloned().unwrap_or(Value::String("exclusion".into()));
                let kind = match &kind {
                    Value::String(k) if KINDS.contains(&k.as_str()) => k.clone(),
                    _ => return Err("kind must be one of ('exclusion', 'keepout')".into()),
                };
                let items = py_items(cmd.get("points").unwrap_or(&Value::Array(Vec::new())))?;
                if !(3..=MAX_ZONE_POINTS).contains(&items.len()) {
                    return Err(format!("zone needs 3..{MAX_ZONE_POINTS} points"));
                }
                let mut pts = Vec::new();
                for p in &items {
                    let x = py_round(py_float(&py_index(p, 0)?)?, 3);
                    let y = py_round(py_float(&py_index(p, 1)?)?, 3);
                    pts.push([x, y]);
                }
                if !pts.iter().flatten().all(|v| v.is_finite() && v.abs() < 1000.0) {
                    return Err("point coordinates out of range".into());
                }
                if self.zones.len() >= MAX_ZONES {
                    return Err(format!("at most {MAX_ZONES} zones"));
                }
                let name = Self::clean_name(cmd.get("name")).unwrap_or_else(|| self.next_name(&kind));
                if self.zones.iter().any(|z| z.get("name").is_some_and(|v| v == &name)) {
                    return Err(format!("zone \"{name}\" already exists"));
                }
                let mut zone = Map::new();
                zone.insert("name".into(), Value::String(name.clone()));
                zone.insert("points".into(), points_value(&pts));
                zone.insert("kind".into(), Value::String(kind.clone()));
                if let Some(m) = cmd.get("margin").filter(|m| !m.is_null()) {
                    let margin = py_float(m)?;
                    if !(0.0..=1.0).contains(&margin) {
                        return Err("margin must be 0..1 m".into());
                    }
                    zone.insert("margin".into(), fval(py_round(margin, 3)));
                }
                if let Some(t) = cmd.get("ttl_s").filter(|t| !t.is_null()) {
                    let ttl = py_float(t)?;
                    if !(ttl > 0.0 && ttl <= 86400.0) {
                        return Err("ttl_s must be 0..86400".into());
                    }
                    zone.insert("expires".into(), fval(py_round(now + ttl, 1)));
                }
                self.zones.push(zone);
                out.push(Output::Log(Level::Info, format!("Added {kind} zone {name} ({} corners)", pts.len())));
            }
            Value::String(op) if op == "delete" => {
                let name = get("name");
                let before = self.zones.len();
                self.zones.retain(|z| !py_eq(&name_of(z), &name));
                if self.zones.len() == before {
                    return Err(format!("no zone \"{}\"", py_str(&name)));
                }
                out.push(Output::Log(Level::Info, format!("Deleted {}", py_str(&name))));
            }
            Value::String(op) if op == "rename" => {
                let new = Self::clean_name(cmd.get("new_name"));
                let new = match new {
                    Some(n) if !self.zones.iter().any(|z| z.get("name").is_some_and(|v| v == &n)) => n,
                    _ => return Err("invalid or duplicate new_name".into()),
                };
                let old = get("name");
                let zone = self.zones.iter_mut().find(|z| py_eq(&name_of(z), &old));
                match zone {
                    Some(z) => {
                        z.insert("name".into(), Value::String(new));
                    }
                    None => return Err(format!("no zone \"{}\"", py_str(&old))),
                }
            }
            Value::String(op) if op == "clear" => {
                self.zones.clear();
                self.draft.clear();
            }
            op => return Err(format!("unknown op \"{}\"", py_str(&op))),
        }
        Ok(())
    }

    /// The 1 s marker timer: outlines and names of all zones, plus the draft polygon.
    pub fn markers(&self) -> Output {
        let line = |id: i32, pts: &[[f64; 2]], closed: bool, color: [f32; 4], ns: &str| {
            let mut points: Vec<[f64; 3]> = pts.iter().map(|p| [p[0], p[1], 0.02]).collect();
            if closed && !pts.is_empty() {
                points.push([pts[0][0], pts[0][1], 0.02]);
            }
            MarkerSpec {
                ns: ns.into(),
                id,
                kind: LINE_STRIP,
                action: ADD,
                position: [0.0; 3],
                orientation_w: 1.0,
                scale: [0.04, 0.0, 0.0],
                color,
                points,
                text: String::new(),
            }
        };
        let mut markers = Vec::new();
        for (i, z) in self.zones.iter().enumerate() {
            let pts = points_of(z.get("points"));
            let keepout = z.get("kind").is_some_and(|k| k == "keepout");
            let color = if keepout { [0.9, 0.15, 0.15, 1.0] } else { [1.0, 0.45, 0.0, 1.0] };
            markers.push(line(i as i32, &pts, true, color, "zones"));
            let raw = match z.get("points") {
                Some(Value::Array(a)) => a.clone(),
                _ => Vec::new(),
            };
            let coord = |k: usize| py_sum(raw.iter().filter_map(|p| p.as_array().and_then(|xy| xy.get(k)).cloned())) / raw.len() as f64;
            markers.push(MarkerSpec {
                ns: "zone_names".into(),
                id: i as i32,
                kind: TEXT_VIEW_FACING,
                action: ADD,
                position: [coord(0), coord(1), 0.15],
                orientation_w: 1.0,
                scale: [0.0, 0.0, 0.2],
                color: [1.0, 0.6, 0.2, 1.0],
                points: Vec::new(),
                text: match z.get("name") {
                    Some(Value::String(s)) => s.clone(),
                    other => py_str(other.unwrap_or(&Value::Null)),
                },
            });
        }
        if !self.draft.is_empty() {
            markers.push(line(0, &self.draft, false, [1.0, 1.0, 0.0, 1.0], "draft"));
            markers.push(MarkerSpec {
                ns: "draft_corners".into(),
                id: 0,
                kind: SPHERE_LIST,
                action: ADD,
                position: [0.0; 3],
                orientation_w: 1.0,
                scale: [0.08, 0.08, 0.08],
                color: [1.0, 1.0, 0.0, 1.0],
                points: self.draft.iter().map(|p| [p[0], p[1], 0.02]).collect(),
                text: String::new(),
            });
        }
        Output::Markers(markers)
    }

    pub fn zones_file(&self) -> &Path {
        &self.params.zones_file
    }
}

fn points_value(pts: &[[f64; 2]]) -> Value {
    Value::Array(pts.iter().map(|p| Value::Array(vec![fval(p[0]), fval(p[1])])).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_float_formatting() {
        for (x, s) in [(0.0, "0.0"), (-0.0, "-0.0"), (1.5, "1.5"), (0.1, "0.1"), (1e-4, "0.0001"), (1e-5, "1e-05"),
            (123456789.0, "123456789.0"), (1e16, "1e+16"), (1e15, "1000000000000000.0"), (-2.5e-7, "-2.5e-07"),
            (1794730509.6, "1794730509.6")]
        {
            assert_eq!(py_repr(x), s);
        }
        assert_eq!(py_round(0.125, 2), 0.12);
        assert_eq!(py_round(2.675, 2), 2.67); // 2.675 is just below the half
        assert_eq!(py_round(-1.0005, 3), -1.0);
    }

    #[test]
    fn json_like_python() {
        let v: Value = serde_json::from_str(r#"{"frame": "map", "zones": [{"name": "t\u00e9", "points": [[1.0, 2], []]}]}"#).unwrap();
        assert_eq!(py_json(&v, None), r#"{"frame": "map", "zones": [{"name": "t\u00e9", "points": [[1.0, 2], []]}]}"#);
        assert_eq!(
            py_json(&v, Some(1)),
            "{\n \"frame\": \"map\",\n \"zones\": [\n  {\n   \"name\": \"t\\u00e9\",\n   \"points\": [\n    [\n     1.0,\n     2\n    ],\n    []\n   ]\n  }\n ]\n}"
        );
    }

    #[test]
    fn mask_covers_polygon_and_margin() {
        let square = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let m = keepout_mask(&[square], 0.05, 0.2, &[None]).unwrap();
        let at = |x: f64, y: f64| {
            let (c, r) = (((x - m.origin_x) / 0.05) as usize, ((y - m.origin_y) / 0.05) as usize);
            m.data[r * m.width + c]
        };
        assert_eq!((at(0.5, 0.5), at(1.15, 0.5), at(1.3, 0.5)), (100, 100, 0));
        assert!(keepout_mask(&[], 0.05, 0.2, &[]).is_none());
    }

    fn zones(dir: &str) -> IgnoreZones {
        let file = std::env::temp_dir().join(format!("roomba_rs_zones_{dir}_{}", std::process::id())).join("z.json");
        let _ = std::fs::remove_file(&file);
        let params = Params {
            zones_file: file,
            map_frame: "map".into(),
            close_distance_m: 0.25,
            keepout_margin_m: 0.2,
            keepout_resolution_m: 0.05,
            clear_on_start: false,
        };
        IgnoreZones::new(params).0
    }

    #[test]
    fn commands_and_expiry() {
        let mut z = zones("cmd");
        let out = z.on_command(r#"{"op": "add", "kind": "keepout", "name": "bump_1!", "points": [[0, 0], [1, 0], ["1", 1]], "margin": 0.15, "ttl_s": 600}"#, 100.0);
        assert!(out.contains(&Output::Log(Level::Info, "Added keepout zone bump_1 (3 corners)".into())), "{out:?}");
        assert_eq!(z.zones[0]["expires"], serde_json::json!(700.0));
        let saved = std::fs::read_to_string(z.zones_file()).unwrap();
        assert!(saved.contains("\"zones\": []"), "temporary zones are not saved: {saved}");
        let rejected = z.on_command(r#"{"op": "add", "points": [[0, 0], [1, 0]]}"#, 100.0);
        assert_eq!(rejected, vec![Output::Log(Level::Warn, "Rejected zone command: zone needs 3..64 points".into())]);
        assert!(z.expire(699.9).is_empty());
        let out = z.expire(700.0);
        assert_eq!(out[0], Output::Log(Level::Info, "Expired temporary zone(s): bump_1".into()));
        assert!(z.zones.is_empty());
        let _ = std::fs::remove_dir_all(z.zones_file().parent().unwrap());
    }

    #[test]
    fn clicks_build_a_polygon() {
        let mut z = zones("click");
        let tf = TfBuffer::new();
        for p in [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.1, 0.1, 0.0]] {
            z.on_click("map", p, &tf);
        }
        assert_eq!(z.zones.len(), 1);
        assert_eq!(z.zones[0]["name"], "zone_1");
        let out = z.on_click("camera", [0.0; 3], &tf);
        assert!(matches!(&out[0], Output::Log(Level::Warn, m) if m.starts_with("Cannot place point from camera")));
        let _ = std::fs::remove_dir_all(z.zones_file().parent().unwrap());
    }
}
