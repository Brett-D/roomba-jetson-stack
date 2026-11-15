//! The parts of Python the web server's JSON goes through, reproduced so the Rust server answers
//! byte for byte like roomba_web/server.py: json.loads (NaN/Infinity, arbitrary-size ints,
//! duplicate keys: last value at the first key's position), json.dumps (", " / ": " separators,
//! ensure_ascii, float repr), float(), truthiness, ==, str() and list.sort's comparison order.
//!
//! Not reproduced: lone UTF-16 surrogates in strings (Python keeps them, Rust strings cannot hold
//! them; they become U+FFFD).

use std::fmt::Write as _;

use crate::zones::py_repr;

/// A value as Python's json.loads returns it (dicts keep their key order).
#[derive(Clone, Debug, PartialEq)]
pub enum Py {
    None,
    Bool(bool),
    Int(i64),
    /// An int outside i64, as its canonical decimal digits (with a leading '-' if negative).
    BigInt(String),
    Float(f64),
    Str(String),
    List(Vec<Py>),
    Dict(Vec<(String, Py)>),
}

/// Python's exception classes as far as the handlers tell them apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyErr {
    Value,
    Type,
    Key,
    Index,
    Overflow,
}

impl Py {
    pub fn str(s: &str) -> Py {
        Py::Str(s.to_string())
    }

    pub fn float_list(points: &[[f64; 2]]) -> Py {
        Py::List(points.iter().map(|p| Py::List(vec![Py::Float(p[0]), Py::Float(p[1])])).collect())
    }

    /// d[key] for a str key: KeyError if missing, TypeError for non-dicts.
    pub fn get(&self, key: &str) -> Result<&Py, PyErr> {
        match self {
            Py::Dict(items) => items.iter().find(|(k, _)| k == key).map(|(_, v)| v).ok_or(PyErr::Key),
            _ => Err(PyErr::Type),
        }
    }

    /// d.get(key) for a dict (None if missing).
    pub fn get_or_none(&self, key: &str) -> Py {
        self.get(key).cloned().unwrap_or(Py::None)
    }

    /// v[i] for a non-negative int index: lists and strs (IndexError past the end); dicts look up
    /// the int key, which JSON objects never have (KeyError); anything else is a TypeError.
    pub fn index(&self, i: usize) -> Result<Py, PyErr> {
        match self {
            Py::List(items) => items.get(i).cloned().ok_or(PyErr::Index),
            Py::Str(s) => s.chars().nth(i).map(|c| Py::Str(c.to_string())).ok_or(PyErr::Index),
            Py::Dict(_) => Err(PyErr::Key),
            _ => Err(PyErr::Type),
        }
    }

    /// iter(v): list items, dict keys or the characters of a str.
    pub fn iter(&self) -> Result<Vec<Py>, PyErr> {
        match self {
            Py::List(items) => Ok(items.clone()),
            Py::Dict(items) => Ok(items.iter().map(|(k, _)| Py::Str(k.clone())).collect()),
            Py::Str(s) => Ok(s.chars().map(|c| Py::Str(c.to_string())).collect()),
            _ => Err(PyErr::Type),
        }
    }

    /// bool(v)
    pub fn truthy(&self) -> bool {
        match self {
            Py::None => false,
            Py::Bool(b) => *b,
            Py::Int(i) => *i != 0,
            Py::BigInt(_) => true,
            Py::Float(f) => *f != 0.0,
            Py::Str(s) => !s.is_empty(),
            Py::List(items) => !items.is_empty(),
            Py::Dict(items) => !items.is_empty(),
        }
    }

    /// Whether `v` can be a dict key (lists and dicts are unhashable).
    pub fn hashable(&self) -> bool {
        !matches!(self, Py::List(_) | Py::Dict(_))
    }
}

// ---------------------------------------------------------------- json.loads

/// Python 3.12's limit on int string conversion (sys.int_info.default_max_str_digits).
const MAX_INT_DIGITS: usize = 4300;

/// json.loads(text); Err is the reason (a JSONDecodeError or ValueError in Python).
pub fn loads(text: &str) -> Result<Py, String> {
    let mut p = Parser { s: text.as_bytes(), text, i: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.s.len() {
        return Err(format!("Extra data at {}", p.i));
    }
    Ok(v)
}

struct Parser<'a> {
    s: &'a [u8],
    text: &'a str,
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn starts(&self, lit: &str) -> bool {
        self.s[self.i..].starts_with(lit.as_bytes())
    }

    fn value(&mut self) -> Result<Py, String> {
        let expecting = || "Expecting value".to_string();
        match self.peek().ok_or_else(expecting)? {
            b'"' => {
                self.i += 1;
                Ok(Py::Str(self.string()?))
            }
            b'{' => self.object(),
            b'[' => self.array(),
            b'n' if self.starts("null") => {
                self.i += 4;
                Ok(Py::None)
            }
            b't' if self.starts("true") => {
                self.i += 4;
                Ok(Py::Bool(true))
            }
            b'f' if self.starts("false") => {
                self.i += 5;
                Ok(Py::Bool(false))
            }
            b'N' if self.starts("NaN") => {
                self.i += 3;
                Ok(Py::Float(f64::NAN))
            }
            b'I' if self.starts("Infinity") => {
                self.i += 8;
                Ok(Py::Float(f64::INFINITY))
            }
            b'-' if self.starts("-Infinity") => {
                self.i += 9;
                Ok(Py::Float(f64::NEG_INFINITY))
            }
            b'-' | b'0'..=b'9' => self.number(),
            _ => Err(expecting()),
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.i;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.i += 1;
        }
        self.i - start
    }

    fn number(&mut self) -> Result<Py, String> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        match self.peek() {
            Some(b'1'..=b'9') => {
                self.digits();
            }
            Some(b'0') => self.i += 1,
            _ => return Err("Expecting value".into()),
        }
        let mut is_float = false;
        if self.peek() == Some(b'.') && self.s.get(self.i + 1).is_some_and(|c| c.is_ascii_digit()) {
            self.i += 1;
            self.digits();
            is_float = true;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            let save = self.i;
            self.i += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if self.digits() == 0 {
                self.i = save;
            } else {
                is_float = true;
            }
        }
        let num = &self.text[start..self.i];
        if is_float {
            return num.parse::<f64>().map(Py::Float).map_err(|e| e.to_string());
        }
        if let Ok(i) = num.parse::<i64>() {
            return Ok(Py::Int(i));
        }
        let digits = num.trim_start_matches('-').len();
        if digits > MAX_INT_DIGITS {
            return Err(format!("Exceeds the limit ({MAX_INT_DIGITS} digits) for integer string conversion"));
        }
        Ok(Py::BigInt(num.to_string()))
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let h = self.text.get(self.i..self.i + 4).filter(|h| h.bytes().all(|c| c.is_ascii_hexdigit()));
        let h = h.ok_or_else(|| "Invalid \\uXXXX escape".to_string())?;
        self.i += 4;
        Ok(u32::from_str_radix(h, 16).unwrap_or(0))
    }

    /// The rest of a string after its opening quote (strict: no raw control characters).
    fn string(&mut self) -> Result<String, String> {
        let mut out = String::new();
        loop {
            let start = self.i;
            while self.peek().is_some_and(|c| c != b'"' && c != b'\\' && c >= 0x20) {
                self.i += 1;
            }
            out.push_str(&self.text[start..self.i]);
            match self.peek() {
                None => return Err("Unterminated string".into()),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    let c = self.peek().ok_or("Unterminated string")?;
                    self.i += 1;
                    match c {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let mut u = self.hex4()?;
                            if (0xd800..0xdc00).contains(&u) && self.starts("\\u") {
                                let save = self.i;
                                self.i += 2;
                                let lo = self.hex4()?;
                                if (0xdc00..0xe000).contains(&lo) {
                                    u = 0x10000 + ((u - 0xd800) << 10) + (lo - 0xdc00);
                                } else {
                                    self.i = save;
                                }
                            }
                            out.push(char::from_u32(u).unwrap_or('\u{fffd}'));
                        }
                        _ => return Err("Invalid \\escape".into()),
                    }
                }
                Some(_) => return Err("Invalid control character".into()),
            }
        }
    }

    fn object(&mut self) -> Result<Py, String> {
        self.i += 1;
        let mut items: Vec<(String, Py)> = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Py::Dict(items));
        }
        loop {
            if self.peek() != Some(b'"') {
                return Err("Expecting property name enclosed in double quotes".into());
            }
            self.i += 1;
            let key = self.string()?;
            self.ws();
            if self.peek() != Some(b':') {
                return Err("Expecting ':' delimiter".into());
            }
            self.i += 1;
            self.ws();
            let v = self.value()?;
            match items.iter_mut().find(|(k, _)| *k == key) {
                Some(slot) => slot.1 = v,
                None => items.push((key, v)),
            }
            self.ws();
            match self.peek() {
                Some(b',') => {
                    self.i += 1;
                    self.ws();
                }
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Py::Dict(items));
                }
                _ => return Err("Expecting ',' delimiter".into()),
            }
        }
    }

    fn array(&mut self) -> Result<Py, String> {
        self.i += 1;
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Py::List(items));
        }
        loop {
            items.push(self.value()?);
            self.ws();
            match self.peek() {
                Some(b',') => {
                    self.i += 1;
                    self.ws();
                }
                Some(b']') => {
                    self.i += 1;
                    return Ok(Py::List(items));
                }
                _ => return Err("Expecting ',' delimiter".into()),
            }
        }
    }
}

// ---------------------------------------------------------------- json.dumps

fn float_json(out: &mut String, x: f64) {
    if x.is_nan() {
        out.push_str("NaN");
    } else if x.is_infinite() {
        out.push_str(if x > 0.0 { "Infinity" } else { "-Infinity" });
    } else {
        out.push_str(&py_repr(x));
    }
}

fn str_json(out: &mut String, s: &str) {
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

fn dump_into(out: &mut String, v: &Py) {
    match v {
        Py::None => out.push_str("null"),
        Py::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Py::Int(i) => {
            let _ = write!(out, "{i}");
        }
        Py::BigInt(d) => out.push_str(d),
        Py::Float(f) => float_json(out, *f),
        Py::Str(s) => str_json(out, s),
        Py::List(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                dump_into(out, item);
            }
            out.push(']');
        }
        Py::Dict(items) => {
            out.push('{');
            for (i, (k, item)) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                str_json(out, k);
                out.push_str(": ");
                dump_into(out, item);
            }
            out.push('}');
        }
    }
}

/// json.dumps(v)
pub fn dumps(v: &Py) -> String {
    let mut out = String::new();
    dump_into(&mut out, v);
    out
}

// ---------------------------------------------------------------- float(), ==, str(), sort

/// float(text) for a str: surrounding whitespace, PEP 515 underscores, inf/infinity/nan.
fn parse_float_str(s: &str) -> Option<f64> {
    let t = s.trim();
    let b = t.as_bytes();
    for (i, &c) in b.iter().enumerate() {
        if c == b'_' && !(i > 0 && i + 1 < b.len() && b[i - 1].is_ascii_digit() && b[i + 1].is_ascii_digit()) {
            return None;
        }
    }
    let t = t.replace('_', "");
    let body = t.strip_prefix(['+', '-']).unwrap_or(&t).to_ascii_lowercase();
    if matches!(body.as_str(), "inf" | "infinity" | "nan") {
        let v = if body == "nan" { f64::NAN } else { f64::INFINITY };
        return Some(if t.starts_with('-') { -v } else { v });
    }
    // Python's grammar: digits with an optional '.', optional exponent; at least one digit.
    let (mant, exp) = match body.find('e') {
        Some(k) => (&body[..k], Some(&body[k + 1..])),
        None => (&body[..], None),
    };
    let (int_part, frac) = mant.split_once('.').unwrap_or((mant, ""));
    let digits_ok = |d: &str| d.bytes().all(|c| c.is_ascii_digit());
    if !(digits_ok(int_part) && digits_ok(frac)) || int_part.len() + frac.len() == 0 {
        return None;
    }
    if let Some(e) = exp {
        let e = e.strip_prefix(['+', '-']).unwrap_or(e);
        if e.is_empty() || !digits_ok(e) {
            return None;
        }
    }
    t.parse().ok()
}

/// float(v)
pub fn py_float(v: &Py) -> Result<f64, PyErr> {
    match v {
        Py::Int(i) => Ok(*i as f64),
        Py::BigInt(d) => match d.parse::<f64>() {
            Ok(f) if f.is_finite() => Ok(f),
            _ => Err(PyErr::Overflow),
        },
        Py::Float(f) => Ok(*f),
        Py::Bool(b) => Ok(*b as u8 as f64),
        Py::Str(s) => parse_float_str(s).ok_or(PyErr::Value),
        _ => Err(PyErr::Type),
    }
}

/// Exact comparison of an int (as decimal digits) with a float, as Python does.
fn int_digits_eq_float(digits: &str, f: f64) -> bool {
    if !f.is_finite() || f.fract() != 0.0 {
        return false;
    }
    let s = format!("{f:.0}");
    let s = if s == "-0" { "0".to_string() } else { s };
    s == digits
}

/// a == b
pub fn py_eq(a: &Py, b: &Py) -> bool {
    use Py::*;
    let int = |v: &Py| match v {
        Int(i) => Some(*i as i128),
        Bool(b) => Some(*b as i128),
        _ => Option::None,
    };
    match (a, b) {
        (None, None) => true,
        (Str(x), Str(y)) => x == y,
        (List(x), List(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| py_eq(p, q)),
        (Dict(x), Dict(y)) => x.len() == y.len() && x.iter().all(|(k, v)| y.iter().any(|(k2, w)| k == k2 && py_eq(v, w))),
        (Float(x), Float(y)) => x == y,
        (BigInt(x), BigInt(y)) => x == y,
        (BigInt(d), Float(f)) | (Float(f), BigInt(d)) => int_digits_eq_float(d, *f),
        (BigInt(_), _) | (_, BigInt(_)) => false,
        (Float(f), o) | (o, Float(f)) => match int(o) {
            Some(i) => int_digits_eq_float(&i.to_string(), *f),
            Option::None => false,
        },
        _ => match (int(a), int(b)) {
            (Some(x), Some(y)) => x == y,
            _ => false,
        },
    }
}

/// repr(str): the quote Python picks, backslash escapes, \x/\u/\U for non-printable characters
/// (approximated: C0/C1 controls, soft hyphen, zero-width and line/paragraph separators, BOM,
/// private use).
pub fn repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::from(quote);
    for c in s.chars() {
        let u = c as u32;
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            _ if u < 0x20 || (0x7f..=0xa0).contains(&u) || u == 0xad => {
                let _ = write!(out, "\\x{u:02x}");
            }
            _ if (0x2028..=0x202f).contains(&u)
                || (0x200b..=0x200f).contains(&u)
                || (0x2060..=0x206f).contains(&u)
                || u == 0xfeff
                || (0xe000..=0xf8ff).contains(&u)
                || (0xfff9..=0xfffb).contains(&u) =>
            {
                let _ = write!(out, "\\u{u:04x}");
            }
            _ if u >= 0xf0000 => {
                let _ = write!(out, "\\U{u:08x}");
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

fn py_repr_value(v: &Py) -> String {
    match v {
        Py::Str(s) => repr_str(s),
        _ => py_str(v),
    }
}

/// str(v)
pub fn py_str(v: &Py) -> String {
    match v {
        Py::None => "None".into(),
        Py::Bool(b) => if *b { "True" } else { "False" }.into(),
        Py::Int(i) => i.to_string(),
        Py::BigInt(d) => d.clone(),
        Py::Float(f) => py_repr(*f),
        Py::Str(s) => s.clone(),
        Py::List(items) => format!("[{}]", items.iter().map(py_repr_value).collect::<Vec<_>>().join(", ")),
        Py::Dict(items) => format!(
            "{{{}}}",
            items.iter().map(|(k, v)| format!("{}: {}", repr_str(k), py_repr_value(v))).collect::<Vec<_>>().join(", ")
        ),
    }
}

/// sorted(items, key=key) with float keys compared by `<` (so NaNs land where CPython 3.12's
/// list.sort puts them): the initial run, then binary insertion - CPython's algorithm for lists
/// shorter than 64, which detection lists always are. Stable.
pub fn py_sort_by_key<T>(items: Vec<T>, key: impl Fn(&T) -> f64) -> Vec<T> {
    let mut v: Vec<(f64, T)> = items.into_iter().map(|t| (key(&t), t)).collect();
    let n = v.len();
    if n < 2 {
        return v.into_iter().map(|(_, t)| t).collect();
    }
    // count_run (3.12): a strictly descending run is reversed, otherwise take the non-descending run
    let mut run = 2;
    if v[1].0 < v[0].0 {
        while run < n && v[run].0 < v[run - 1].0 {
            run += 1;
        }
        v[..run].reverse();
    } else {
        while run < n && !(v[run].0 < v[run - 1].0) {
            run += 1;
        }
    }
    for start in run..n {
        let pivot = v[start].0;
        let (mut l, mut r) = (0, start);
        while l < r {
            let p = l + ((r - l) >> 1);
            if pivot < v[p].0 {
                r = p;
            } else {
                l = p + 1;
            }
        }
        let item = v.remove(start);
        v.insert(l, item);
    }
    v.into_iter().map(|(_, t)| t).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_dumps_round_trip_like_python() {
        let cases = [
            (r#"{"a": 1, "b": [1.5, -0.0, 1e16, 1e-05, 0.1], "c": null}"#, r#"{"a": 1, "b": [1.5, -0.0, 1e+16, 1e-05, 0.1], "c": null}"#),
            (r#"[NaN, Infinity, -Infinity, 1E400]"#, "[NaN, Infinity, -Infinity, Infinity]"),
            (r#" {"x":"\u00e9\ud83d\ude00\n\"", "x" : 2 , "y":true} "#, r#"{"x": 2, "y": true}"#),
            (r#"{"k": "é↑", "n": 123456789012345678901234567890}"#, r#"{"k": "\u00e9\u2191", "n": 123456789012345678901234567890}"#),
            ("[1.0, 100, -7, 0.5e1, 3E-2]", "[1.0, 100, -7, 5.0, 0.03]"),
            ("\"\\u007f\\/\"", "\"\\u007f/\""),
        ];
        for (text, want) in cases {
            assert_eq!(dumps(&loads(text).unwrap()), want, "{text}");
        }
        for bad in ["", "1.", "01", "[1,]", "{\"a\" 1}", "\"\t\"", "-NaN", "1 2", "{'a': 1}", "[1e]", "tru"] {
            assert!(loads(bad).is_err(), "{bad}");
        }
        assert!(loads(&"9".repeat(4301)).is_err());
        assert!(matches!(loads(&"9".repeat(4300)), Ok(Py::BigInt(_))));
    }

    #[test]
    fn python_float_conversion() {
        let f = |v: Py| py_float(&v);
        assert_eq!(f(Py::str(" 1_000.5 ")), Ok(1000.5));
        assert_eq!(f(Py::str("-inf")), Ok(f64::NEG_INFINITY));
        assert!(f(Py::str("nan")).unwrap().is_nan());
        assert_eq!(f(Py::str("1__0")), Err(PyErr::Value));
        assert_eq!(f(Py::str("0x10")), Err(PyErr::Value));
        assert_eq!(f(Py::str(".5")), Ok(0.5));
        assert_eq!(f(Py::str("5.")), Ok(5.0));
        assert_eq!(f(Py::str(".")), Err(PyErr::Value));
        assert_eq!(f(Py::Bool(true)), Ok(1.0));
        assert_eq!(f(Py::None), Err(PyErr::Type));
        assert_eq!(f(Py::List(vec![])), Err(PyErr::Type));
        assert_eq!(f(Py::BigInt("1".repeat(400))), Err(PyErr::Overflow));
    }

    #[test]
    fn python_equality_and_str() {
        assert!(py_eq(&Py::Int(1), &Py::Float(1.0)));
        assert!(py_eq(&Py::Bool(true), &Py::Int(1)));
        assert!(!py_eq(&Py::Float(f64::NAN), &Py::Float(f64::NAN)));
        assert!(py_eq(&Py::BigInt("100000000000000000000".into()), &Py::Float(1e20)));
        assert!(!py_eq(&Py::str("1"), &Py::Int(1)));
        let d = loads(r#"{"a": [1, "x'"], "b": null}"#).unwrap();
        assert_eq!(py_str(&d), r#"{'a': [1, "x'"], 'b': None}"#);
        assert_eq!(py_str(&Py::Float(2.0)), "2.0");
    }

    #[test]
    fn sort_matches_cpython_with_nan() {
        let n = f64::NAN;
        // Expected orders from CPython 3.12: sorted(v, key=lambda x: x)
        let s = |v: Vec<f64>| py_sort_by_key(v, |x| *x).iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(",");
        assert_eq!(s(vec![3.0, n, 1.0, 2.0]), "3,NaN,1,2");
        assert_eq!(s(vec![2.0, 1.0, n, 0.5]), "0.5,1,2,NaN");
        assert_eq!(s(vec![n, 3.0, 2.0, 1.0]), "NaN,1,2,3");
        assert_eq!(s(vec![1.0, n, 0.5, n, 0.2]), "1,NaN,0.5,NaN,0.2");
        assert_eq!(s(vec![0.5, 0.2, 0.9, 0.2]), "0.2,0.2,0.5,0.9");
        let stable = py_sort_by_key(vec![(1.0, 'a'), (0.0, 'b'), (1.0, 'c')], |x| x.0);
        assert_eq!(stable.iter().map(|x| x.1).collect::<String>(), "bac");
    }
}
