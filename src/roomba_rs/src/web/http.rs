//! The aiohttp 3.9 / yarl 1.8 behaviour roomba_web/server.py relies on, for the Rust web server:
//! request path decoding, the static file route (no index, no escape from the www directory,
//! symlinks resolved like pathlib), FileResponse (ETag, Last-Modified, conditional and range
//! requests, .gz siblings), Python's mimetypes, error bodies, cookies and form decoding.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub const TEXT_PLAIN: &str = "text/plain; charset=utf-8";
pub const APPLICATION_JSON: &str = "application/json; charset=utf-8";

/// A response before it goes to hyper.
#[derive(Clone, Debug, PartialEq)]
pub struct Reply {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

/// Python's http.HTTPStatus phrases (aiohttp's reason and default error text).
pub fn reason(status: u16) -> &'static str {
    match status {
        101 => "Switching Protocols",
        200 => "OK",
        206 => "Partial Content",
        302 => "Found",
        304 => "Not Modified",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
        412 => "Precondition Failed",
        413 => "Request Entity Too Large",
        416 => "Requested Range Not Satisfiable",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "",
    }
}

impl Reply {
    pub fn new(status: u16, content_type: &str, body: Vec<u8>) -> Reply {
        Reply { status, headers: vec![("Content-Type".into(), content_type.into())], body }
    }

    /// web.HTTPException(text=...)
    pub fn text(status: u16, text: &str) -> Reply {
        Reply::new(status, TEXT_PLAIN, text.as_bytes().to_vec())
    }

    /// web.HTTPException without text: "<status>: <reason>".
    pub fn error(status: u16) -> Reply {
        Reply::text(status, &format!("{status}: {}", reason(status)))
    }

    /// web.json_response
    pub fn json(text: String) -> Reply {
        Reply::new(200, APPLICATION_JSON, text.into_bytes())
    }

    /// web.HTTPFound(location)
    pub fn found(location: &str) -> Reply {
        Reply::error(302).with("Location", location)
    }

    /// aiohttp's response to an unhandled exception in a handler.
    pub fn internal_error() -> Reply {
        Reply::text(500, "500 Internal Server Error\n\nServer got itself in trouble")
    }

    pub fn with(mut self, name: &str, value: &str) -> Reply {
        self.set(name, value);
        self
    }

    /// Sets a header, replacing any of the same name.
    pub fn set(&mut self, name: &str, value: &str) {
        self.headers.retain(|(n, _)| !n.eq_ignore_ascii_case(name));
        self.headers.push((name.into(), value.into()));
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(n, _)| n.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

// ---------------------------------------------------------------- yarl / urllib decoding

/// One step of Python's strict incremental UTF-8 decoder: Ok(Some(c)) when `pending` + `b`
/// completes a character, Ok(None) when more bytes are needed, Err when `b` makes it invalid.
fn utf8_feed(pending: &mut Vec<u8>, b: u8) -> Result<Option<char>, ()> {
    if pending.is_empty() {
        return match b {
            0x00..=0x7f => Ok(Some(b as char)),
            0xc2..=0xf4 => {
                pending.push(b);
                Ok(None)
            }
            _ => Err(()),
        };
    }
    let lead = pending[0];
    let ok = if pending.len() == 1 {
        match lead {
            0xe0 => (0xa0..=0xbf).contains(&b),
            0xed => (0x80..=0x9f).contains(&b),
            0xf0 => (0x90..=0xbf).contains(&b),
            0xf4 => (0x80..=0x8f).contains(&b),
            _ => (0x80..=0xbf).contains(&b),
        }
    } else {
        (0x80..=0xbf).contains(&b)
    };
    if !ok {
        return Err(());
    }
    pending.push(b);
    let need = match lead {
        0xc2..=0xdf => 2,
        0xe0..=0xef => 3,
        _ => 4,
    };
    if pending.len() < need {
        return Ok(None);
    }
    let c = std::str::from_utf8(pending).ok().and_then(|s| s.chars().next());
    pending.clear();
    Ok(c)
}

/// yarl's path unquoting (URL.path, aiohttp's request.path and route match values): %XX escapes
/// forming valid UTF-8 are decoded (including %2F), anything else is kept as written.
pub fn unquote_path(raw: &str) -> String {
    let b = raw.as_bytes();
    let mut out = String::with_capacity(raw.len());
    let mut pending: Vec<u8> = Vec::new();
    let mut pending_start = 0;
    let hex = |c: u8| (c as char).to_digit(16);
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 3 <= b.len() {
            if let (Some(h), Some(l)) = (hex(b[i + 1]), hex(b[i + 2])) {
                let byte = (h * 16 + l) as u8;
                if pending.is_empty() {
                    pending_start = i;
                }
                match utf8_feed(&mut pending, byte) {
                    Ok(Some(c)) => out.push(c),
                    Ok(None) => {}
                    Err(()) => {
                        // the buffered escapes stay as written; retry this byte on its own
                        out.push_str(&raw[pending_start..i]);
                        pending.clear();
                        pending_start = i;
                        match utf8_feed(&mut pending, byte) {
                            Ok(Some(c)) => out.push(c),
                            Ok(None) => {}
                            Err(()) => out.push_str(&raw[i..i + 3]),
                        }
                    }
                }
                i += 3;
                continue;
            }
        }
        if !pending.is_empty() {
            out.push_str(&raw[pending_start..i]);
            pending.clear();
        }
        let ch = raw[i..].chars().next().unwrap_or('\u{fffd}');
        out.push(ch);
        i += ch.len_utf8();
    }
    if !pending.is_empty() {
        out.push_str(&raw[pending_start..]);
    }
    out
}

/// urllib.parse.unquote(s, errors='replace') after '+' -> ' ' (parse_qsl's name/value decoding).
fn unquote_plus(s: &str) -> String {
    let s = s.replace('+', " ");
    if !s.contains('%') {
        return s;
    }
    let mut out = String::new();
    let mut run = String::new();
    let flush = |run: &mut String, out: &mut String| {
        if run.is_empty() {
            return;
        }
        let parts: Vec<&str> = run.split('%').collect();
        let mut bytes = parts[0].as_bytes().to_vec();
        for part in &parts[1..] {
            let pb = part.as_bytes();
            match (pb.first().and_then(|c| (*c as char).to_digit(16)), pb.get(1).and_then(|c| (*c as char).to_digit(16))) {
                (Some(h), Some(l)) => {
                    bytes.push((h * 16 + l) as u8);
                    bytes.extend_from_slice(&pb[2..]);
                }
                _ => {
                    bytes.push(b'%');
                    bytes.extend_from_slice(pb);
                }
            }
        }
        out.push_str(&String::from_utf8_lossy(&bytes));
        run.clear();
    };
    for c in s.chars() {
        if c.is_ascii() {
            run.push(c);
        } else {
            flush(&mut run, &mut out);
            out.push(c);
        }
    }
    flush(&mut run, &mut out);
    out
}

/// urllib.parse.parse_qsl(text, keep_blank_values=True)
pub fn parse_qsl(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for item in text.split('&') {
        if item.is_empty() {
            continue;
        }
        let (name, value) = item.split_once('=').unwrap_or((item, ""));
        out.push((unquote_plus(name), unquote_plus(value)));
    }
    out
}

/// The main type of a Content-Type header as aiohttp's request.content_type reports it.
pub fn content_type(header: Option<&str>) -> String {
    match header {
        None => "application/octet-stream".into(),
        Some(h) => {
            let main = h.split(';').next().unwrap_or("").trim().to_ascii_lowercase();
            if main.matches('/').count() == 1 && !main.starts_with('/') && !main.ends_with('/') {
                main
            } else {
                "text/plain".into()
            }
        }
    }
}

/// The charset parameter of a Content-Type header.
pub fn charset(header: Option<&str>) -> Option<String> {
    header?.split(';').skip(1).find_map(|p| {
        let (k, v) = p.split_once('=')?;
        k.trim().eq_ignore_ascii_case("charset").then(|| v.trim().trim_matches('"').to_string())
    })
}

/// bytes.decode(charset) for the charsets clients send. Err(true): unknown charset (LookupError
/// in Python); Err(false): invalid data (UnicodeDecodeError, a ValueError).
pub fn decode(body: &[u8], charset: Option<&str>) -> Result<String, bool> {
    let cs = charset.unwrap_or("utf-8").to_ascii_lowercase().replace('_', "-");
    match cs.as_str() {
        "utf-8" | "utf8" | "u8" | "utf" => String::from_utf8(body.to_vec()).map_err(|_| false),
        "ascii" | "us-ascii" | "646" => {
            if body.is_ascii() {
                Ok(String::from_utf8_lossy(body).into_owned())
            } else {
                Err(false)
            }
        }
        "latin-1" | "latin1" | "iso-8859-1" | "iso8859-1" | "l1" => Ok(body.iter().map(|&b| b as char).collect()),
        _ => Err(true),
    }
}

/// The value of one cookie in a Cookie header (the last one of that name, quotes removed).
pub fn cookie(header: Option<&str>, name: &str) -> Option<String> {
    let mut found = None;
    for part in header?.split(';') {
        if let Some((k, v)) = part.split_once('=') {
            if k.trim() == name {
                let v = v.trim();
                let v = if v.len() >= 2 && v.starts_with('"') && v.ends_with('"') { &v[1..v.len() - 1] } else { v };
                found = Some(v.to_string());
            }
        }
    }
    found
}

// ---------------------------------------------------------------- mimetypes

/// Python's mimetypes.guess_type: the built-in table overridden by /etc/mime.types.
pub struct MimeTypes {
    types: HashMap<String, String>,
}

const DEFAULT_TYPES: &[(&str, &str)] = &[
    (".js", "text/javascript"),
    (".mjs", "text/javascript"),
    (".json", "application/json"),
    (".webmanifest", "application/manifest+json"),
    (".css", "text/css"),
    (".html", "text/html"),
    (".htm", "text/html"),
    (".txt", "text/plain"),
    (".xml", "text/xml"),
    (".csv", "text/csv"),
    (".png", "image/png"),
    (".jpg", "image/jpeg"),
    (".jpeg", "image/jpeg"),
    (".gif", "image/gif"),
    (".ico", "image/vnd.microsoft.icon"),
    (".svg", "image/svg+xml"),
    (".webp", "image/webp"),
    (".wasm", "application/wasm"),
    (".pdf", "application/pdf"),
    (".zip", "application/zip"),
    (".tar", "application/x-tar"),
    (".mp4", "video/mp4"),
    (".wav", "audio/x-wav"),
    (".mp3", "audio/mpeg"),
];
const SUFFIX_MAP: &[(&str, &str)] =
    &[(".svgz", ".svg.gz"), (".tgz", ".tar.gz"), (".taz", ".tar.gz"), (".tz", ".tar.gz"), (".tbz2", ".tar.bz2"), (".txz", ".tar.xz")];
const ENCODINGS_MAP: &[(&str, &str)] = &[(".gz", "gzip"), (".Z", "compress"), (".bz2", "bzip2"), (".xz", "xz"), (".br", "br")];
const KNOWN_FILES: &[&str] = &[
    "/etc/mime.types",
    "/etc/httpd/mime.types",
    "/etc/httpd/conf/mime.types",
    "/etc/apache/mime.types",
    "/etc/apache2/mime.types",
    "/usr/local/etc/httpd/conf/mime.types",
    "/usr/local/lib/netscape/mime.types",
    "/usr/local/etc/mime.types",
];

/// posixpath.splitext
fn splitext(p: &str) -> (&str, &str) {
    let base_start = p.rfind('/').map_or(0, |i| i + 1);
    let name = &p[base_start..];
    let lead = name.len() - name.trim_start_matches('.').len();
    match name[lead..].rfind('.') {
        Some(i) => p.split_at(base_start + lead + i),
        None => (p, ""),
    }
}

impl MimeTypes {
    pub fn load() -> MimeTypes {
        let mut types: HashMap<String, String> = DEFAULT_TYPES.iter().map(|(e, t)| (e.to_string(), t.to_string())).collect();
        for file in KNOWN_FILES {
            if let Ok(text) = fs::read_to_string(file) {
                for line in text.lines() {
                    let words: Vec<&str> = line.split_whitespace().take_while(|w| !w.starts_with('#')).collect();
                    if let Some((t, exts)) = words.split_first() {
                        for e in exts {
                            types.insert(format!(".{e}"), t.to_string());
                        }
                    }
                }
            }
        }
        MimeTypes { types }
    }

    /// (type, encoding)
    pub fn guess(&self, path: &str) -> (Option<String>, Option<&'static str>) {
        let mut cur = path.to_string();
        loop {
            let (base, ext) = splitext(&cur);
            let lower = ext.to_lowercase();
            match SUFFIX_MAP.iter().find(|(from, _)| *from == lower) {
                Some((_, to)) => cur = format!("{base}{to}"),
                None => break,
            }
        }
        let (base, ext) = splitext(&cur);
        let (mut ext, mut encoding) = (ext.to_string(), None);
        // encodings_map is case sensitive
        if let Some((_, enc)) = ENCODINGS_MAP.iter().find(|(e, _)| *e == ext) {
            encoding = Some(*enc);
            ext = splitext(base).1.to_string();
        }
        (self.types.get(&ext.to_lowercase()).cloned(), encoding)
    }
}

// ---------------------------------------------------------------- static files

/// posixpath.split
fn split(p: &str) -> (String, String) {
    let i = p.rfind('/').map_or(0, |i| i + 1);
    let (head, tail) = p.split_at(i);
    let head = if !head.is_empty() && !head.bytes().all(|c| c == b'/') { head.trim_end_matches('/') } else { head };
    (head.to_string(), tail.to_string())
}

/// posixpath.join for two parts
fn join(a: &str, b: &str) -> String {
    if b.starts_with('/') {
        b.to_string()
    } else if a.is_empty() || a.ends_with('/') {
        format!("{a}{b}")
    } else {
        format!("{a}/{b}")
    }
}

/// posixpath.normpath for an absolute path
fn normpath(p: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for c in p.split('/') {
        match c {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            c => parts.push(c),
        }
    }
    format!("/{}", parts.join("/"))
}

/// posixpath._joinrealpath (strict=False)
fn join_realpath(mut path: String, rest: &str, seen: &mut HashMap<String, Option<String>>) -> (String, bool) {
    let mut rest = rest.to_string();
    if rest.starts_with('/') {
        rest = rest[1..].to_string();
        path = "/".into();
    }
    while !rest.is_empty() {
        let (name, tail) = match rest.find('/') {
            Some(i) => (rest[..i].to_string(), rest[i + 1..].to_string()),
            None => (rest.clone(), String::new()),
        };
        rest = tail;
        if name.is_empty() || name == "." {
            continue;
        }
        if name == ".." {
            if path.is_empty() {
                path = "..".into();
            } else {
                let (head, tail) = split(&path);
                path = head;
                if tail == ".." {
                    path = join(&join(&path, ".."), "..");
                }
            }
            continue;
        }
        let newpath = join(&path, &name);
        let is_link = fs::symlink_metadata(&newpath).is_ok_and(|m| m.file_type().is_symlink());
        if !is_link {
            path = newpath;
            continue;
        }
        if let Some(cached) = seen.get(&newpath) {
            match cached {
                Some(p) => {
                    path = p.clone();
                    continue;
                }
                None => return (join(&newpath, &rest), false),
            }
        }
        seen.insert(newpath.clone(), None);
        let target = fs::read_link(&newpath).map(|t| t.to_string_lossy().into_owned()).unwrap_or_default();
        let (p, ok) = join_realpath(path, &target, seen);
        if !ok {
            return (join(&p, &rest), false);
        }
        path = p;
        seen.insert(newpath, Some(path.clone()));
    }
    (path, true)
}

/// os.path.realpath(p, strict=False) for an absolute path (pathlib's Path.resolve()).
pub fn realpath(p: &str) -> String {
    let (path, _) = join_realpath(String::new(), p, &mut HashMap::new());
    normpath(&path)
}

/// aiohttp's static route: the file for `filename` (already unquoted) under `www` (a resolved
/// directory), or the status to answer: 403 for absolute names and directories, 404 for anything
/// missing or outside `www`.
pub fn static_file(www: &Path, filename: &str) -> Result<PathBuf, u16> {
    if filename.contains('\0') {
        return Err(404);
    }
    if filename.starts_with('/') {
        return Err(403);
    }
    // pathlib collapses empty and '.' components (not '..')
    let rel: Vec<&str> = filename.split('/').filter(|c| !c.is_empty() && *c != ".").collect();
    let mut joined = www.to_string_lossy().into_owned();
    for c in rel {
        joined = join(&joined, c);
    }
    let resolved = PathBuf::from(realpath(&joined));
    if !resolved.starts_with(www) {
        return Err(404);
    }
    match fs::metadata(&resolved) {
        Ok(m) if m.is_dir() => Err(403),
        Ok(m) if m.is_file() => Ok(resolved),
        _ => Err(404),
    }
}

/// The request headers FileResponse looks at (first value of each).
#[derive(Default, Debug, Clone)]
pub struct Conditional {
    pub head: bool,
    pub accept_encoding: Option<String>,
    pub if_match: Option<String>,
    pub if_none_match: Option<String>,
    pub if_modified_since: Option<String>,
    pub if_unmodified_since: Option<String>,
    pub if_range: Option<String>,
    pub range: Option<String>,
}

struct ETag {
    weak: bool,
    value: String,
}

/// aiohttp's If-Match / If-None-Match parsing (None for a missing or empty header).
fn etags(header: Option<&str>) -> Option<Vec<ETag>> {
    let h = header.filter(|h| !h.is_empty())?;
    if h == "*" {
        return Some(vec![ETag { weak: false, value: "*".into() }]);
    }
    let b = h.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    let etag_char = |c: u8| c == 0x21 || (0x23..=0x7e).contains(&c) || c >= 0x80;
    while i < b.len() {
        let mut j = i;
        let weak = b[j..].starts_with(b"W/");
        if weak {
            j += 2;
        }
        // (W/)?"etagc+" followed by \s*,\s* or the end; anything else stops the list
        let mut parsed = None;
        if b.get(j) == Some(&b'"') {
            let start = j + 1;
            let mut k = start;
            while k < b.len() && etag_char(b[k]) {
                k += 1;
            }
            if k > start && b.get(k) == Some(&b'"') {
                let value = String::from_utf8_lossy(&b[start..k]).into_owned();
                let mut m = k + 1;
                while m < b.len() && b[m].is_ascii_whitespace() {
                    m += 1;
                }
                if m < b.len() && b[m] == b',' {
                    m += 1;
                    while m < b.len() && b[m].is_ascii_whitespace() {
                        m += 1;
                    }
                    parsed = Some((value, m));
                } else if k + 1 == b.len() {
                    parsed = Some((value, k + 1));
                }
            }
        }
        match parsed {
            Some((value, next)) => {
                out.push(ETag { weak, value });
                i = next;
            }
            None => break,
        }
    }
    Some(out)
}

fn strong_match(etag: &str, tags: &[ETag]) -> bool {
    if tags.len() == 1 && tags[0].value == "*" {
        return true;
    }
    tags.iter().any(|t| !t.weak && t.value == etag)
}

fn http_date(h: Option<&str>) -> Option<f64> {
    let t = httpdate::parse_http_date(h?.trim()).ok()?;
    Some(t.duration_since(UNIX_EPOCH).ok()?.as_secs_f64())
}

fn format_http_date(secs: f64) -> String {
    httpdate::fmt_http_date(UNIX_EPOCH + Duration::from_secs(secs.ceil().max(0.0) as u64))
}

/// request.http_range: Ok(None) without a Range header, Err for a malformed one.
fn http_range(h: Option<&str>) -> Result<Option<(Option<i64>, Option<i64>)>, ()> {
    let Some(h) = h else { return Ok(None) };
    let spec = h.strip_prefix("bytes=").ok_or(())?;
    let (a, b) = spec.split_once('-').ok_or(())?;
    let num = |s: &str| -> Result<Option<i64>, ()> {
        if s.is_empty() {
            Ok(None)
        } else if s.bytes().all(|c| c.is_ascii_digit()) {
            s.parse().map(Some).map_err(|_| ())
        } else {
            Err(())
        }
    };
    let (mut start, mut end) = (num(a)?, num(b)?);
    if start.is_none() && end.is_some() {
        start = end.map(|e| -e);
        end = None;
    }
    if let (Some(s), Some(e)) = (start, end) {
        let e = e + 1;
        if s >= e {
            return Err(());
        }
        end = Some(e);
    }
    if start.is_none() && end.is_none() {
        return Err(());
    }
    Ok(Some((start, end)))
}

/// web.FileResponse(path).prepare(): 200/206 with the file (or the requested range), 304 when the
/// client's copy is current, 412/416 for failed preconditions and bad ranges.
pub fn file_response(path: &Path, req: &Conditional, mime: &MimeTypes) -> Reply {
    let mut filepath = path.to_path_buf();
    let mut gzip = false;
    if req.accept_encoding.as_deref().is_some_and(|a| a.contains("gzip")) {
        let mut gz = filepath.as_os_str().to_owned();
        gz.push(".gz");
        let gz = PathBuf::from(gz);
        if gz.is_file() {
            filepath = gz;
            gzip = true;
        }
    }
    let Ok(st) = fs::metadata(&filepath) else { return Reply::internal_error() };
    let mtime = st.modified().unwrap_or(UNIX_EPOCH);
    let since_epoch = mtime.duration_since(UNIX_EPOCH).unwrap_or_default();
    let mtime_ns = since_epoch.as_nanos();
    let mtime_s = since_epoch.as_secs_f64();
    let size = st.len() as i64;
    let etag = format!("{mtime_ns:x}-{size:x}");
    let last_modified = format_http_date(mtime_s);
    let mut reply = Reply { status: 200, headers: Vec::new(), body: Vec::new() };
    // aiohttp adds its default Content-Type to bodiless replies, except to HEAD requests
    let failed = |status: u16| {
        let mut r = Reply { status, headers: Vec::new(), body: Vec::new() };
        if !req.head {
            r.set("Content-Type", "application/octet-stream");
        }
        r
    };

    let if_match = etags(req.if_match.as_deref());
    if if_match.as_ref().is_some_and(|t| !strong_match(&etag, t)) {
        return failed(412).with("Content-Length", "0");
    }
    if let Some(t) = http_date(req.if_unmodified_since.as_deref()) {
        if if_match.is_none() && mtime_s > t {
            return failed(412).with("Content-Length", "0");
        }
    }
    let if_none_match = etags(req.if_none_match.as_deref());
    let not_modified = |mut r: Reply| {
        r.status = 304;
        r.set("ETag", &format!("\"{etag}\""));
        r.set("Last-Modified", &last_modified);
        r
    };
    if if_none_match.as_ref().is_some_and(|t| strong_match(&etag, t)) {
        return not_modified(reply);
    }
    if let Some(t) = http_date(req.if_modified_since.as_deref()) {
        if if_none_match.is_none() && mtime_s <= t {
            return not_modified(reply);
        }
    }

    let (ct, encoding) = mime.guess(&filepath.to_string_lossy());
    let ct = ct.unwrap_or_else(|| "application/octet-stream".into());
    let mut start: Option<i64> = None;
    let mut count = size;
    let if_range = http_date(req.if_range.as_deref());
    if if_range.is_none_or(|t| mtime_s <= t) {
        let unsatisfiable = || failed(416).with("Content-Range", &format!("bytes */{size}"));
        match http_range(req.range.as_deref()) {
            Err(()) => return unsatisfiable(),
            Ok(None) => {}
            Ok(Some((s, e))) => {
                let mut s = s.unwrap_or(0);
                if s < 0 && e.is_none() {
                    s = (s + size).max(0);
                    count = size - s;
                } else {
                    count = e.unwrap_or(size).min(size) - s;
                }
                if s >= size {
                    return unsatisfiable();
                }
                start = Some(s);
                reply.status = 206;
            }
        }
    }
    reply.set("Content-Type", &ct);
    if let Some(enc) = encoding {
        reply.set("Content-Encoding", enc);
    }
    if gzip {
        reply.set("Vary", "Accept-Encoding");
    }
    reply.set("ETag", &format!("\"{etag}\""));
    reply.set("Last-Modified", &last_modified);
    reply.set("Content-Length", &count.to_string());
    reply.set("Accept-Ranges", "bytes");
    if reply.status == 206 {
        let s = start.unwrap_or(0);
        reply.set("Content-Range", &format!("bytes {}-{}/{}", s, s + count - 1, size));
    }
    if count > 0 && !req.head {
        let Ok(data) = fs::read(&filepath) else { return Reply::internal_error() };
        let s = start.unwrap_or(0).max(0) as usize;
        let e = (s + count.max(0) as usize).min(data.len());
        reply.body = data[s.min(e)..e].to_vec();
    }
    reply
}

/// ament_index_python's get_package_share_directory: <prefix>/share/<package> for the first
/// AMENT_PREFIX_PATH entry that has the package.
pub fn package_share(package: &str) -> Option<PathBuf> {
    let prefixes = std::env::var("AMENT_PREFIX_PATH").ok()?;
    prefixes.split(':').filter(|p| !p.is_empty()).find_map(|prefix| {
        let marker = Path::new(prefix).join("share/ament_index/resource_index/packages").join(package);
        marker.exists().then(|| Path::new(prefix).join("share").join(package))
    })
}

/// time.time()
pub fn wall_time() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_unquoting_like_yarl() {
        let cases = [
            ("/a%2Fb", "/a/b"),
            ("/a%20b", "/a b"),
            ("/a+b", "/a+b"),
            ("/a%2Bb", "/a+b"),
            ("/%61pi/state", "/api/state"),
            ("/static/%2e%2e/x", "/static/../x"),
            ("/a%C3%A9", "/aé"),
            ("/a%ZZ", "/a%ZZ"),
            ("/a%00b", "/a\0b"),
            ("/a%25b", "/a%b"),
            ("/a%FFb", "/a%FFb"),
            ("/a%C3", "/a%C3"),
            ("/a%C3%28", "/a%C3("),
            ("/a%E2%82%AC%", "/a€%"),
            ("/a%e2%82x", "/a%e2%82x"),
        ];
        for (raw, want) in cases {
            assert_eq!(unquote_path(raw), want, "{raw}");
        }
    }

    #[test]
    fn form_decoding_like_parse_qsl() {
        assert_eq!(parse_qsl("password=a+b%21%C3%A9&x"), vec![("password".into(), "a b!é".into()), ("x".into(), "".into())]);
        assert_eq!(parse_qsl("password=%FF%zz&&password=2"), vec![("password".into(), "\u{fffd}%zz".into()), ("password".into(), "2".into())]);
        assert_eq!(content_type(Some("Application/X-WWW-Form-Urlencoded; charset=UTF-8")), "application/x-www-form-urlencoded");
        assert_eq!(content_type(None), "application/octet-stream");
        assert_eq!(charset(Some("text/plain; charset=\"latin-1\"")).as_deref(), Some("latin-1"));
        assert_eq!(cookie(Some("a=1; roomba_session=\"tok\"; b=2"), "roomba_session").as_deref(), Some("tok"));
        assert_eq!(cookie(Some("roomba_session=x; roomba_session=y"), "roomba_session").as_deref(), Some("y"));
        assert_eq!(cookie(None, "roomba_session"), None);
    }

    #[test]
    fn mimetypes_like_python() {
        let m = MimeTypes { types: DEFAULT_TYPES.iter().map(|(e, t)| (e.to_string(), t.to_string())).collect() };
        assert_eq!(m.guess("/x/app.js"), (Some("text/javascript".into()), None));
        assert_eq!(m.guess("/x/APP.PNG"), (Some("image/png".into()), None));
        assert_eq!(m.guess("/x/app.js.gz"), (Some("text/javascript".into()), Some("gzip")));
        assert_eq!(m.guess("/x/.hidden"), (None, None));
        assert_eq!(m.guess("/x/a.tgz"), (Some("application/x-tar".into()), Some("gzip")));
        assert_eq!(splitext("/a.b/c"), ("/a.b/c", ""));
        assert_eq!(splitext("/a/..x.y"), ("/a/..x", ".y"));
    }

    #[test]
    fn static_files_stay_inside_www() {
        let root = std::env::temp_dir().join(format!("roomba_web_static_test_{}", std::process::id()));
        let www = root.join("www");
        fs::create_dir_all(www.join("sub")).unwrap();
        fs::write(www.join("app.js"), "x").unwrap();
        fs::write(root.join("secret"), "s").unwrap();
        std::os::unix::fs::symlink(root.join("secret"), www.join("link")).unwrap();
        std::os::unix::fs::symlink(www.join("app.js"), www.join("inside")).unwrap();
        let www = PathBuf::from(realpath(&www.to_string_lossy()));
        assert_eq!(static_file(&www, "app.js"), Ok(www.join("app.js")));
        assert_eq!(static_file(&www, "sub/../app.js"), Ok(www.join("app.js")));
        assert_eq!(static_file(&www, "./app.js"), Ok(www.join("app.js")));
        assert_eq!(static_file(&www, "inside"), Ok(www.join("app.js")));
        assert_eq!(static_file(&www, ""), Err(403));
        assert_eq!(static_file(&www, "sub"), Err(403));
        assert_eq!(static_file(&www, "app.js/.."), Err(403));
        assert_eq!(static_file(&www, "/etc/passwd"), Err(403));
        assert_eq!(static_file(&www, "../secret"), Err(404));
        assert_eq!(static_file(&www, "sub/../../secret"), Err(404));
        assert_eq!(static_file(&www, "link"), Err(404));
        assert_eq!(static_file(&www, "missing"), Err(404));
        assert_eq!(static_file(&www, "a\0b"), Err(404));
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn file_response_conditionals_and_ranges() {
        let root = std::env::temp_dir().join(format!("roomba_web_file_test_{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let f = root.join("a.css");
        fs::write(&f, "0123456789").unwrap();
        let m = MimeTypes { types: DEFAULT_TYPES.iter().map(|(e, t)| (e.to_string(), t.to_string())).collect() };
        let full = file_response(&f, &Conditional::default(), &m);
        assert_eq!((full.status, full.body.as_slice(), full.header("Content-Type")), (200, b"0123456789".as_slice(), Some("text/css")));
        let etag = full.header("ETag").unwrap().to_string();
        let req = |c: Conditional| file_response(&f, &c, &m);
        let r = req(Conditional { if_none_match: Some(etag.clone()), ..Default::default() });
        assert_eq!((r.status, r.body.len(), r.header("Content-Length")), (304, 0, None));
        let r = req(Conditional { if_none_match: Some(format!("W/{etag}")), ..Default::default() });
        assert_eq!(r.status, 200);
        let r = req(Conditional { if_match: Some("\"nope\"".into()), ..Default::default() });
        assert_eq!(r.status, 412);
        let r = req(Conditional { if_modified_since: full.header("Last-Modified").map(String::from), ..Default::default() });
        assert_eq!(r.status, 304);
        let r = req(Conditional { range: Some("bytes=2-4".into()), ..Default::default() });
        assert_eq!((r.status, r.body.as_slice(), r.header("Content-Range")), (206, b"234".as_slice(), Some("bytes 2-4/10")));
        let r = req(Conditional { range: Some("bytes=-3".into()), ..Default::default() });
        assert_eq!((r.status, r.body.as_slice()), (206, b"789".as_slice()));
        let r = req(Conditional { range: Some("bytes=20-".into()), ..Default::default() });
        assert_eq!((r.status, r.header("Content-Range")), (416, Some("bytes */10")));
        let r = req(Conditional { range: Some("items=1-2".into()), ..Default::default() });
        assert_eq!(r.status, 416);
        let r = req(Conditional { head: true, ..Default::default() });
        assert_eq!((r.status, r.body.len(), r.header("Content-Length")), (200, 0, Some("10")));
        fs::remove_dir_all(&root).unwrap();
    }
}
