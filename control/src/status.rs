//! /api/status: the robot stack, the current action, the Roomba link and the Jetson's health.

use std::net::{SocketAddr, TcpStream};
use std::path::Path;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::comms::Link;
use crate::robot::{self, CheckVia, Op, Power};
use crate::App;

/// A COMMS CHECK result counts for this long; after that the link shows as just "connected".
const LINK_RESULT_TTL: Duration = Duration::from_secs(10 * 60);

pub fn cpu_temp_c() -> Option<f64> {
    let zones: Vec<_> = std::fs::read_dir("/sys/class/thermal").ok()?.flatten().map(|e| e.path()).collect();
    let read = |p: &Path| std::fs::read_to_string(p.join("temp")).ok()?.trim().parse::<f64>().ok().map(|m| m / 1000.0);
    let kind = |p: &Path| std::fs::read_to_string(p.join("type")).unwrap_or_default().trim().to_string();
    zones.iter().find(|z| kind(z) == "cpu-thermal").and_then(|z| read(z)).or_else(|| zones.iter().find_map(|z| read(z)))
}

/// /proc/net/wireless: interface, link quality and signal level (dBm) of the first interface.
pub fn parse_wireless(text: &str) -> Option<(String, f64, f64)> {
    let line = text.lines().skip(2).find(|l| l.contains(':'))?;
    let (iface, rest) = line.split_once(':')?;
    let mut f = rest.split_whitespace().skip(1).map(|v| v.trim_end_matches('.').parse::<f64>());
    let link = f.next()?.ok()?;
    let level = f.next()?.ok()?;
    Some((iface.trim().to_string(), link, level))
}

fn uptime_s() -> Option<f64> {
    std::fs::read_to_string("/proc/uptime").ok()?.split_whitespace().next()?.parse().ok()
}

fn disk(path: &Path) -> Option<(u64, u64)> {
    let c = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).ok()?;
    // SAFETY: statvfs fills the struct for a NUL-terminated path.
    unsafe {
        let mut s: libc::statvfs = std::mem::zeroed();
        if libc::statvfs(c.as_ptr(), &mut s) != 0 {
            return None;
        }
        Some((s.f_bavail as u64 * s.f_frsize as u64, s.f_blocks as u64 * s.f_frsize as u64))
    }
}

fn usb_present(vid: &str, pid: &str) -> bool {
    let Ok(dir) = std::fs::read_dir("/sys/bus/usb/devices") else { return false };
    let attr = |p: &Path, a: &str| std::fs::read_to_string(p.join(a)).map(|s| s.trim().to_ascii_lowercase()).unwrap_or_default();
    dir.flatten().any(|d| attr(&d.path(), "idVendor") == vid && attr(&d.path(), "idProduct") == pid)
}

fn port_open(port: u16) -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok()
}

/// The last `n` lines of the robot log (without terminal colour codes), and its age in seconds.
pub fn log_tail(path: &Path, n: usize) -> (Vec<String>, Option<u64>) {
    use std::io::{Read, Seek, SeekFrom};
    let Ok(mut f) = std::fs::File::open(path) else { return (Vec::new(), None) };
    let age = f.metadata().ok().and_then(|m| m.modified().ok()).and_then(|t| t.elapsed().ok()).map(|d| d.as_secs());
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let _ = f.seek(SeekFrom::Start(len.saturating_sub(32 * 1024)));
    let mut buf = Vec::new();
    let _ = f.read_to_end(&mut buf);
    let text = String::from_utf8_lossy(&buf);
    let lines: Vec<String> = text.lines().map(clean_line).filter(|l| !l.trim().is_empty()).collect();
    (lines[lines.len().saturating_sub(n)..].to_vec(), age)
}

fn clean_line(line: &str) -> String {
    let mut out = String::new();
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            // CSI ... final byte
            if chars.peek() == Some(&'[') {
                chars.next();
                for d in chars.by_ref() {
                    if ('@'..='~').contains(&d) {
                        break;
                    }
                }
            }
        } else if c == '\t' {
            out.push(' ');
        } else if !c.is_control() {
            out.push(c);
        }
        if out.chars().count() >= 400 {
            out.push('…');
            break;
        }
    }
    out
}

/// The Host header as given (host[:port]) if it is a plain host name / address, else None.
pub fn clean_host(host: Option<&str>) -> Option<String> {
    let h = host?.trim();
    let ok = !h.is_empty() && h.len() <= 255 && h.chars().all(|c| c.is_ascii_alphanumeric() || ".-:[]_".contains(c));
    ok.then(|| h.to_string())
}

/// host[:port] -> host (IPv6 literals keep their brackets).
pub fn host_name(host: &str) -> &str {
    if host.starts_with('[') {
        return host.find(']').map_or(host, |i| &host[..=i]);
    }
    host.rsplit_once(':').map_or(host, |(h, _)| h)
}

pub fn status(app: &App, host: Option<&str>) -> Value {
    let cfg = &app.cfg;
    let launches = robot::launches(app);
    let port = app.port_config();
    let port_present = port.port.exists();
    let web_up = launches.any() && port_open(cfg.robot_web_port);
    let (log_lines, log_age) = log_tail(&cfg.log_file, 10);
    let wifi = std::fs::read_to_string("/proc/net/wireless").ok().as_deref().and_then(parse_wireless);
    let host = clean_host(host);
    let now = Instant::now();

    let ctl = robot::lock(app);
    let state = match &ctl.busy {
        Some(b) if b.op == Op::Start => "starting",
        Some(b) if b.stopping => "stopping",
        _ if launches.any() => "running",
        _ => "stopped",
    };
    let busy = ctl.busy.as_ref().map(|b| {
        json!({"op": b.op.as_str(), "message": b.message, "elapsed_s": b.since.elapsed().as_secs(), "then_poweroff": b.then_poweroff})
    });
    let last = ctl.last.as_ref().map(|d| json!({"op": d.op.as_str(), "ok": d.ok, "message": d.message, "ago_s": now.duration_since(d.at).as_secs()}));
    let result = ctl.link.as_ref().filter(|r| now.duration_since(r.at) < LINK_RESULT_TTL);
    let link_state = if ctl.checking.is_some() {
        "checking"
    } else if !port_present {
        "missing"
    } else {
        match result.map(|r| r.link) {
            Some(Link::Responding) => "responding",
            Some(Link::NotResponding) => "not_responding",
            _ => "connected",
        }
    };
    let via = |v: CheckVia| if v == CheckVia::Ros { "ros" } else { "serial" };
    json!({
        "page_url": host.as_ref().map(|h| format!("http://{h}/")),
        "robot_url": host.as_ref().map(|h| format!("http://{}:{}/", host_name(h), cfg.robot_web_port)),
        "robot": {
            "state": state,
            "pid": launches.ours,
            "external": !launches.others.is_empty(),
            "web_up": web_up,
            "web_port": cfg.robot_web_port,
        },
        "busy": busy,
        "last": last,
        "power": match ctl.power { Power::On => "on", Power::PoweringOff => "powering_off", Power::Off => "off" },
        "dry_run": cfg.dry_run_poweroff,
        "roomba": {
            "port": port.port.display().to_string(),
            "port_present": port_present,
            "link": link_state,
            "checking": ctl.checking.map(|(v, t)| json!({"via": via(v), "elapsed_s": t.elapsed().as_secs()})),
            "result": ctl.link.as_ref().map(|r| json!({
                "link": r.link.as_str(), "message": r.message, "via": via(r.via), "ago_s": now.duration_since(r.at).as_secs(),
            })),
        },
        "camera": usb_present(&cfg.camera_usb.0, &cfg.camera_usb.1),
        "cpu_temp_c": cpu_temp_c(),
        "wifi": wifi.map(|(iface, link, level)| json!({"iface": iface, "link": link, "level_dbm": level})),
        "uptime_s": uptime_s().map(|u| u as u64),
        "disk": disk(&cfg.disk_path).map(|(free, total)| json!({"free_bytes": free, "total_bytes": total})),
        "log": {"lines": log_lines, "age_s": log_age},
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_wifi_signal() {
        let text = "Inter-| sta-|   Quality        |   Discarded packets               | Missed | WE\n face | tus | link level noise |  nwid  crypt   frag  retry   misc | beacon | 22\nwlP1p1s0: 0000   67.  -43.  -256        0      0      0      0      0        0\n";
        assert_eq!(parse_wireless(text), Some(("wlP1p1s0".into(), 67.0, -43.0)));
        assert_eq!(parse_wireless(&text[..text.find("wlP1").unwrap()]), None);
    }

    #[test]
    fn hosts_from_the_host_header() {
        assert_eq!(clean_host(Some("robot.local")).as_deref(), Some("robot.local"));
        assert_eq!(clean_host(Some("bad host\"; x")), None);
        assert_eq!(host_name("robot.local:8000"), "robot.local");
        assert_eq!(host_name("[fe80::1]:80"), "[fe80::1]");
        assert_eq!(host_name("[fe80::1]"), "[fe80::1]");
    }

    #[test]
    fn log_lines_lose_colour_codes() {
        assert_eq!(clean_line("\u{1b}[33m[WARN] x\u{1b}[0m\tdone"), "[WARN] x done");
    }
}
