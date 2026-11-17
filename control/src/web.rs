//! HTTP: login, the page, /api/status and the actions. Every action is a POST from this page's own
//! origin (Origin header == http://<Host>) with a valid session cookie.

use std::io::Read;
use std::net::{IpAddr, Ipv4Addr};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use tiny_http::{Header, Request, Response};

use crate::auth::{self, COOKIE};
use crate::robot;
use crate::ros;
use crate::status;
use crate::App;

const INDEX_HTML: &str = include_str!("../static/index.html");
const LOGIN_HTML: &str = include_str!("../static/login.html");
const APP_JS: &str = include_str!("../static/app.js");
const STYLE_CSS: &str = include_str!("../static/style.css");
const MANIFEST: &str = include_str!("../static/manifest.webmanifest");
const ICON_192: &[u8] = include_bytes!("../static/icon-192.png");
const ICON_512: &[u8] = include_bytes!("../static/icon-512.png");

pub struct Reply {
    status: u16,
    ctype: &'static str,
    body: Vec<u8>,
    headers: Vec<(&'static str, String)>,
}

impl Reply {
    fn new(status: u16, ctype: &'static str, body: impl Into<Vec<u8>>) -> Reply {
        Reply { status, ctype, body: body.into(), headers: Vec::new() }
    }

    fn json(status: u16, v: &Value) -> Reply {
        Reply::new(status, "application/json", v.to_string())
    }

    fn result(r: Result<String, String>) -> Reply {
        match r {
            Ok(m) => Reply::json(202, &json!({"ok": true, "message": m})),
            Err(m) => Reply::json(409, &json!({"ok": false, "message": m})),
        }
    }

    fn redirect(location: &str) -> Reply {
        Reply::new(303, "text/plain", "").with("Location", location)
    }

    fn with(mut self, name: &'static str, value: &str) -> Reply {
        self.headers.push((name, value.to_string()));
        self
    }
}

fn header<'a>(rq: &'a Request, name: &str) -> Option<&'a str> {
    rq.headers().iter().find(|h| h.field.as_str().as_str().eq_ignore_ascii_case(name)).map(|h| h.value.as_str())
}

/// The Origin header must be this page's own origin: blocks requests from other sites and from
/// the robot web app on another port of the same host.
pub fn same_origin(host: Option<&str>, origin: Option<&str>) -> bool {
    match (host, origin) {
        (Some(h), Some(o)) if !h.is_empty() => {
            o.eq_ignore_ascii_case(&format!("http://{h}")) || o.eq_ignore_ascii_case(&format!("https://{h}"))
        }
        _ => false,
    }
}

fn client_ip(rq: &Request) -> IpAddr {
    rq.remote_addr().map_or(IpAddr::V4(Ipv4Addr::UNSPECIFIED), |a| a.ip().to_canonical())
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

fn login_page(status: u16, error: Option<&str>) -> Reply {
    let err = error.map_or(String::new(), |e| format!("<p class=\"error\" role=\"alert\">{}</p>", html_escape(e)));
    Reply::new(status, "text/html; charset=utf-8", LOGIN_HTML.replace("<!--ERROR-->", &err))
}

fn cookie_header(token: &str, max_age: u64) -> String {
    format!("{COOKIE}={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age={max_age}")
}

fn read_body(rq: &mut Request, limit: u64) -> Vec<u8> {
    let mut body = Vec::new();
    let _ = rq.as_reader().take(limit).read_to_end(&mut body);
    body
}

pub fn handle(app: &Arc<App>, mut rq: Request) {
    let host = header(&rq, "Host").map(str::to_string);
    let reply = route(app, &mut rq, host.as_deref());
    let frame_src = status::clean_host(host.as_deref())
        .map_or("http:".to_string(), |h| format!("http://{}:{}", status::host_name(&h), app.cfg.robot_web_port));
    let csp = format!(
        "default-src 'self'; img-src 'self' data:; frame-src {frame_src}; frame-ancestors 'none'; \
         form-action 'self'; base-uri 'none'; object-src 'none'"
    );
    let mut resp = Response::from_data(reply.body).with_status_code(reply.status);
    let mut add = |name: &str, value: &str| {
        if let Ok(h) = Header::from_bytes(name.as_bytes(), value.as_bytes()) {
            resp.add_header(h);
        }
    };
    add("Content-Type", reply.ctype);
    add("Cache-Control", "no-cache, no-store");
    add("X-Content-Type-Options", "nosniff");
    add("X-Frame-Options", "DENY");
    // Not no-referrer: with it, browsers send "Origin: null" on form POSTs (login, logout), which
    // the origin check has to refuse.
    add("Referrer-Policy", "same-origin");
    add("Content-Security-Policy", &csp);
    for (name, value) in &reply.headers {
        add(name, value);
    }
    let _ = rq.respond(resp);
}

fn route(app: &Arc<App>, rq: &mut Request, host: Option<&str>) -> Reply {
    let url = rq.url().to_string();
    let path = url.split('?').next().unwrap_or("/");
    let method = rq.method().as_str().to_ascii_uppercase();
    let get = method == "GET" || method == "HEAD";
    let post = method == "POST";
    let now = Instant::now();
    let token = auth::cookie(header(rq, "Cookie"), COOKIE).map(str::to_string);
    let authed = app.sessions.lock().unwrap_or_else(|e| e.into_inner()).valid(token.as_deref(), now);
    let origin_ok = same_origin(host, header(rq, "Origin"));

    let static_file = |ctype: &'static str, body: &[u8]| Reply::new(200, ctype, body.to_vec());
    match (path, get, post) {
        ("/app.js", true, _) => static_file("text/javascript; charset=utf-8", APP_JS.as_bytes()),
        ("/style.css", true, _) => static_file("text/css; charset=utf-8", STYLE_CSS.as_bytes()),
        ("/manifest.webmanifest", true, _) => static_file("application/manifest+json", MANIFEST.as_bytes()),
        ("/icon-192.png" | "/favicon.ico", true, _) => static_file("image/png", ICON_192),
        ("/icon-512.png", true, _) => static_file("image/png", ICON_512),
        ("/login", true, _) if authed => Reply::redirect("/"),
        ("/login", true, _) => login_page(200, None),
        ("/login", _, true) => login(app, rq, origin_ok),
        ("/logout", _, true) => {
            if origin_ok {
                app.sessions.lock().unwrap_or_else(|e| e.into_inner()).revoke(token.as_deref());
            }
            Reply::redirect("/login").with("Set-Cookie", &cookie_header("", 0))
        }
        ("/", true, _) if authed => Reply::new(200, "text/html; charset=utf-8", INDEX_HTML),
        ("/", true, _) => Reply::redirect("/login"),
        ("/api/status", true, _) if authed => Reply::json(200, &status::status(app, host)),
        ("/api/status", true, _) => Reply::json(401, &json!({"ok": false, "message": "log in"})),
        (p, _, true) if p.starts_with("/api/tone/") || p == "/api/vacuum/arm" || p == "/api/vacuum/disarm" || p == "/api/battery/new_pack" => {
            if !authed {
                return Reply::json(401, &json!({"ok": false, "message": "log in"}));
            }
            if !origin_ok {
                return Reply::json(403, &json!({"ok": false, "message": "blocked: not from this page (Origin check)"}));
            }
            eprintln!("{}: {p}", client_ip(rq));
            Reply::result(match p {
                "/api/vacuum/arm" => ros::request_vacuum(app, true),
                "/api/vacuum/disarm" => ros::request_vacuum(app, false),
                "/api/battery/new_pack" => ros::request_pack(app),
                _ => ros::request_tone(app, &p["/api/tone/".len()..]),
            })
        }
        ("/api/start" | "/api/stop" | "/api/shutdown" | "/api/comms" | "/api/sound/mute" | "/api/sound/unmute", _, true) => {
            if !authed {
                return Reply::json(401, &json!({"ok": false, "message": "log in"}));
            }
            if !origin_ok {
                return Reply::json(403, &json!({"ok": false, "message": "blocked: not from this page (Origin check)"}));
            }
            eprintln!("{}: {path}", client_ip(rq));
            Reply::result(match path {
                "/api/start" => robot::request_start(app),
                "/api/stop" => robot::request_stop(app),
                "/api/shutdown" => robot::request_shutdown(app),
                "/api/comms" => robot::request_check(app),
                p => robot::set_sound(app, p == "/api/sound/mute"),
            })
        }
        ("/" | "/login" | "/logout" | "/api/status" | "/api/start" | "/api/stop" | "/api/shutdown" | "/api/comms", _, _)
        | ("/api/sound/mute" | "/api/sound/unmute", _, _)
        | ("/app.js" | "/style.css" | "/manifest.webmanifest" | "/icon-192.png" | "/icon-512.png" | "/favicon.ico", _, _) => {
            Reply::new(405, "text/plain", "method not allowed")
        }
        _ => Reply::new(404, "text/plain", "not found"),
    }
}

fn limiter(app: &App) -> std::sync::MutexGuard<'_, auth::Limiter> {
    app.limiter.lock().unwrap_or_else(|e| e.into_inner())
}

fn login(app: &Arc<App>, rq: &mut Request, origin_ok: bool) -> Reply {
    if !origin_ok {
        return login_page(403, Some("Blocked by the origin check: reload this page and try again."));
    }
    let ip = client_ip(rq);
    let lock = limiter;
    if let Some(wait) = lock(app).locked(ip, Instant::now()) {
        return login_page(429, Some(&format!("Too many wrong passwords: try again in {wait} s.")));
    }
    let body = read_body(rq, 4096);
    let password = auth::form_field(&body, "password").unwrap_or_default();
    let deadline = Instant::now() + Duration::from_secs(5);
    let _gate = loop {
        match app.login_gate.try_lock() {
            Ok(g) => break g,
            Err(std::sync::TryLockError::Poisoned(e)) => break e.into_inner(),
            Err(_) if Instant::now() > deadline => return login_page(429, Some("Busy: try again in a moment.")),
            Err(_) => std::thread::sleep(Duration::from_millis(50)),
        }
    };
    match auth::check_password(&app.cfg.auth_file, &password) {
        Ok(true) => {
            lock(app).success(ip);
            let mut sessions = app.sessions.lock().unwrap_or_else(|e| e.into_inner());
            let token = sessions.create(Instant::now());
            eprintln!("{ip}: logged in");
            Reply::redirect("/").with("Set-Cookie", &cookie_header(&token, sessions.ttl().as_secs()))
        }
        Ok(false) => {
            lock(app).failure(ip, Instant::now());
            eprintln!("{ip}: wrong password");
            std::thread::sleep(Duration::from_secs(1));
            login_page(401, Some("Wrong password."))
        }
        Err(e) => {
            eprintln!("login: {e}");
            std::thread::sleep(Duration::from_secs(1));
            login_page(500, Some("Cannot read the password file: set the password with set_password."))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_must_match_the_host_exactly() {
        assert!(same_origin(Some("robot.local"), Some("http://robot.local")));
        assert!(same_origin(Some("10.0.0.5:18081"), Some("http://10.0.0.5:18081")));
        assert!(!same_origin(Some("10.0.0.5"), Some("http://10.0.0.5:8080")), "the robot web app's port");
        assert!(!same_origin(Some("robot.local"), Some("http://evil.example")));
        assert!(!same_origin(Some("robot.local"), Some("null")));
        assert!(!same_origin(Some("robot.local"), None));
        assert!(!same_origin(None, Some("http://")));
    }
}
